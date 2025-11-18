import nvdiffrast.torch as dr
import torch
import trimesh
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
from utils.graphics_utils import focal2fov
import numpy as np
import os
from argparse import ArgumentParser
import json, math
import cv2
from tqdm import tqdm

def get_camera_params(camera_intr, resolution=1):
    """获取相机参数"""
    if camera_intr.model == "SIMPLE_PINHOLE":
        focal_length_x = camera_intr.params[0]
        focal_length_y = camera_intr.params[0]
    elif camera_intr.model == "PINHOLE":
        focal_length_x = camera_intr.params[0]
        focal_length_y = camera_intr.params[1]
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
    width = camera_intr.width // resolution
    height = camera_intr.height // resolution
    
    return focal_length_x, focal_length_y, width, height

def read_extrinsics(args: ArgumentParser):
    train_test_split = None
    split_file = os.path.join(args.model, "split.json")
    if os.path.exists(split_file):
        train_test_split = json.load(open(split_file))
        test_list = train_test_split["test"]
    
    extrinsics = read_extrinsics_binary(os.path.join(args.model, "sparse/0/images.bin"))
    
    if train_test_split is not None:
        test_extrinsics_idx = [extrinsic for extrinsic in extrinsics if extrinsics[extrinsic].name.split(".")[0] in test_list]
        test_extrinsics = {extrinsic: extrinsics[extrinsic] for extrinsic in test_extrinsics_idx}
    else:
        test_extrinsics = extrinsics
    
    return test_extrinsics

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def getTranspose(qvec, tvec):
    rotation_matrix = qvec2rotmat(qvec)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = tvec
    return extrinsic_matrix

def get_projection_matrix(fx, fy, cx, cy, width, height, znear=0.01, zfar=100.0):
    """构建OpenGL风格的投影矩阵（NDC坐标系）- 修正版本"""
    projection = torch.zeros(4, 4)
    
    # OpenGL NDC: x,y in [-1,1], z in [-1,1]
    projection[0, 0] = 2.0 * fx / width
    projection[1, 1] = -2.0 * fy / height  # 注意这里是负号，因为OpenGL的y轴向上，图像y轴向下
    projection[0, 2] = (2.0 * cx - width) / width
    projection[1, 2] = (2.0 * cy - height) / height
    projection[2, 2] = -(zfar + znear) / (zfar - znear)
    projection[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    projection[3, 2] = -1.0
    
    return projection

def render_mesh_nvdiffrast(glctx, vertices, faces, mvp, resolution, vertex_colors=None):
    """使用nvdiffrast渲染mesh"""
    # 转换为齐次坐标
    vertices_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
    
    # 应用MVP变换
    vertices_clip = vertices_homo @ mvp.T
    
    # 光栅化
    rast_out, rast_out_db = dr.rasterize(glctx, vertices_clip[None, ...], faces, resolution=resolution)
    
    # 如果没有提供顶点颜色，使用白色
    if vertex_colors is None:
        vertex_colors = torch.ones_like(vertices)
    
    # 插值顶点属性（颜色）
    color, _ = dr.interpolate(vertex_colors[None, ...], rast_out, faces)
    
    # 获取mask
    mask = rast_out[..., 3:4] > 0
    
    # 应用mask
    color = color * mask
    
    # 添加抗锯齿
    color = dr.antialias(color, rast_out, vertices_clip[None, ...], faces)
    
    return color[0]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--mesh", type=str)
    
    args = parser.parse_args()
    
    # 初始化nvdiffrast context
    device = torch.device('cuda:0')
    glctx = dr.RasterizeGLContext()
    
    # 加载mesh
    print(f"Loading mesh from: {args.mesh}")
    mesh = trimesh.load(args.mesh)
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    faces = torch.from_numpy(mesh.faces).int().to(device)
    
    # 尝试加载顶点颜色，如果没有则使用白色
    if hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = torch.from_numpy(mesh.visual.vertex_colors[:, :3]).float().to(device) / 255.0
        print("Using mesh vertex colors")
    else:
        vertex_colors = torch.ones_like(vertices) * 0.8  # 使用浅灰色
        print("No vertex colors found, using default gray")
    
    # 读取相机内参
    intrinsics = read_intrinsics_binary(os.path.join(args.model, "sparse/0/cameras.bin"))
    fx, fy, width, height = get_camera_params(intrinsics[1], resolution=1)
    
    print(f"Camera intrinsics: fx={fx}, fy={fy}, width={width}, height={height}")
    
    # 调整分辨率
    render_width = 1920
    render_height = 1440
    resolution = [render_height, render_width]
    
    # 根据分辨率缩放调整焦距
    fx_scaled = fx * (render_width / width)
    fy_scaled = fy * (render_height / height)
    
    # 计算mesh的边界框，用于设置合适的near/far平面
    vertices_np = mesh.vertices
    bbox_min = vertices_np.min(axis=0)
    bbox_max = vertices_np.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    
    print(f"Mesh bounding box: min={bbox_min}, max={bbox_max}, size={bbox_size}")
    
    znear = 0.01
    zfar = 100.0
    
    print(f"Using znear={znear}, zfar={zfar}")
    
    # 构建投影矩阵
    projection_matrix = get_projection_matrix(
        fx_scaled, fy_scaled, 
        render_width/2, render_height/2, 
        render_width, render_height, 
        znear, zfar
    )
    projection_matrix = projection_matrix.to(device)
    
    # 创建输出目录
    output_dir = "view_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving rendered images to: {os.path.abspath(output_dir)}")
    
    # 读取外参并渲染
    test_extrinsics = read_extrinsics(args)
    
    for extrinsic_idx in tqdm(test_extrinsics, desc="Rendering views"):
        extrinsic = test_extrinsics[extrinsic_idx]
        
        # COLMAP外参：R和t使得 x_cam = R * x_world + t
        # 即这是world-to-camera变换
        camera_pose_w2c = getTranspose(extrinsic.qvec, extrinsic.tvec)
        
        # COLMAP坐标系：X右，Y下，Z前
        # OpenGL坐标系：X右，Y上，Z后
        # 需要的变换：Y取反，Z取反
        colmap_to_opengl = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # 先转换到OpenGL坐标系，然后得到view矩阵
        view_matrix = colmap_to_opengl @ camera_pose_w2c
        view_matrix = torch.from_numpy(view_matrix).float().to(device)
        
        # 计算MVP矩阵
        mvp = projection_matrix @ view_matrix
        
        # 渲染
        with torch.no_grad():
            color = render_mesh_nvdiffrast(glctx, vertices, faces, mvp, resolution, vertex_colors)
        
        # 转换为图像
        color_np = (torch.clamp(color, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        color_np = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        color_np = cv2.resize(color_np, (color_np.shape[1]//2, color_np.shape[0]//2))
        
        # 保存
        output_path = os.path.join(output_dir, extrinsic.name)
        cv2.imwrite(output_path, color_np)
    
    print(f"\nRendering complete! {len(test_extrinsics)} images saved to {os.path.abspath(output_dir)}")

# Example:
# (1)use ply:
# python mesh_viewer_nvdiffrast.py --model data/3drealcar/2024_06_04_13_44_39/colmap_processed/pcd_rescale --mesh output/3drealcar/2024_06_04_13_44_39/test/mesh/tsdf_fusion_post.ply
# (2)use obj:
# python mesh_viewer_nvdiffrast.py --model data/3drealcar/2024_06_04_13_44_39/colmap_processed/pcd_rescale --mesh 