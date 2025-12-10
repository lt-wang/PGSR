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
from scipy.ndimage import gaussian_filter

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
    """构建OpenGL风格的投影矩阵（NDC坐标系）"""
    projection = torch.zeros(4, 4)
    
    # OpenGL NDC: x,y in [-1,1], z in [-1,1]
    projection[0, 0] = 2.0 * fx / width
    projection[1, 1] = -2.0 * fy / height
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
    
    # 如果没有提供顶点颜色,使用白色
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
    
    # 返回深度图用于遮挡检测
    depth = rast_out[..., 2:3]  # z值
    
    return color[0], depth[0], mask[0]

# ==================== 阴影渲染：5步骤流程 ====================

def estimate_ground_plane(vertices_np):
    """
    估计地面平面参数
    输入: vertices_np (N, 3) - numpy数组，mesh的所有顶点
    输出: ground_normal (3,) - 地面法向量
          ground_point (3,) - 地面上的一个点
    """
    min_z = vertices_np[:, 2].min()
    ground_point = np.array([0, 0, min_z])
    ground_normal = np.array([0, 0, 1])
    
    print(f"[Ground Plane] point={ground_point}, normal={ground_normal}")
    return ground_normal, ground_point

def get_light_direction(pitch_angle, yaw_angle):
    """
    通过pitch和yaw角度获取光照方向
    输入:
        pitch_angle - 俯仰角（度），0度为水平，90度为正下方
        yaw_angle - 偏航角（度），0度为+X方向，逆时针旋转
    输出: 
        light_dir (3,) - 归一化的光照方向向量（从光源指向场景）
    说明:
        光线方向计算: 
        light_vec = [sin(pitch)*cos(yaw), sin(pitch)*sin(yaw), cos(pitch)]
    """
    # 转换为弧度
    pitch_rad = pitch_angle / 180 * np.pi
    yaw_rad = yaw_angle / 180 * np.pi
    
    # 计算光照方向向量
    light_dir = np.array([
        np.sin(pitch_rad) * np.cos(yaw_rad),
        np.sin(pitch_rad) * np.sin(yaw_rad),
        np.cos(pitch_rad)
    ])
    
    # 归一化（理论上已经是单位向量，但为了安全还是归一化）
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    print(f"[Light Direction] pitch={pitch_angle}°, yaw={yaw_angle}° -> direction={light_dir}")
    
    return light_dir

# ========== 步骤1: 将mesh放置到世界坐标系 ==========
def step1_mesh_to_world_coords(vertices, faces, vertex_colors):
    """
    步骤1: 将mesh放置到世界坐标系
    输入:
        vertices (N, 3) - torch tensor, mesh顶点
        faces (M, 3) - mesh面索引
        vertex_colors (N, 3) - 顶点颜色
    输出:
        world_vertices (N, 3) - 世界坐标系中的顶点
        world_faces (M, 3) - 面索引
        world_vertex_colors (N, 3) - 顶点颜色
    说明: 此处假设mesh已经在世界坐标系中，如需变换可在此添加变换矩阵
    """
    # 当前mesh已在世界坐标系，直接返回
    world_vertices = vertices
    world_faces = faces
    world_vertex_colors = vertex_colors
    
    print(f"[Step 1] Mesh in world coords: {world_vertices.shape[0]} vertices, {world_faces.shape[0]} faces")
    
    return world_vertices, world_faces, world_vertex_colors

# ========== 步骤2: 将3D点投影到地面 ==========
def step2_project_to_ground(world_vertices, light_dir, ground_point):
    """
    步骤2: 将3D点沿光线方向投影到地面
    输入:
        world_vertices (N, 3) - 世界坐标系中的顶点
        light_dir (3,) - 光照方向

        ground_point (3,) - 地面上的一点
    输出:
        ground_projected_vertices (N, 3) - 投影到地面的顶点坐标
    说明: 
        使用改进的投影算法：
        - 如果ground_point的z坐标为None，使用最低点的中位数作为地面高度
        - 投影沿光线方向进行，确保所有投影点落在地面上
    """
    device = world_vertices.device
    light_dir_t = torch.from_numpy(light_dir).float().to(device)
    
    # 获取地面高度
    if ground_point is None or ground_point[2] is None:
        # 使用最低点的z坐标中位数作为地面高度
        z_coords = world_vertices[:, 2].cpu().numpy()
        z_sorted = np.sort(z_coords)
        ground_height = np.median(z_sorted[:max(1, len(z_sorted)//10)])  # 使用最低10%的中位数
        print(f"[Step 2] Auto-detected ground height: {ground_height:.4f}")
    else:
        ground_height = ground_point[2]
    
    # 计算投影系数: coef = (z - ground_height) / light_z
    # 投影公式: projected = point - light_dir * coef
    z_diff = world_vertices[:, 2] - ground_height
    light_z = light_dir_t[2]
    
    # 避免除零，并确保只投影向下的点
    coef = z_diff / (light_z + 1e-8)
    coef = torch.clamp(coef, min=0)  # 只考虑光线前方的投影
    
    # 计算投影点
    ground_projected_vertices = world_vertices - light_dir_t.unsqueeze(0) * coef.unsqueeze(1)
    
    # 验证投影结果（所有点的z坐标应该等于ground_height）
    projected_z = ground_projected_vertices[:, 2]
    z_error = torch.abs(projected_z - ground_height).max().item()
    print(f"[Step 2] Projected {world_vertices.shape[0]} vertices to ground plane (z_error: {z_error:.6f})")
    
    return ground_projected_vertices

# ========== 步骤3: 将3D点投影到2D图像 ==========
def step3_project_to_2d(ground_projected_vertices, mvp, render_width, render_height):
    """
    步骤3: 将地面投影点投回相机平面得到2D像素坐标
    输入:
        ground_projected_vertices (N, 3) - 地面上的投影点
        mvp (4, 4) - MVP矩阵
        render_width, render_height - 渲染分辨率
    输出:
        shadow_pixels (N, 2) - 2D像素坐标，范围[0, W-1], [0, H-1]
        valid_mask (N,) - bool tensor，标记有效的投影点（在图像范围内且在相机前方）
    说明:
        应用MVP变换 -> 透视除法(齐次坐标到NDC) -> NDC转像素坐标
    """
    # 转换为齐次坐标
    shadow_homo = torch.cat([ground_projected_vertices, torch.ones_like(ground_projected_vertices[:, :1])], dim=1)
    
    # 应用MVP变换
    shadow_clip = shadow_homo @ mvp.T
    
    # 透视除法，得到NDC坐标 [-1, 1]
    shadow_ndc = shadow_clip[:, :3] / (shadow_clip[:, 3:4] + 1e-8)
    
    # NDC [-1, 1] 转换为像素坐标 [0, width-1], [0, height-1]
    shadow_pixels_x = (shadow_ndc[:, 0] + 1.0) * 0.5 * render_width
    shadow_pixels_y = (1.0 - shadow_ndc[:, 1]) * 0.5 * render_height  # y轴翻转
    
    shadow_pixels = torch.stack([shadow_pixels_x, shadow_pixels_y], dim=1)
    
    # 有效性检测：在图像范围内且在相机前方
    valid_mask = (shadow_pixels[:, 0] >= 0) & (shadow_pixels[:, 0] < render_width) & \
                 (shadow_pixels[:, 1] >= 0) & (shadow_pixels[:, 1] < render_height) & \
                 (shadow_clip[:, 3] > 0)  # w > 0 表示在相机前方
    
    print(f"[Step 3] Projected to 2D: {valid_mask.sum().item()}/{ground_projected_vertices.shape[0]} points valid")
    
    return shadow_pixels, valid_mask

# ========== 辅助函数 ==========
def interpolate_shadow_mask(shadow_mask_mat, r=3, iterations=3):
    """
    使用形态学方法插值阴影mask
    输入:
        shadow_mask_mat (H, W, C) - 阴影mask numpy数组
        r - 形态学核的大小
        iterations - 迭代次数
    返回:
        插值后的阴影mask (H, W, C)
    """
    h, w = shadow_mask_mat.shape[:2]
    # 添加padding以防止边缘情况
    shadow_mask_mat = np.pad(shadow_mask_mat, ((h, h), (w, w), (0, 0)) if len(shadow_mask_mat.shape) == 3 else ((h, h), (w, w)))
    
    # Close操作填充空洞
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    shadow_mask_mat = cv2.morphologyEx(shadow_mask_mat, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
    
    # 移除padding
    if len(shadow_mask_mat.shape) == 3:
        return shadow_mask_mat[h:2*h, w:2*w, ...]
    else:
        return shadow_mask_mat[h:2*h, w:2*w]


def check_the_occlusion(mask_mat, shadow_mask_mat):
    """
    从阴影mask中分离出车辆mask，去除遮挡
    输入:
        mask_mat (H, W) - 车辆mask
        shadow_mask_mat (H, W) - 阴影mask
    返回:
        去除遮挡后的阴影mask (H, W)
    """
    # 二值化
    shadow_mask_mat = shadow_mask_mat > 0
    mask_mat = mask_mat > 0
    
    # 集合差操作：阴影 - (阴影 ∩ 车辆)
    shadow_mask_mat = np.logical_and(shadow_mask_mat, np.logical_not(np.logical_and(shadow_mask_mat, mask_mat)))
    
    return shadow_mask_mat.astype(np.uint8) * 255


def shadow_refine(mask_mat, shadow_mask_mat, r=30, iterations=2):
    """
    填充阴影和车辆之间的空隙
    输入:
        mask_mat (H, W) - 车辆mask
        shadow_mask_mat (H, W) - 阴影mask
        r - 形态学核大小
        iterations - 迭代次数
    返回:
        精炼后的阴影mask (H, W)
    """
    # 二值化
    shadow_mask_mat = shadow_mask_mat > 0
    mask_mat = mask_mat > 0
    
    # 填充空隙：阴影 ∪ 车辆
    shadow_mask_mat = np.logical_or(shadow_mask_mat, mask_mat).astype(np.uint8) * 255
    
    # Close操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    shadow_mask_mat = cv2.morphologyEx(shadow_mask_mat, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # 集合差操作：移除车辆部分
    shadow_mask_mat = shadow_mask_mat > 0
    shadow_mask_mat = np.logical_and(shadow_mask_mat, np.logical_not(np.logical_and(shadow_mask_mat, mask_mat)))
    
    return shadow_mask_mat.astype(np.uint8) * 255


# ========== 步骤4: 生成2D阴影mask ==========
def step4_generate_shadow_mask(shadow_pixels, valid_mask, render_width, render_height):
    """
    步骤4: 从2D像素点生成阴影mask
    输入:
        shadow_pixels (N, 2) - 2D像素坐标
        valid_mask (N,) - 有效点mask
        render_width, render_height - 图像分辨率
    输出:
        shadow_mask (H, W) - 2D阴影mask，0-1之间的浮点数
    说明:
        遍历有效的投影点，在对应像素位置设置阴影值
    """
    device = shadow_pixels.device
    shadow_mask = torch.zeros((render_height, render_width), dtype=torch.float32, device=device)
    
    # 只处理有效的投影点
    valid_pixels = shadow_pixels[valid_mask]
    
    if valid_pixels.shape[0] == 0:
        print(f"[Step 4] No valid shadow pixels")
        return shadow_mask
    
    # 将像素坐标转换为整数
    pixel_coords = valid_pixels.long()
    pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, render_width - 1)
    pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, render_height - 1)
    
    # 在对应位置设置阴影值
    shadow_mask[pixel_coords[:, 1], pixel_coords[:, 0]] = 1.0
    
    print(f"[Step 4] Generated shadow mask: {valid_pixels.shape[0]} pixels marked")
    
    return shadow_mask

# ========== 步骤5: 优化阴影mask ==========
def step5_refine_shadow_mask(shadow_mask, object_depth, object_mask, 
                             interpolate_r=3, interpolate_iter=3,
                             refine_r=30, refine_iter=2,
                             blur_kernel=None, shadow_intensity=0.6):
    """
    步骤5: 优化和精细化阴影mask（使用标准方法）
    输入:
        shadow_mask (H, W) - 原始阴影mask
        object_depth (H, W) - 物体深度图（未使用但保留接口）
        object_mask (H, W) - 物体有效性mask（1表示有物体）
        interpolate_r - 插值形态学核大小
        interpolate_iter - 插值迭代次数
        refine_r - 精炼形态学核大小
        refine_iter - 精炼迭代次数
        blur_kernel - 模糊核大小（None表示不模糊）
        shadow_intensity - 最终阴影强度[0,1]
    输出:
        refined_mask (H, W) - 优化后的阴影mask
    说明:
        5.1 形态学插值：填充空洞，使阴影连续
        5.2 去遮挡：移除物体自身上的阴影
        5.3 精炼：填充阴影和车辆之间的空隙
        5.4 模糊（可选）：柔化阴影边缘
        5.5 强度调整
    """
    # 转换为numpy进行处理
    shadow_mask_np = (shadow_mask.cpu().numpy() * 255).astype(np.uint8)
    object_mask_np = (object_mask.cpu().numpy().squeeze() * 255).astype(np.uint8)
    
    # 扩展到3通道以兼容插值函数
    if len(shadow_mask_np.shape) == 2:
        shadow_mask_np = np.stack([shadow_mask_np] * 3, axis=-1)
    if len(object_mask_np.shape) == 2:
        object_mask_np = np.stack([object_mask_np] * 3, axis=-1)
    
    # 5.1 形态学插值：填充稀疏点
    print(f"[Step 5.1] Interpolate shadow mask (r={interpolate_r}, iter={interpolate_iter})")
    shadow_mask_np = interpolate_shadow_mask(shadow_mask_np, r=interpolate_r, iterations=interpolate_iter)
    
    # 5.2 去遮挡：移除物体自身上的阴影
    print(f"[Step 5.2] Remove occlusion")
    shadow_mask_np = check_the_occlusion(object_mask_np, shadow_mask_np)
    
    # 转换为单通道
    if len(shadow_mask_np.shape) == 3:
        shadow_mask_np = shadow_mask_np[:, :, 0]
    
    # 5.3 精炼：填充阴影和车辆之间的空隙
    print(f"[Step 5.3] Refine shadow (r={refine_r}, iter={refine_iter})")
    object_mask_2d = object_mask_np[:, :, 0] if len(object_mask_np.shape) == 3 else object_mask_np
    shadow_mask_np = shadow_refine(object_mask_2d, shadow_mask_np, r=refine_r, iterations=refine_iter)
    
    # 5.4 模糊（可选）
    if blur_kernel is not None and blur_kernel > 0:
        print(f"[Step 5.4] Blur shadow (kernel={blur_kernel})")
        shadow_mask_np = cv2.blur(shadow_mask_np, ksize=(blur_kernel, blur_kernel))
    
    # 转换为float [0, 1]
    shadow_mask_np = shadow_mask_np.astype(np.float32) / 255.0
    
    # 5.5 归一化并调整阴影强度
    if shadow_mask_np.max() > 0:
        shadow_mask_np = shadow_mask_np / shadow_mask_np.max()
    
    print(f"[Step 5.5] Adjust shadow intensity: {shadow_intensity}")
    shadow_mask_np = shadow_mask_np * shadow_intensity
    
    refined_mask = torch.from_numpy(shadow_mask_np).float().to(shadow_mask.device)
    
    print(f"[Step 5] Shadow mask refined - intensity range: [{refined_mask.min():.3f}, {refined_mask.max():.3f}]")
    
    return refined_mask

# ========== 完整的阴影渲染流程 ==========
def render_with_shadow(glctx, vertices, faces, mvp, resolution, vertex_colors, 
                       light_dir, ground_normal, ground_point, render_width, render_height,
                       interpolate_r=3, interpolate_iter=3, refine_r=30, refine_iter=2,
                       blur_kernel=3, shadow_intensity=0.6):
    """
    完整的阴影渲染流程（5个步骤）
    输入:
        glctx - nvdiffrast context
        vertices (N, 3) - mesh顶点
        faces (M, 3) - mesh面
        mvp (4, 4) - MVP矩阵
        resolution [H, W] - 渲染分辨率
        vertex_colors (N, 3) - 顶点颜色
        light_dir (3,) - 光照方向
        ground_normal (3,) - 地面法向量
        ground_point (3,) - 地面点
        render_width, render_height - 渲染尺寸
    输出:
        final_image (H, W, 3) - 带阴影的最终图像
        shadow_mask (H, W) - 最终阴影mask
    """
    print("\n" + "="*60)
    print("Shadow Rendering Pipeline (5 Steps)")
    print("="*60)
    
    # 步骤0: 渲染原始物体
    print("\n[Step 0] Render original mesh...")
    color, depth, object_mask = render_mesh_nvdiffrast(glctx, vertices, faces, mvp, resolution, vertex_colors)
    print(f"[Step 0] Rendered image: {color.shape}, depth: {depth.shape}")
    
    # 步骤1: 将mesh放置到世界坐标系
    print("\n[Step 1] Put mesh to world coords...")
    world_vertices, world_faces, world_vertex_colors = step1_mesh_to_world_coords(vertices, faces, vertex_colors)
    
    # 步骤2: 将3D点投影到地面
    print("\n[Step 2] Project 3D points to ground...")
    ground_projected_vertices = step2_project_to_ground(world_vertices, light_dir, None)
    
    # 步骤3: 将3D点投影到2D
    print("\n[Step 3] Project 3D points to 2D...")
    shadow_pixels, valid_mask = step3_project_to_2d(ground_projected_vertices, mvp, render_width, render_height)
    
    # 步骤4: 生成2D阴影mask
    print("\n[Step 4] Generate 2D shadow mask...")
    shadow_mask = step4_generate_shadow_mask(shadow_pixels, valid_mask, render_width, render_height)
    
    # 步骤5: 优化阴影mask
    print("\n[Step 5] Refine shadow mask...")
    shadow_mask = step5_refine_shadow_mask(shadow_mask, depth, object_mask,
                                           interpolate_r=interpolate_r,
                                           interpolate_iter=interpolate_iter,
                                           refine_r=refine_r,
                                           refine_iter=refine_iter,
                                           blur_kernel=blur_kernel,
                                           shadow_intensity=shadow_intensity)
    
    # 融合阴影到图像
    print("\n[Blending] Blend shadow with image...")
    shadow_factor = 1.0 - shadow_mask.unsqueeze(-1)
    final_image = color * shadow_factor
    
    # 创建alpha通道: 车辆区域 + 阴影区域都不透明
    print("\n[Alpha Channel] Create alpha channel...")
    # object_mask是物体的mask (H, W, 1)
    # shadow_mask是阴影的mask (H, W)
    alpha_channel = torch.clamp(object_mask.squeeze(-1) + shadow_mask, 0, 1)  # 合并车辆和阴影区域
    print(f"[Alpha Channel] Created: min={alpha_channel.min():.3f}, max={alpha_channel.max():.3f}")
    
    print("="*60 + "\n")
    
    return final_image, shadow_mask, alpha_channel

# ==================== 主程序 ====================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--mesh", type=str)
    parser.add_argument("--enable_shadow", action="store_true", help="Enable shadow rendering")
    parser.add_argument("--pitch_angle", type=float, default=80.0, help="Light pitch angle in degrees (0=horizontal, 90=downward)")
    parser.add_argument("--yaw_angle", type=float, default=45.0, help="Light yaw angle in degrees (0=+X axis, counter-clockwise)")
    
    # 阴影refine参数
    parser.add_argument("--interpolate_r", type=int, default=3, help="Interpolation morphology kernel size")
    parser.add_argument("--interpolate_iter", type=int, default=3, help="Interpolation iterations")
    parser.add_argument("--refine_r", type=int, default=30, help="Refinement morphology kernel size")
    parser.add_argument("--refine_iter", type=int, default=2, help="Refinement iterations")
    parser.add_argument("--blur_kernel", type=int, default=3, help="Blur kernel size (0 to disable)")
    parser.add_argument("--shadow_intensity", type=float, default=0.6, help="Shadow intensity [0, 1]")
    
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
        vertex_colors = torch.ones_like(vertices) * 0.8
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
    
    # 计算mesh的边界框
    vertices_np = mesh.vertices
    bbox_min = vertices_np.min(axis=0)
    bbox_max = vertices_np.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    
    print(f"Mesh bounding box: min={bbox_min}, max={bbox_max}, size={bbox_size}")
    
    znear = 0.01
    zfar = 100.0
    
    # 构建投影矩阵
    projection_matrix = get_projection_matrix(
        fx_scaled, fy_scaled, 
        render_width/2, render_height/2, 
        render_width, render_height, 
        znear, zfar
    )
    projection_matrix = projection_matrix.to(device)
    
    # 阴影渲染准备
    if args.enable_shadow:
        print("\n=== Shadow rendering enabled ===")
        print(f"Light parameters: pitch={args.pitch_angle}°, yaw={args.yaw_angle}°")
        ground_normal, ground_point = estimate_ground_plane(vertices_np)
        light_dir = get_light_direction(args.pitch_angle, args.yaw_angle)
    
    # 创建输出目录
    output_dir = "view_output"
    if args.enable_shadow:
        output_dir = "view_output_shadow"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建阴影mask可视化目录
    if args.enable_shadow:
        shadow_dir = os.path.join(output_dir, "shadow_masks")
        os.makedirs(shadow_dir, exist_ok=True)
    
    print(f"Saving rendered images to: {os.path.abspath(output_dir)}")
    
    # 读取外参并渲染
    test_extrinsics = read_extrinsics(args)
    
    for extrinsic_idx in tqdm(test_extrinsics, desc="Rendering views"):
        extrinsic = test_extrinsics[extrinsic_idx]
        
        # COLMAP外参
        camera_pose_w2c = getTranspose(extrinsic.qvec, extrinsic.tvec)
        
        # 坐标系转换
        colmap_to_opengl = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        view_matrix = colmap_to_opengl @ camera_pose_w2c
        view_matrix = torch.from_numpy(view_matrix).float().to(device)
        
        # 计算MVP矩阵
        mvp = projection_matrix @ view_matrix
        
        # 渲染
        with torch.no_grad():
            if args.enable_shadow:
                color, shadow_mask, alpha_channel = render_with_shadow(
                    glctx, vertices, faces, mvp, resolution, vertex_colors,
                    light_dir, ground_normal, ground_point, render_width, render_height,
                    interpolate_r=args.interpolate_r,
                    interpolate_iter=args.interpolate_iter,
                    refine_r=args.refine_r,
                    refine_iter=args.refine_iter,
                    blur_kernel=args.blur_kernel,
                    shadow_intensity=args.shadow_intensity
                )
                
                # 保存阴影mask可视化
                shadow_vis = (shadow_mask.cpu().numpy() * 255).astype(np.uint8)
                shadow_path = os.path.join(shadow_dir, extrinsic.name)
                cv2.imwrite(shadow_path, shadow_vis)
                
                # 创建RGBA图像
                color_np = (torch.clamp(color, 0, 1).cpu().numpy() * 255).astype(np.uint8)
                alpha_np = (torch.clamp(alpha_channel, 0, 1).cpu().numpy() * 255).astype(np.uint8)
                
                # 合并RGB和Alpha通道
                rgba_np = np.dstack([color_np, alpha_np])  # (H, W, 4)
                
                # 转换RGB为BGR（OpenCV格式）但保持Alpha通道不变
                rgba_np = cv2.cvtColor(rgba_np, cv2.COLOR_RGBA2BGRA)
                rgba_np = cv2.resize(rgba_np, (rgba_np.shape[1]//2, rgba_np.shape[0]//2))
                
                # 保存RGBA图像（需要使用PNG格式支持透明度）
                output_path = os.path.join(output_dir, extrinsic.name.replace('.jpg', '.png').replace('.jpeg', '.png'))
                cv2.imwrite(output_path, rgba_np)
            else:
                color, _, _ = render_mesh_nvdiffrast(glctx, vertices, faces, mvp, resolution, vertex_colors)
                
                # 转换为图像
                color_np = (torch.clamp(color, 0, 1).cpu().numpy() * 255).astype(np.uint8)
                color_np = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
                color_np = cv2.resize(color_np, (color_np.shape[1]//2, color_np.shape[0]//2))
                
                # 保存
                output_path = os.path.join(output_dir, extrinsic.name)
                cv2.imwrite(output_path, color_np)
    
    print(f"\nRendering complete! {len(test_extrinsics)} images saved to {os.path.abspath(output_dir)}")
    if args.enable_shadow:
        print(f"Shadow masks saved to {os.path.abspath(shadow_dir)}")
# Example:
# (1) 不带阴影渲染:
# python mesh_viewer_nvdiffrast.py --model data/3drealcar/2024_06_04_13_44_39/colmap_processed/pcd_rescale --mesh output/3drealcar/2024_06_04_13_44_39/test/mesh/tsdf_fusion_post.ply
# 
# (2) 带阴影渲染（默认参数）:
# python mesh_viewer_nvdiffrast.py --model data/3drealcar/2024_06_04_13_44_39/colmap_processed/pcd_rescale --mesh output/3drealcar/2024_06_04_13_44_39/test/mesh/tsdf_fusion_post.ply --enable_shadow
#
# (3) 带阴影渲染（自定义光源角度）:
# python mesh_viewer_nvdiffrast.py --model data/3drealcar/2024_06_04_13_44_39/colmap_processed/pcd_rescale --mesh output/3drealcar/2024_06_04_13_44_39/test/mesh/tsdf_fusion_post.ply --enable_shadow --pitch_angle 45 --yaw_angle 30
#
# (4) 带阴影渲染（完整自定义参数）:
# python mesh_viewer_nvdiffrast.py --model data/3drealcar/2024_06_04_13_44_39/colmap_processed/pcd_rescale --mesh output/3drealcar/2024_06_04_13_44_39/test/mesh/tsdf_fusion_post.ply --enable_shadow --pitch_angle 80 --yaw_angle 45 --interpolate_r 3 --interpolate_iter 3 --refine_r 30 --refine_iter 2 --blur_kernel 3 --shadow_intensity 0.6