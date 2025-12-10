import numpy as np
import os
from argparse import ArgumentParser
import json, math
import cv2
from tqdm import tqdm
import mitsuba as mi
import json

# 设置 Mitsuba 3 变体 (例如 'gpu_autodiff_rgb')
# 请根据您的需求选择合适的变体。如果不需要 GPU 或 AD，可以使用 'scalar_rgb'
mi.set_variant('scalar_rgb')

# ----------------- 原始工具函数 -----------------

# 确保您可以从您的环境中导入这些文件
# from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
# from utils.graphics_utils import focal2fov

# 简化版本的 read_intrinsics_binary 和 read_extrinsics_binary
# 假设这些函数能够从 COLMAP 文件中正确读取数据结构。
# ⚠️ 确保您的 `scene.colmap_loader` 模块可用。
try:
    from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
    from utils.graphics_utils import focal2fov
except ImportError:
    print("Warning: Missing custom colmap_loader/graphics_utils. Using placeholders for testing!")
    # Placeholder classes for demonstration if the custom files are missing
    class CameraIntrinsics:
        def __init__(self, model, width, height, params):
            self.model = model
            self.width = width
            self.height = height
            self.params = params
    class ImageExtrinsics:
        def __init__(self, name, qvec, tvec):
            self.name = name
            self.qvec = qvec
            self.tvec = tvec
    # Dummy implementation for focal2fov if needed
    def focal2fov(focal, pixels):
        return 2 * np.arctan(pixels / (2 * focal)) * 180 / np.pi

def read_extrinsics(args : ArgumentParser):
    # ... (与原代码相同，用于读取 COLMAP 相机位姿)
    train_test_split = None
    split_file = os.path.join(args.model, "split.json")
    if os.path.exists(split_file):
        train_test_split = json.load(open(split_file))
        test_list = train_test_split["test"]
    extrinsics = read_extrinsics_binary(os.path.join(args.model, "sparse/0/images.bin"))
    if train_test_split is not None:
        # get those partern in extrinsics which name in test_list
        test_extrinsics_idx = [extrinsic for extrinsic in extrinsics if extrinsics[extrinsic].name.split(".")[0] in test_list]
        test_extrinsics = {extrinsic: extrinsics[extrinsic] for extrinsic in test_extrinsics_idx}
    else :
        test_extrinsics = extrinsics
    return test_extrinsics

def qvec2rotmat(qvec):
    # ... (与原代码相同，用于将四元数转换为旋转矩阵)
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
    # ... (与原代码相同，用于构建 COLMAP 世界坐标系到相机坐标系的变换矩阵 C2W)
    rotation_matrix = qvec2rotmat(qvec)
    
    # Colmap camera-to-world (C2W) matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = tvec
    
    # ⚠️ NOTE: Colmap uses W2C (World to Camera) in its Image structure,
    # and the rotation matrix in Image is C2W (Camera to World).
    # The structure `extrinsic_matrix` built here is C2W.
    # W2C = C2W^-1
    W2C = np.linalg.inv(extrinsic_matrix)
    return W2C

def get_mitsuba_camera_params(camera_intr, W2C_colmap, resolution=1):
    # 提取 COLMAP 内参
    if camera_intr.model == "SIMPLE_PINHOLE":
        focal_length_x = camera_intr.params[0]
        focal_length_y = camera_intr.params[0]
    elif camera_intr.model == "PINHOLE":
        focal_length_x = camera_intr.params[0]
        focal_length_y = camera_intr.params[1]
    else:
        raise ValueError("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")
        
    width = camera_intr.width // resolution
    height = camera_intr.height // resolution

    # ------------------ 调整位姿 (COLMAP W2C -> Mitsuba W2C) ------------------
    # Mitsuba (OpenGL/Computer Graphics) 约定：Y 轴向上，Z 轴向外 (与 COLMAP Z 轴朝前不同)
    # COLMAP 的 W2C 矩阵: [R | t] 形式
    # W_mitsuba = T_colmap_to_mitsuba * W_colmap
    # C_mitsuba = C_colmap
    
    # COLMAP (W2C) 到世界坐标系到 GL 相机坐标系 (W2C_gl) 的转换
    
    # COLMAP Camera: X right, Y down, Z forward.
    # Mitsuba/GL Camera: X right, Y up, Z backward (or forward, depends on convention, let's stick to X right, Y up, Z backward for standard GL view matrix).
    # Since Mitsuba uses a standard W2C matrix where C is the camera in world coords, 
    # we need to convert the COLMAP W2C (which is W2C_COLMAP) to W2C_MITS
    
    # 1. Colmap C2W (即 W2C_colmap 的逆)
    C2W_colmap = np.linalg.inv(W2C_colmap.T)
    
    # 2. 从 Colmap 空间 (R, T) 转换为 Mitsuba 空间
    R_colmap = C2W_colmap[:3, :3]
    T_colmap = C2W_colmap[:3, 3]

    # COLMAP 相机坐标系转换矩阵 (Y_colmap -> -Y_GL, Z_colmap -> -Z_GL)
    # This matrix flips Y and Z axes to go from COLMAP camera space to GL camera space (Y up, Z back)
    # C_GL = F * C_COLMAP, where F = diag(1, -1, -1)
    # W2C_MITS = W2C_COLMAP @ F^-1, 
    # and W2C_MITS is a lookat matrix.
    
    # Mitsuba expects the camera transformation matrix to be World-to-Camera (W2C), 
    # which is the inverse of Camera-to-World (C2W).
    # We use the LookAt utility in Mitsuba to define the camera position and orientation, 
    # which is often simpler than manually constructing the W2C matrix.

    # Extract world coordinates of the camera center (C2W translation)
    origin = mi.Point3f(float(T_colmap[0]), float(T_colmap[1]), float(T_colmap[2]))
    
    # Build the Camera-to-World (C2W) transformation matrix directly
    # COLMAP W2C -> C2W (by inverting)
    C2W_colmap = np.linalg.inv(W2C_colmap)
    
    # Extract rotation and translation from C2W
    R_c2w = C2W_colmap[:3, :3]
    t_c2w = C2W_colmap[:3, 3]
    
    # Convert COLMAP coordinate system (X right, Y down, Z forward) 
    # to Mitsuba/GL coordinate system (X right, Y up, Z backward)
    # Flip Y and Z axes
    coord_transform = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Apply coordinate transformation: R_mitsuba = coord_transform @ R_colmap
    R_mitsuba = coord_transform @ R_c2w
    t_mitsuba = coord_transform @ t_c2w
    
    # Construct 4x4 C2W matrix for Mitsuba
    C2W_mitsuba = np.eye(4, dtype=np.float32)
    C2W_mitsuba[:3, :3] = R_mitsuba
    C2W_mitsuba[:3, 3] = t_mitsuba
    
    # Convert numpy array to Mitsuba Transform4f
    to_world = mi.Transform4f(C2W_mitsuba.tolist())
    
    # Calculate FOV in degrees for Mitsuba
    FovX = focal2fov(focal_length_x, width)
    
    # Mitsuba perspective camera definition with film sensor
    camera_params = mi.load_dict({
        'type': 'perspective',
        'to_world': to_world,
        'fov': FovX,
        'fov_axis': 'x',
        'film': {
            'type': 'hdrfilm',
            'width': int(width),
            'height': int(height),
            'pixel_format': 'rgb',
            'rfilter': { 'type': 'gaussian' }
        },
    })
    return camera_params, width, height

# ----------------- Mitsuba 渲染函数 -----------------

def render_with_mitsuba(mesh_path, camera_intr, test_extrinsics, output_dir):
    
    # 1. 创建基础场景
    scene_dict = {
        'type': 'scene',
        # 几何体：加载 OBJ 文件
        'mesh': {
            'type': 'obj',
            'filename': mesh_path,
            'bsdf': {
                'type': 'twosided',
                'material': { 'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.7, 0.7, 0.7]} }
            }
        },
        # 光源：简单的环境光 (可根据需要替换为其他光源，如 directional)
        'integrator': {
            'type': 'path',
            'max_depth': 8,
        },
        'sampler': {
            'type': 'independent',
            'sample_count': 16 # 减少采样数以加快渲染速度
        },
        # 环境光 (Ambient Light)
        'emitter': {
            'type': 'envmap',
            'filename': 'data/Environment_Maps/high_res_envmaps_1k/sunrise.hdr', # 假设有一个环境贴图文件
            'scale': 1,
        }
    }
    # 2. 渲染循环
    print("Initializing Mitsuba scene...")
    
    for extrinsic_idx in tqdm(test_extrinsics, desc="Rendering views with Mitsuba 3"):
        extrinsic = test_extrinsics[extrinsic_idx]
        
        # 获取 COLMAP 世界坐标系到相机坐标系的变换矩阵 (W2C)
        W2C_colmap = getTranspose(extrinsic.qvec, extrinsic.tvec)
        
        # 获取 Mitsuba 相机参数和分辨率
        camera_params, width, height = get_mitsuba_camera_params(camera_intr, W2C_colmap, resolution=1)
        
        # 为每一帧构建完整的场景字典（包含动态相机）
        frame_scene_dict = dict(scene_dict)  # 浅拷贝
        frame_scene_dict['sensor'] = camera_params
        
        # 加载场景
        scene = mi.load_dict(frame_scene_dict)
        
        # 渲染
        try:
            image = mi.render(scene, spp=16) # 使用 SPP (Samples Per Pixel) 替代 sampler.sample_count
        except Exception as e:
            print(f"Error rendering {extrinsic.name}: {e}")
            continue
            
        # 3. 处理和保存图像
        
        # 将 Mitsuba 图像 (RGB/HDR) 转换为 numpy 数组
        # Mitsuba 3 返回的是 Bitmap 对象，直接转换为 numpy
        image_np = np.array(image)
        
        # 确保是 RGB 格式，应用 Gamma 校正
        image_np = np.clip(image_np**(1/2.2), 0.0, 1.0) # 简单的 Gamma 校正
        image_bgr = (image_np * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

        # 调整大小 (与原代码保持一致)
        image_bgr = cv2.resize(image_bgr, (width//2, height//2))
        
        output_path = os.path.join(output_dir, extrinsic.name.replace(".png", ".jpg")) # 使用 jpg/png
        cv2.imwrite(output_path, image_bgr)
        
    print(f"\nRendering complete! {len(test_extrinsics)} images saved to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to COLMAP sparse model directory.")
    parser.add_argument("--mesh", type=str, help="Path to the OBJ or PLY mesh file.")
    
    args = parser.parse_args()
    
    # 读取内参和外参
    intrinsics = read_intrinsics_binary(os.path.join(args.model, "sparse/0/cameras.bin"))
    
    # 假设使用第一个相机 ID 的内参 (通常为 1)
    if 1 not in intrinsics:
        raise KeyError("Camera ID 1 not found in intrinsics.bin")
        
    camera_intr = intrinsics[1]
    test_extrinsics = read_extrinsics(args)
    
    # 创建输出目录
    output_dir = "view_output_mitsuba"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving rendered images to: {os.path.abspath(output_dir)}")

    # 调用 Mitsuba 渲染函数
    render_with_mitsuba(args.mesh, camera_intr, test_extrinsics, output_dir)