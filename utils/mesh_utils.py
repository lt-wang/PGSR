# -*- coding: utf-8 -*-
"""
Mesh生成和优化工具模块
包含TSDF融合、形态学操作、Poisson重建、连通组件合并等功能
"""

import numpy as np
import open3d as o3d
import copy
from scipy import ndimage
from scipy.spatial import cKDTree
import os
import datetime


def voxel_morphology_close(volume, voxel_size, iterations=2):
    """
    在体素级别进行形态学闭运算，填充小孔洞
    
    Args:
        volume: TSDF体积对象
        voxel_size: 体素大小
        iterations: 形态学操作迭代次数
    
    Returns:
        处理后的点云和网格
    """
    print(f"执行体素形态学闭运算 (iterations={iterations})...")
    
    # 提取初始网格
    mesh = volume.extract_triangle_mesh()
    
    # 转换为点云
    pcd = mesh.sample_points_uniformly(number_of_points=500000)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        return mesh
    
    # 创建体素网格
    min_bound = points.min(axis=0) - voxel_size * 5
    max_bound = points.max(axis=0) + voxel_size * 5
    
    voxel_grid_size = ((max_bound - min_bound) / voxel_size).astype(int) + 1
    voxel_grid = np.zeros(voxel_grid_size, dtype=bool)
    
    # 填充体素网格
    voxel_indices = ((points - min_bound) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, voxel_grid_size - 1)
    voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
    
    # 形态学闭运算（膨胀后腐蚀）
    struct = ndimage.generate_binary_structure(3, 2)  # 3D连通性
    voxel_grid_closed = ndimage.binary_closing(voxel_grid, structure=struct, iterations=iterations)
    
    # 填充内部空洞
    voxel_grid_filled = ndimage.binary_fill_holes(voxel_grid_closed)
    
    # 转换回点云
    filled_indices = np.argwhere(voxel_grid_filled)
    filled_points = filled_indices * voxel_size + min_bound
    
    # 创建新点云并进行Poisson重建
    filled_pcd = o3d.geometry.PointCloud()
    filled_pcd.points = o3d.utility.Vector3dVector(filled_points)
    
    # 估计法线
    filled_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
    )
    filled_pcd.orient_normals_consistent_tangent_plane(30)
    
    return filled_pcd, mesh


def poisson_reconstruction(pcd, depth=9, min_density=0.1):
    """
    使用Poisson重建生成流形网格
    
    Args:
        pcd: 输入点云（带法线）
        depth: Poisson深度（越大越精细）
        min_density: 最小密度阈值
    
    Returns:
        重建的网格
    """
    print(f"执行Poisson重建 (depth={depth})...")
    
    # Poisson重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False
    )
    
    # 根据密度过滤低质量区域
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, min_density)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"Poisson重建完成: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面")
    return mesh


def merge_nearby_clusters(mesh, distance_threshold=0.01, min_triangles=50):
    """
    改进的连通组件合并算法：保留大组件并尝试融合
    
    Args:
        mesh: 输入网格
        distance_threshold: 距离阈值
        min_triangles: 小组件的最小三角形数阈值
    
    Returns:
        合并后的网格
    """
    print(f"检测并合并邻近的连通组件 (阈值={distance_threshold:.4f}m, 最小面数={min_triangles})...")
    
    # 分析连通组件
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    n_clusters = len(cluster_n_triangles)
    
    if n_clusters <= 1:
        print("只有1个连通组件，无需合并")
        return mesh
    
    print(f"发现 {n_clusters} 个连通组件，前10大: {sorted(cluster_n_triangles, reverse=True)[:10]}")
    
    # 计算每个组件的信息
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    
    cluster_info = []
    for i in range(n_clusters):
        cluster_mask = triangle_clusters == i
        cluster_faces = triangles[cluster_mask]
        cluster_verts_idx = np.unique(cluster_faces.flatten())
        cluster_verts = vertices[cluster_verts_idx]
        
        # 计算边界框和中心
        bbox_min = cluster_verts.min(axis=0)
        bbox_max = cluster_verts.max(axis=0)
        center = cluster_verts.mean(axis=0)
        
        cluster_info.append({
            'id': i,
            'n_triangles': cluster_n_triangles[i],
            'center': center,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'verts_idx': cluster_verts_idx,
            'faces': cluster_faces
        })
    
    # 找出主要组件（最大的）
    main_cluster_id = np.argmax(cluster_n_triangles)
    main_cluster = cluster_info[main_cluster_id]
    print(f"主组件: ID={main_cluster_id}, 面数={main_cluster['n_triangles']}")
    
    # 尝试融合策略：
    # 1. 保留所有 >= min_triangles 的组件
    # 2. 尝试将小组件融合到最近的大组件
    large_clusters = [c for c in cluster_info if c['n_triangles'] >= min_triangles]
    small_clusters = [c for c in cluster_info if c['n_triangles'] < min_triangles]
    
    print(f"大组件(>={min_triangles}面): {len(large_clusters)}个")
    print(f"小组件(<{min_triangles}面): {len(small_clusters)}个")
    
    # 尝试将小组件融合到距离最近的大组件
    merged_small_count = 0
    for small in small_clusters:
        # 找最近的大组件
        min_dist = float('inf')
        nearest_large = None
        
        for large in large_clusters:
            dist = np.linalg.norm(small['center'] - large['center'])
            if dist < min_dist:
                min_dist = dist
                nearest_large = large
        
        # 如果距离在阈值内，标记为可融合
        if min_dist < distance_threshold:
            merged_small_count += 1
    
    print(f"可以融合的小组件: {merged_small_count}个")
    
    # 保留所有大组件 + 不能融合的小组件
    clusters_to_keep = [c['id'] for c in large_clusters]
    
    # 移除可融合的小组件
    for small in small_clusters:
        min_dist = float('inf')
        for large in large_clusters:
            dist = np.linalg.norm(small['center'] - large['center'])
            if dist < min_dist:
                min_dist = dist
        
        if min_dist >= distance_threshold:  # 太远，保留
            clusters_to_keep.append(small['id'])
    
    print(f"保留 {len(clusters_to_keep)} 个组件（移除{n_clusters - len(clusters_to_keep)}个小碎片）")
    
    # 构建新网格
    new_mesh = o3d.geometry.TriangleMesh()
    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_offset = 0
    
    for cluster_id in clusters_to_keep:
        cluster = cluster_info[cluster_id]
        verts_idx = cluster['verts_idx']
        
        # 添加顶点
        cluster_verts = vertices[verts_idx]
        new_vertices.append(cluster_verts)
        
        # 添加颜色
        if vertex_colors is not None:
            cluster_colors = vertex_colors[verts_idx]
            new_colors.append(cluster_colors)
        
        # 重新索引面
        vert_map = {old_idx: new_idx + vertex_offset for new_idx, old_idx in enumerate(verts_idx)}
        reindexed_faces = np.array([[vert_map[v] for v in face] for face in cluster['faces']])
        new_faces.append(reindexed_faces)
        
        vertex_offset += len(verts_idx)
    
    # 合并所有数据
    new_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(new_vertices))
    new_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(new_faces))
    if new_colors:
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(new_colors))
    
    # 清理
    new_mesh.remove_duplicated_vertices()
    new_mesh.remove_duplicated_triangles()
    new_mesh.remove_degenerate_triangles()
    
    print(f"合并后: {len(new_mesh.vertices)}顶点, {len(new_mesh.triangles)}面")
    
    return new_mesh


def clean_mesh(mesh, min_len=1000):
    """
    清理网格，移除小的连通组件
    
    Args:
        mesh: 输入网格
        min_len: 最小三角形数阈值
    
    Returns:
        清理后的网格
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    改进的网格后处理
    
    Args:
        mesh: 输入网格
        cluster_to_keep: 要保留的最大组件数量
    
    Returns:
        后处理后的网格
    """
    print(f"后处理网格，保留 {cluster_to_keep} 个最大连通组件")
    mesh_0 = copy.deepcopy(mesh)
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    # 保留前cluster_to_keep个最大的组件
    if cluster_to_keep == 1:
        n_cluster = np.sort(cluster_n_triangles.copy())[-1]
    else:
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    
    n_cluster = max(n_cluster, 500)  # 降低阈值以保留更多几何
    
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    
    print(f"顶点数: {len(mesh.vertices)} -> {len(mesh_0.vertices)}")
    print(f"三角面数: {len(mesh.triangles)} -> {len(mesh_0.triangles)}")
    
    return mesh_0


def visualize_components(mesh, filepath, log_func=print):
    """
    可视化多个连通组件，每个组件用不同颜色
    
    Args:
        mesh: 输入网格
        filepath: 保存路径
        log_func: 日志函数
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 1:
        log_func(f"  检测到 {len(cluster_n_triangles)} 个连通组件，保存可视化结果...")
        import matplotlib.pyplot as plt
        colors = plt.cm.get_cmap('tab20', min(len(cluster_n_triangles), 20)).colors
        mesh_vis = copy.deepcopy(mesh)
        triangle_clusters = np.asarray(triangle_clusters)
        mesh_vis.vertex_colors = o3d.utility.Vector3dVector(np.zeros((len(mesh_vis.vertices), 3)))
        
        for i in range(len(cluster_n_triangles)):
            mask = triangle_clusters == i
            tris = np.asarray(mesh_vis.triangles)[mask]
            for tri in tris:
                for idx in tri:
                    mesh_vis.vertex_colors[idx] = colors[i % len(colors)][:3]
        
        o3d.io.write_triangle_mesh(filepath, mesh_vis, 
                                 write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        log_func(f"  可视化结果已保存: {filepath}")


def transfer_vertex_colors(source_mesh, target_mesh):
    """
    从源网格传递顶点颜色到目标网格
    
    Args:
        source_mesh: 源网格（有颜色）
        target_mesh: 目标网格（需要颜色）
    
    Returns:
        带颜色的目标网格
    """
    source_colors = np.asarray(source_mesh.vertex_colors)
    if len(source_colors) == 0:
        return target_mesh
    
    try:
        source_verts = np.asarray(source_mesh.vertices)
        target_verts = np.asarray(target_mesh.vertices)
        tree = cKDTree(source_verts)
        distances, indices = tree.query(target_verts, k=1)
        new_colors = source_colors[indices]
        target_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        print(f"  颜色传递完成: {len(new_colors)} 个顶点")
    except Exception as e:
        print(f"  警告: 颜色传递失败: {e}")
    
    return target_mesh


# def optimize_mesh_with_poco(mesh, voxel_size, output_dir):
#     """
#     使用POCO优化网格
    
#     Args:
#         mesh: 输入网格
#         voxel_size: 体素大小（用于归一化）
#         output_dir: 输出目录
    
#     Returns:
#         优化后的网格
#     """
#     try:
#         import torch
#         from models.POCO.generate_1 import POCO_get_geo, create_POCO_network, POCO_config
        
#         POCO_net = create_POCO_network(POCO_config)
        
#         mesh_vertices = np.asarray(mesh.vertices)
#         mesh_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        
#         if mesh_colors is not None and len(mesh_vertices) > 0:
#             xyz = mesh_vertices
#             rgb = mesh_colors * 255.0 if mesh_colors.max() <= 1.0 else mesh_colors
            
#             MAX_POINTS = 1000000
#             if len(xyz) > MAX_POINTS:
#                 indices = np.random.choice(len(xyz), MAX_POINTS, replace=False)
#                 xyz = xyz[indices]
#                 rgb = rgb[indices]
            
#             xyz_tensor = torch.tensor(xyz).float().to(device="cuda")
#             rgb_tensor = torch.tensor(rgb).float().to(device="cuda")
#             if rgb_tensor.max() > 1:
#                 rgb_tensor = rgb_tensor / 255.0
            
#             vertices_min = xyz_tensor.min(0)[0]
#             vertices_max = xyz_tensor.max(0)[0]
#             xyz_normalized = xyz_tensor - (vertices_max + vertices_min) / 2.
#             xyz_normalized = xyz_normalized / (vertices_max - vertices_min).max()
            
#             poco_temp_dir = os.path.join(output_dir, "poco_optimize")
#             os.makedirs(poco_temp_dir, exist_ok=True)
            
#             vertices_poco, faces_poco = POCO_get_geo(
#                 POCO_config, xyz_normalized, POCO_net,
#                 savedir_mesh_root=poco_temp_dir,
#                 object_name='poco_optimize',
#                 is_noisy=True
#             )
            
#             vertices_poco = vertices_poco * (vertices_max - vertices_min).max()
#             vertices_poco = vertices_poco + (vertices_max + vertices_min) / 2.
            
#             verts_np = vertices_poco.detach().cpu().numpy()
#             faces_np = faces_poco.detach().cpu().numpy()
            
#             poco_mesh = o3d.geometry.TriangleMesh()
#             poco_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
#             poco_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
            
#             # 颜色传递
#             poco_mesh = transfer_vertex_colors(mesh, poco_mesh)
            
#             return poco_mesh
#         else:
#             print("  警告: mesh无颜色信息，跳过POCO")
#             return mesh
    
#     except Exception as e:
#         print(f"  POCO优化失败: {e}")
#         return mesh


def extract_and_optimize_mesh(volume, voxel_size, output_path, 
                              use_morphology=True, use_poisson=True, 
                              use_poco=False, poisson_depth=9,
                              log_func=print):
    """
    从TSDF volume提取并优化网格的完整流程
    
    Args:
        volume: TSDF体积对象
        voxel_size: 体素大小
        output_path: 输出路径
        use_morphology: 是否使用形态学操作
        use_poisson: 是否使用Poisson重建
        use_poco: 是否使用POCO优化
        poisson_depth: Poisson重建深度
        log_func: 日志函数
    
    Returns:
        优化后的网格
    """
    log_func("=" * 80)
    log_func(f"开始提取和处理网格... {datetime.datetime.now()}")
    log_func("=" * 80)
    
    # ========== 步骤0: TSDF提取 ==========
    mesh = volume.extract_triangle_mesh()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    log_func("\n[步骤0: TSDF提取] 初始网格统计:")
    log_func(f"  顶点数: {len(mesh.vertices)}")
    log_func(f"  三角面数: {len(mesh.triangles)}")
    log_func(f"  连通组件数: {len(cluster_n_triangles)}")
    log_func(f"  前10大组件面数: {sorted(cluster_n_triangles, reverse=True)[:10]}")
    
    o3d.io.write_triangle_mesh(os.path.join(output_path, "step0_tsdf_raw.ply"), mesh, 
                             write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
    visualize_components(mesh, os.path.join(output_path, "step0_tsdf_raw_components.ply"), log_func)
    
    mesh_with_original_colors = copy.deepcopy(mesh)
    
    # ========== 步骤1: 形态学操作 ==========
    filled_pcd = None
    if use_morphology:
        log_func("\n[步骤1: 形态学操作]")
        filled_pcd, morphology_mesh = voxel_morphology_close(volume, voxel_size, iterations=5)
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            _, cluster_n_before, _ = mesh.cluster_connected_triangles()
            _, cluster_n_after, _ = morphology_mesh.cluster_connected_triangles()
        
        log_func(f"  形态学前组件数: {len(cluster_n_before)}")
        log_func(f"  形态学后组件数: {len(cluster_n_after)}")
        log_func(f"  组件数变化: {len(cluster_n_before)} → {len(cluster_n_after)} (减少 {len(cluster_n_before)-len(cluster_n_after)})")
        log_func(f"  顶点数: {len(mesh.vertices)} → {len(morphology_mesh.vertices)}")
        log_func(f"  三角面数: {len(mesh.triangles)} → {len(morphology_mesh.triangles)}")
        
        mesh = morphology_mesh
        o3d.io.write_triangle_mesh(os.path.join(output_path, "step1_morphology.ply"), mesh, 
                                 write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        visualize_components(mesh, os.path.join(output_path, "step1_morphology_components.ply"), log_func)
    else:
        log_func("\n[步骤1: 形态学操作] 跳过（未启用）")
    
    # ========== 步骤2: Poisson重建 ==========
    if use_poisson and filled_pcd is not None:
        log_func("\n[步骤2: Poisson重建]")
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            _, cluster_n_before, _ = mesh.cluster_connected_triangles()
        
        poisson_mesh = poisson_reconstruction(filled_pcd, depth=poisson_depth, min_density=0.05)
        
        # 颜色传递
        poisson_mesh = transfer_vertex_colors(mesh_with_original_colors, poisson_mesh)
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            _, cluster_n_after, _ = poisson_mesh.cluster_connected_triangles()
        
        log_func(f"  Poisson前组件数: {len(cluster_n_before)}")
        log_func(f"  Poisson后组件数: {len(cluster_n_after)}")
        log_func(f"  组件数变化: {len(cluster_n_before)} → {len(cluster_n_after)} (减少 {len(cluster_n_before)-len(cluster_n_after)})")
        log_func(f"  顶点数: {len(mesh.vertices)} → {len(poisson_mesh.vertices)}")
        log_func(f"  三角面数: {len(mesh.triangles)} → {len(poisson_mesh.triangles)}")
        
        mesh = poisson_mesh
        o3d.io.write_triangle_mesh(os.path.join(output_path, "step2_poisson.ply"), mesh, 
                                 write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        visualize_components(mesh, os.path.join(output_path, "step2_poisson_components.ply"), log_func)
    else:
        log_func("\n[步骤2: Poisson重建] 跳过（未启用或无输入点云）")
    
    # ========== 步骤3: 合并邻近组件 ==========
    log_func("\n[步骤3: 合并邻近组件]")
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        _, cluster_n_before, _ = mesh.cluster_connected_triangles()
    
    merged_mesh = merge_nearby_clusters(mesh, distance_threshold=voxel_size * 10, min_triangles=50)
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        _, cluster_n_after, _ = merged_mesh.cluster_connected_triangles()
    
    log_func(f"  合并前组件数: {len(cluster_n_before)}")
    log_func(f"  合并后组件数: {len(cluster_n_after)}")
    log_func(f"  组件数变化: {len(cluster_n_before)} → {len(cluster_n_after)} (减少 {len(cluster_n_before)-len(cluster_n_after)})")
    log_func(f"  顶点数: {len(mesh.vertices)} → {len(merged_mesh.vertices)}")
    log_func(f"  三角面数: {len(mesh.triangles)} → {len(merged_mesh.triangles)}")
    
    mesh = merged_mesh
    o3d.io.write_triangle_mesh(os.path.join(output_path, "step3_merged.ply"), mesh, 
                             write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
    visualize_components(mesh, os.path.join(output_path, "step3_merged_components.ply"), log_func)
    
    # ========== 步骤4: POCO优化 ==========
    if use_poco:
        log_func("\n[步骤4: POCO优化]")
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            _, cluster_n_before, _ = mesh.cluster_connected_triangles()
        
        poco_mesh = optimize_mesh_with_poco(mesh, voxel_size, output_path)
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            _, cluster_n_after, _ = poco_mesh.cluster_connected_triangles()
        
        log_func(f"  POCO前组件数: {len(cluster_n_before)}")
        log_func(f"  POCO后组件数: {len(cluster_n_after)}")
        log_func(f"  组件数变化: {len(cluster_n_before)} → {len(cluster_n_after)} (减少 {len(cluster_n_before)-len(cluster_n_after)})")
        log_func(f"  顶点数: {len(mesh.vertices)} → {len(poco_mesh.vertices)}")
        log_func(f"  三角面数: {len(mesh.triangles)} → {len(poco_mesh.triangles)}")
        
        mesh = poco_mesh
        o3d.io.write_triangle_mesh(os.path.join(output_path, "step4_poco.ply"), mesh, 
                                 write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        visualize_components(mesh, os.path.join(output_path, "step4_poco_components.ply"), log_func)
    else:
        log_func("\n[步骤4: POCO优化] 跳过（未启用）")
    
    # ========== 最终统计 ==========
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    log_func("\n" + "=" * 80)
    log_func("[最终统计]")
    log_func(f"  顶点数: {len(mesh.vertices)}")
    log_func(f"  三角面数: {len(mesh.triangles)}")
    log_func(f"  连通组件数: {len(cluster_n_triangles)}")
    log_func(f"  前10大组件面数: {sorted(cluster_n_triangles, reverse=True)[:10]}")
    
    o3d.io.write_triangle_mesh(os.path.join(output_path, "final_mesh.ply"), mesh, 
                             write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
    visualize_components(mesh, os.path.join(output_path, "final_mesh_components.ply"), log_func)
    
    log_func(f"\n✓ mesh优化完成，已保存所有结果 {datetime.datetime.now()}")
    log_func("=" * 80)
    
    return mesh
