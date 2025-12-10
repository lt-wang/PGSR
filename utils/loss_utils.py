#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
def cos_loss(x, y):
    """
    计算余弦相似度损失
    
    参数:
        x: 第一个向量张量，shape为 (C, H, W) 或 (C, N)
           其中C是通道数（如3用于RGB法线），H/W是高度/宽度，N是像素数
        y: 第二个向量张量，shape与x相同
    
    返回:
        loss: 标量损失值
    """
    # 确保输入形状相同
    assert x.shape == y.shape, f"Input shapes must match: {x.shape} vs {y.shape}"
    assert x.dim() in [2, 3], f"Expected 2D or 3D tensor, got shape {x.shape}"
    
    # 获取通道维度
    C = x.shape[0]
    
    # 计算每个像素位置的L2范数（沿通道维度）
    x_norm = torch.norm(x, p=2, dim=0, keepdim=True)  # (1, H, W) 或 (1, N)
    y_norm = torch.norm(y, p=2, dim=0, keepdim=True)  # (1, H, W) 或 (1, N)
    
    # 避免除以零
    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)
    
    # 归一化向量（沿通道维度）
    x_normalized = x / x_norm  # (C, H, W) 或 (C, N)
    y_normalized = y / y_norm  # (C, H, W) 或 (C, N)
    
    # 计算余弦相似度：对应通道的点积
    cosine_similarity = (x_normalized * y_normalized).sum(dim=0)  # (H, W) 或 (N,)
    
    # 余弦相似度范围是[-1, 1]，转换为损失值[0, 2]
    # loss = 1 - cosine_similarity，范围是[0, 2]
    loss = (1.0 - cosine_similarity).mean()
    
    return loss
def l1_loss(network_output, gt):
    """计算L1损失（平均绝对误差）"""
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    """计算L2损失（均方误差）"""
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    """
    生成一维高斯核
    
    参数:
        window_size: 窗口大小
        sigma: 高斯标准差
    
    返回:
        归一化的高斯核
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    创建用于SSIM计算的二维高斯窗口
    
    参数:
        window_size: 窗口大小
        channel: 图像通道数
    
    返回:
        二维高斯窗口
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算结构相似性指数（SSIM）
    
    参数:
        img1: 第一张图像
        img2: 第二张图像
        window_size: 高斯窗口大小
        size_average: 是否对所有像素取平均
    
    返回:
        SSIM值
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

def get_img_grad_weight(img, beta=2.0):
    """
    计算图像梯度权重，用于边缘感知
    梯度较大的区域权重较小
    
    参数:
        img: 输入图像 (C, H, W)
        beta: 权重调整参数
    
    返回:
        归一化的梯度权重图
    """
    _, hd, wd = img.shape 
    # 计算四个方向的邻近像素
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    # 计算x和y方向的梯度
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    # 取最大梯度
    grad_img, _ = torch.max(grad_img, dim=0)
    # 归一化到[0,1]
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    # 填充边界
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def lncc(ref, nea):
    """
    计算局部归一化互相关（Local Normalized Cross-Correlation）
    用于多视图光度一致性损失
    
    LNCC是一种衡量两个图像patch相似度的度量，对亮度和对比度变化具有鲁棒性。
    它通过计算局部区域内像素值的归一化相关性来评估两个patch的匹配程度。
    
    参数:
        ref: 参考patch [batch_size, total_patch_size]
             来自参考视图的展平patch，total_patch_size = patch_size^2
        nea: 邻近patch [batch_size, total_patch_size]
             来自邻近视图的展平patch，与ref具有相同的形状
    
    返回:
        ncc: 归一化互相关损失值 [batch_size, 1, 1, 1]
             值越小表示相似度越高，范围被限制在[0.0, 2.0]
        mask: 有效像素掩码 [batch_size, 1, 1, 1]
              True表示该patch的NCC值小于0.9（高相似度/可靠匹配）
    """
    # ============================================================================
    # 步骤1: 获取batch大小和patch尺寸
    # ============================================================================
    # ref和nea的形状: [batch_size, total_patch_size]
    # 其中 total_patch_size = patch_size * patch_size
    bs, tps = nea.shape  # bs: batch size, tps: total patch size
    patch_size = int(np.sqrt(tps))  # 从总像素数反推patch的边长

    # ============================================================================
    # 步骤2: 计算必要的项并重塑为2D patch
    # ============================================================================
    # 计算ref和nea的逐元素乘积（用于后续计算协方差）
    ref_nea = ref * nea  # [batch_size, total_patch_size]
    
    # 将展平的patch重塑为2D形式，以便使用卷积操作
    # view后的形状: [batch_size, 1, patch_size, patch_size]
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    
    # 计算ref和nea的平方（用于后续计算方差）
    ref2 = ref.pow(2)  # ref^2
    nea2 = nea.pow(2)  # nea^2

    # ============================================================================
    # 步骤3: 使用卷积计算局部区域的和
    # ============================================================================
    # 创建一个全1的卷积核，用于对patch内的所有像素求和
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2  # 填充大小，确保输出尺寸与输入相同
    
    # 使用卷积操作对整个patch求和，相当于计算 Σx_i
    # F.conv2d会在整个patch上滑动，但我们只需要中心点的值（即整个patch的和）
    # [:, :, padding, padding] 提取中心点的值，形状: [batch_size, 1]
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]  # Σ(ref_i^2)
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]  # Σ(nea_i^2)
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]  # Σ(ref_i * nea_i)

    # ============================================================================
    # 步骤4: 计算平均值
    # ============================================================================
    # 计算patch内的均值: μ_ref = Σref_i / N, μ_nea = Σnea_i / N
    # 其中 N = tps (total_patch_size)
    ref_avg = ref_sum / tps  # [batch_size, 1]
    nea_avg = nea_sum / tps  # [batch_size, 1]

    # ============================================================================
    # 步骤5: 计算协方差和方差
    # ============================================================================
    # 计算协方差的分子部分:
    # cross = Σ(ref_i * nea_i) - μ_nea * Σref_i
    #       = Σ(ref_i * nea_i) - μ_nea * N * μ_ref
    # 这是协方差 Cov(ref, nea) 的等价形式（未除以N）
    cross = ref_nea_sum - nea_avg * ref_sum
    
    # 计算ref的方差:
    # ref_var = Σ(ref_i^2) - μ_ref * Σref_i
    #         = Σ(ref_i^2) - μ_ref * N * μ_ref
    #         = Σ(ref_i^2) - N * μ_ref^2
    # 这是方差 Var(ref) 的等价形式（未除以N）
    ref_var = ref2_sum - ref_avg * ref_sum
    
    # 计算nea的方差:
    # nea_var = Σ(nea_i^2) - μ_nea * Σnea_i
    nea_var = nea2_sum - nea_avg * nea_sum

    # ============================================================================
    # 步骤6: 计算归一化互相关系数
    # ============================================================================
    # 计算相关系数的平方:
    # cc = Cov(ref, nea)^2 / (Var(ref) * Var(nea))
    # 这是Pearson相关系数的平方，范围在[0, 1]
    # 值越接近1表示两个patch越相似
    # 添加1e-8防止除零错误
    cc = cross * cross / (ref_var * nea_var + 1e-8)
    
    # 将相似度转换为损失:
    # ncc = 1 - cc
    # 当cc=1（完全相关）时，ncc=0（损失最小）
    # 当cc=0（不相关）时，ncc=1（损失较大）
    ncc = 1 - cc
    
    # 将损失限制在合理范围内 [0.0, 2.0]
    # 避免数值不稳定或异常值的影响
    ncc = torch.clamp(ncc, 0.0, 2.0)
    
    # 对batch维度取平均，保持维度用于后续处理
    # 输出形状: [batch_size, 1, 1, 1]
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    
    # ============================================================================
    # 步骤7: 生成有效像素掩码
    # ============================================================================
    # 创建掩码，标记高质量的匹配（NCC < 0.9）
    # NCC值小于0.9表示相关系数大于0.1，说明patch具有足够的相似性
    # 这个掩码可用于过滤掉不可靠的匹配
    mask = (ncc < 0.9)
    
    return ncc, mask