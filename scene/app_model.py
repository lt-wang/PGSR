import torch
import torch.nn as nn
import os

def searchForMaxIteration(folder):
    """
    在文件夹中搜索最大的迭代次数
    
    参数:
        folder: 文件夹路径
    
    返回:
        最大的迭代次数
    """
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

class AppModel(nn.Module):
    """
    外观模型（Appearance Model）
    用于处理不同图像之间的曝光和颜色差异
    为每张图像学习一个2D的外观编码（affine参数）
    """
    def __init__(self, num_images=1600):
        """
        初始化外观模型
        
        参数:
            num_images: 数据集中的图像数量
        """
        super().__init__()   
        # 为每张图像学习2个参数（通常用于affine变换：a和b，即 y = ax + b）
        self.appear_ab = nn.Parameter(torch.zeros(num_images, 2).cuda())
        self.optimizer = torch.optim.Adam([
                                {'params': self.appear_ab, 'lr': 0.001, "name": "appear_ab"},
                                ], betas=(0.9, 0.99))
            
    def save_weights(self, model_path, iteration):
        """
        保存模型权重
        
        参数:
            model_path: 模型保存路径
            iteration: 当前迭代次数
        """
        out_weights_path = os.path.join(model_path, "app_model/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        print(f"save app model. path: {out_weights_path}")
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'app.pth'))

    def load_weights(self, model_path, iteration=-1):
        """
        加载模型权重
        
        参数:
            model_path: 模型路径
            iteration: 要加载的迭代次数，-1表示加载最新的
        """
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "app_model"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "app_model/iteration_{}/app.pth".format(loaded_iter))
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)
