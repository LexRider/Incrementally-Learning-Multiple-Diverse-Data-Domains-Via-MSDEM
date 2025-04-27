import random
from collections import Counter
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from backbone.utils import layers
from backbone.vit import vit_base_patch16_224_prompt_prototype, VisionTransformer
from backbone.resnet34 import BirdResnet

from tqdm import tqdm  # 引入 tqdm 库用于显示进度条
from joblib import Parallel, delayed
from npeet import entropy_estimators as ee  # 正确导入 NPEET
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

# 定义 MINE 网络
import torch.nn as nn

class MineNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, dropout_prob=0.5):
        super(MineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size + output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout_prob)  # 添加 Dropout 层，p 是丢弃概率
        
    def forward(self, x, y):
        combined = torch.cat([x, y], dim=1)
        h = torch.relu(self.fc1(combined))
        h = self.dropout(h)  # 在第一层后添加 Dropout
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)  # 在第二层后添加 Dropout
        return self.fc3(h)

class Dual_Backbone(ContinualModel):
    NAME = 'my-model-2-backbones-0923-vv'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                            ' Fine-tuning Vision Transformer.')
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        # 设置 GPU 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建预训练的模型并绑定到模型中，net1和net2都是vit，net1用于训练TinyImageNet，net2用于训练cub200
        self.net1 = VisionTransformer(num_classes=200)
        self.net1.load_state_dict(torch.load('vit_model_weights_in21k_ft_in1k_ft_tinyimg_6blk.pth'))
        # self.net1.load_state_dict(torch.load('vit_model_weights_ft_525_12blk.pth'))
        self.net1 = self.net1.to(self.device)

        self.net2 = VisionTransformer(num_classes=200)
        self.net2.load_state_dict(torch.load('vit_model_weights_ft_525_12blk.pth'))
        # self.net2.load_state_dict(torch.load('vit_model_weights_in21k_ft_in1k_ft_tinyimg_12blk.pth'))
        self.net2 = self.net2.to(self.device)

        self.vit_extractor = VisionTransformer(num_classes=200)
        self.vit_extractor.load_state_dict(torch.load('vit_model_weights_tinyimg_in21k.pth'))
        self.vit_extractor = self.vit_extractor.to(self.device)

        for param in self.net1.parameters():
            param.requires_grad = False

        for param in self.net2.parameters():
            param.requires_grad = False

        for param in self.vit_extractor.parameters():
            param.requires_grad = False

        self.num_classes = self.net.num_classes
        self.embed_dim = self.net.embed_dim
        self.depth = self.net.depth
        self.num_heads = self.net.num_heads

        self.current_task_num = None  # 追踪当前训练任务的索引
        self.current_eval_num = None  # 追踪当前评估任务的索引

        # 设置vit的head管理器，设置resnet34的分类管理器（保证resnet的输出维度和vit一致）
        self.head = layers.IncrementalClassifier(self.embed_dim, 20).to(self.device)

        self.weight_list = []
        self.weight1 = 0
        self.weight2 = 0
        self.weight_opt = None
        self.T = 1.0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        观察方法用于训练模型。
        """

        # 将输入和标签移动到 GPU
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.opt.zero_grad()

        feats1 = self.net1(inputs, returnt='features')
        feats2 = self.net2(inputs, returnt='features')
        
        sumw = torch.exp(self.weight1 / self.T)  + torch.exp(self.weight2 / self.T)
        feats = torch.exp(self.weight1 / self.T) / sumw * feats1 + torch.exp(self.weight2 / self.T) / sumw * feats2

        outputs = self.head(feats)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss = loss.item()
        
        self.opt.step()
        
        self.weight_opt.step()       
        self.weight_list[self.current_task_num] = [self.weight1.item(), self.weight2.item()]
        # print("\nweight list的内容：\n", self.weight_list)
        
        # # 打印优化器中的参数、梯度和是否 requires_grad=True
        # for param_group in self.weight_opt.param_groups:
        #     for param in param_group['params']:
        #         print(f'Parameter value: {param.data}')
        #         print(f'Gradient: {param.grad}')
        #         print(f'requires_grad: {param.requires_grad}')
        return tot_loss

    def compute_mutual_information(self, train_loader, num_samples=2000, 
                                   use_pca_input=False, pca_input_components=1000, 
                                   use_pca_output=False, pca_output_components=300,
                                   use_vit_input=True, vit_model=None,
                                   method="MINE", mine_network1=MineNetwork(768, 768, 64), mine_network2=MineNetwork(768, 768, 64)):
        """
        从 train_loader 中随机选取 num_samples 个样本，计算两个网络的输出特征相对输入样本的互信息，
        并根据互信息值初始化两个网络的权重。
    
        Args:
            train_loader (DataLoader): PyTorch 的数据加载器，用于加载训练样本。
            num_samples (int): 随机选取的样本数量，默认为 1000。
            use_pca_input (bool): 是否启用 PCA 对输入进行降维。
            pca_input_components (int): PCA 降维时保留的输入主成分数量。
            use_vit_input (bool): 是否使用预训练的 ViT 模型提取输入特征。
            vit_model_name (str): 预训练的 ViT 模型名称（用于从 HuggingFace 加载）。
            method (str): 互信息的计算方法，可以是 "NPEET" 或 "MINE"。
            mine_network (torch.nn.Module): 如果使用 MINE，需要传入训练好的神经网络来计算互信息。
            
        Returns:
            weight1 (float): 第一个网络的权重，基于其与输入样本的互信息。
            weight2 (float): 第二个网络的权重，基于其与输入样本的互信息。
        """
        
        print("开始计算互信息：")
        net1 = self.net1
        net2 = self.net2
    
        # 将所有批次的数据样本加载并展平为单个样本列表
        all_data = []
        for batch in train_loader:
            aug_data, labels, original_inputs = batch  # 取增强后的样本，保留原始样本
            for i in range(aug_data.size(0)):  # 遍历每个 batch 内的样本
                all_data.append((aug_data[i].cpu(), labels[i].cpu()))  # 将样本移到 CPU
    
        # 确保抽取的样本数量不超过总样本数量
        num_samples = min(num_samples, len(all_data))
    
        # 随机选取 num_samples 个样本
        sampled_data = random.sample(all_data, num_samples)
    
        # 分离出输入样本和对应的标签
        sampled_inputs = torch.stack([data[0] for data in sampled_data], dim=0)
        sampled_labels = torch.stack([data[1] for data in sampled_data], dim=0)
    
        # 将输入数据移到设备上
        device = net1.device
        sampled_inputs = sampled_inputs.to(device)
        net1.to(device)
        net2.to(device)
    
        # 根据选择使用 PCA、ViT 或其他方法进行输入降维
        if use_pca_input:
            # 获取输入数据的 numpy 格式
            inputs_numpy = sampled_inputs.cpu().numpy().reshape(sampled_inputs.shape[0], -1)
            print(f"启用 PCA 对输入数据进行降维，将输入数据降到 {pca_input_components} 维度。")
            pca_input = PCA(n_components=pca_input_components)
            inputs_numpy = pca_input.fit_transform(inputs_numpy)
            print("PCA 输入完成。")
        
        elif use_vit_input:
            # 使用 ViT 预训练模型提取特征
            print(f"使用 ViT 模型提取输入特征：")
            inputs_numpy = vit_model(sampled_inputs, returnt='features').cpu().numpy()
            print(f"ViT 特征提取完成，特征形状: {inputs_numpy.shape}")
        
        else:
            inputs_numpy = sampled_inputs.cpu().numpy().reshape(sampled_inputs.shape[0], -1)
    
        # 获取两个网络的输出特征（假设特征作为输出）
        with torch.no_grad():
            output1 = net1(sampled_inputs, returnt='features').cpu().numpy()  # net1 输出特征
            output2 = net2(sampled_inputs, returnt='features').cpu().numpy()  # net2 输出特征
    
        # 对输出进行 PCA 降维（如果需要）
        if use_pca_output:
            print(f"启用 PCA 对输出数据进行降维，将输出数据降到 {pca_output_components} 维度。")
    
            # 对输出1降维
            pca_output = PCA(n_components=pca_output_components)
            output1 = pca_output.fit_transform(output1)
            print("输出1每个主成分的方差比例:", pca_output.explained_variance_ratio_)
    
            # 对输出2降维
            output2 = pca_output.fit_transform(output2)
            print("输出2每个主成分的方差比例:", pca_output.explained_variance_ratio_)
    
        ### 计算互信息
        print(f"开始计算网络1和网络2的互信息，方法: {method}...")
    
        if method == "NPEET":
            # 使用 NPEET 计算互信息
            mi_net1 = ee.mi(inputs_numpy, output1)
            mi_net2 = ee.mi(inputs_numpy, output2)
        elif method == "MINE":
            assert mine_network1 is not None, "使用 MINE 计算互信息时需要提供 mine_network 和 optimizer"
            assert mine_network2 is not None, "使用 MINE 计算互信息时需要提供 mine_network 和 optimizer"
            
            mine_network1.to(device)
            mine_network2.to(device)
            
            # 转换为 tensor 并送入设备
            inputs_tensor = torch.tensor(inputs_numpy, dtype=torch.float32).to(device)
            output_tensor1 = torch.tensor(output1, dtype=torch.float32).to(device)
            output_tensor2 = torch.tensor(output2, dtype=torch.float32).to(device)
            
            # 使用 Adam 优化器

            optimizer1 = torch.optim.Adam(mine_network1.parameters(), lr=0.001, weight_decay=1e-4)
            optimizer2 = torch.optim.Adam(mine_network2.parameters(), lr=0.001, weight_decay=1e-4)
            input_tensor1 = inputs_tensor
            input_tensor2 = inputs_tensor
            
            # 使用 MINE 计算 net1 和 net2 的互信息
            mine_network1.train()
            mine_network2.train()
            mi_net1 = self.train_mine(mine_network1, optimizer1, input_tensor1, output_tensor1)
            mi_net2 = self.train_mine(mine_network2, optimizer2, input_tensor2, output_tensor2)
        else:
            raise ValueError(f"未知的互信息计算方法: {method}")
    
        # 设置一个温度系数 T，T 值越小差异越大，T 值越大差异越小
        T = 2.0  # 你可以调节 T 值，比如 0.5、0.1 等
        
        exp_mi_net1 = np.exp(mi_net1 / T)
        exp_mi_net2 = np.exp(mi_net2 / T)
        
        sum_exp_mi = exp_mi_net1 + exp_mi_net2
        
        weight1 = exp_mi_net1 / sum_exp_mi
        weight2 = exp_mi_net2 / sum_exp_mi
    
        print(f"网络1的互信息: {mi_net1:.4f}, 权重: {weight1:.4f}")
        print(f"网络2的互信息: {mi_net2:.4f}, 权重: {weight2:.4f}")
    
        # 将权重赋值给模型
        self.weight_list.append([weight1, weight2])
        self.weight1 = torch.tensor(weight1, device=self.device, requires_grad=True)
        self.weight2 = torch.tensor(weight2, device=self.device, requires_grad=True)
        self.weight_opt = torch.optim.Adam([self.weight1, self.weight2], 0.001)
   
    def train_mine(self, mine_network, optimizer, x, y, num_epochs=1000, lr_scheduler_type="Cosine", step_size=200, gamma=0.1):
        """
        使用 MINE 进行互信息估计的训练函数，并添加学习率调度器。
        Args:
            mine_network (torch.nn.Module): 用于估计互信息的 MINE 网络。
            optimizer (torch.optim.Optimizer): 用于优化 MINE 网络的优化器。
            x (Tensor): 网络的输入。
            y (Tensor): 网络的输出。
            num_epochs (int): 训练的轮数。
            lr_scheduler_type (str): 学习率调度器类型，默认是 "StepLR"。
            step_size (int): 学习率调度器 StepLR 的 step size，默认为 100。
            gamma (float): 学习率调度器 StepLR 的学习率衰减因子，默认为 0.1。
    
        Returns:
            mi_estimate (float): 估计的互信息。
        """
    
        # 初始化学习率调度器
        if lr_scheduler_type == "StepLR":
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif lr_scheduler_type == "Cosine":
        # 使用 CosineAnnealingLR 调度器
            scheduler = CosineAnnealingLR(optimizer, T_max=1000)
        else:
            raise ValueError(f"未知的学习率调度器类型: {lr_scheduler_type}")
        
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
    
            # MINE 损失函数的计算
            joint = mine_network(x, y)
            marginals = mine_network(x, y[torch.randperm(y.size(0))])
    
            loss = -(joint.mean() - torch.log(marginals.exp().mean()))
            loss.backward()
            optimizer.step()
            # print(loss.item())
            
            # 更新学习率调度器
            scheduler.step()
        
        # 最终的互信息估计值
        with torch.no_grad():
            mi_estimate = joint.mean() - torch.log(marginals.exp().mean())
    
        return mi_estimate.item()

    def update_optimizer(self, lr=None):
        """
        更新优化器以包括新的参数。
        Args:
            lr (float): 可选的学习率，默认为当前优化器的学习率。
        """
        if lr is None:
            # 如果没有提供新的学习率，则使用现有优化器的学习率
            lr = self.opt.param_groups[0]['lr']
        
        # 获取当前需要优化的参数，包括 self.head 和 net2 中 requires_grad 为 True 的参数
        params_to_optimize = list(self.head.parameters())
        
        # 加入 net2 中的可训练参数
        params_to_optimize += [param for param in self.net2.parameters() if param.requires_grad]
        
        # 重新初始化优化器
        self.opt = torch.optim.Adam(params_to_optimize, lr=lr)
        
        # # 可选: 打印出优化器中包含的参数
        # print("优化器已更新，包含以下参数：")
        # for param in self.opt.param_groups[0]['params']:
        #     print(f"参数形状: {param.shape}, requires_grad: {param.requires_grad}")


    def select_head(self, task_id):
        """
        根据任务ID选择对应的分类头，并设置 self.head 的输出结果。
    
        Args:
            task_id (int): 任务的索引
        """
        assert task_id < len(self.head.heads), "任务ID超出范围"
        self.current_eval_num = task_id  # 设置当前任务编号

        # 创建一个新的 forward 方法，用于根据任务编号选择输出头
        def forward_with_selected_head(x):
            # 获取所有头的输出
            all_outputs = [head(x) for head in self.head.heads]

            # 拼接所有头的输出
            concatenated_outputs = torch.cat(all_outputs, dim=1)

            # 初始化masked_outputs为与concatenated_outputs相同形状的零张量
            masked_outputs = torch.zeros_like(concatenated_outputs)

            # 计算每个头的输出范围
            current_index = 0
            for i, head in enumerate(self.head.heads):
                head_output_size = head.out_features  # 当前头的输出大小
                if i == task_id:
                    # 如果是选定的头，则保留其输出
                    masked_outputs[:, current_index:current_index + head_output_size] = \
                        concatenated_outputs[:, current_index:current_index + head_output_size]
                # 更新当前索引到下一个头的位置
                current_index += head_output_size

            # 对选定头的输出执行softmax
            selected_start_idx = task_id * self.head.heads[0].out_features
            selected_end_idx = selected_start_idx + self.head.heads[task_id].out_features
            masked_outputs[:, selected_start_idx:selected_end_idx] = F.softmax(
                masked_outputs[:, selected_start_idx:selected_end_idx], dim=1)

            return masked_outputs

        # 替换现有的 head 的 forward 方法
        self.head.forward = forward_with_selected_head

    def print_all_heads_parameters(self):
        """
        打印 self.heads 列表中每个分类头的参数，且每个参数只打印前50个值，
        每个数值保留小数点后五位。
        """
        for i, head in enumerate(self.resnet34_classifers):
            print(f"\n分类头 {i} 的参数:")
            for name, param in head.named_parameters():
                # 打印参数名和形状
                print(f"参数名: {name}, 形状: {param.shape}")

                # 获取前50个参数值，并保留小数点后五位
                formatted_values = [f"{v:.5f}" for v in param.data.view(-1)[:50]]
                print(f"前50个值: {formatted_values}")

    def check_net_parameters_frozen(self):
        """
        检查 self.net 中的所有参数是否被冻结（requires_grad 为 False）。
        """
        all_frozen = True  # 初始化标志为 True
        print("以下为net1的参数冻结情况")
        for name, param in self.net1.named_parameters():
            if param.requires_grad:
                print(f"参数 {name} 没有被冻结（requires_grad 为 True）")
                all_frozen = False
            else:
                print(f"参数 {name} 已被冻结（requires_grad 为 False）")

        print("以下为net2的参数冻结情况")
        for name, param in self.net2.named_parameters():
            if param.requires_grad:
                print(f"参数 {name} 没有被冻结（requires_grad 为 True）")
                all_frozen = False
            else:
                print(f"参数 {name} 已被冻结（requires_grad 为 False）")

        if all_frozen:
            print("所有参数都已被冻结。")
        else:
            print("部分参数未被冻结。")
