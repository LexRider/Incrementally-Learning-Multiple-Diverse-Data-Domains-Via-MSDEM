import random
from collections import Counter
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

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
# from npeet import entropy_estimators as ee  # 正确导入 NPEET
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, fc_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, fc_dim)

    def forward(self, x):
        # 输入形状为 [batch_size, input_dim]，需要扩展为 [1, batch_size, input_dim]
        x = x.unsqueeze(0)
        attn_output, _ = self.multihead_attention(x, x, x)
        
        # 残差连接 + LayerNorm
        x = self.layer_norm(attn_output + x)
        output = self.fc(x.squeeze(0))  # 再次去掉序列维度

        return output

class GraphBlock(nn.Module):
    def __init__(self, input_dim, fc_dim, num_heads):
        super(GraphBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, fc_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.layer_norm(attn_output + x)
        output = self.fc(x.squeeze(0))  # 再次去掉序列维度

        return output

class Dual_Backbone(ContinualModel):
    NAME = 'my-model-2-backbones-1011-vv-graph_attention'
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

        # 创建预训练的模型并绑定到模型中，net1和net2都是vit
        # net1为预训练在imagenet21K后微调在imagenet1K得到的vit；
        # net2为预训练在imagenet21K得到的vit；
        self.net1 = VisionTransformer(num_classes=200)
        self.net1.load_state_dict(torch.load('vit_model_weights_in21k_ft_in1k.pth'))
        self.net1 = self.net1.to(self.device)

        self.net2 = VisionTransformer(num_classes=200)
        self.net2.load_state_dict(torch.load('vit_model_weights_in21k.pth'))
        self.net2 = self.net2.to(self.device)

        for param in self.net1.parameters():
            param.requires_grad = False

        for param in self.net2.parameters():
            param.requires_grad = False

        self.num_classes = self.net.num_classes
        self.embed_dim = self.net.embed_dim
        self.depth = self.net.depth
        self.num_heads = self.net.num_heads

        self.current_task_num = None  # 追踪当前训练任务的索引
        self.current_eval_num = None  # 追踪当前评估任务的索引

        # 设置head管理器
        self.head = layers.IncrementalClassifier(self.embed_dim, 100).to(self.device)

        self.weight_list = []
        self.weight1 = torch.tensor(1.0)
        self.weight2 = torch.tensor(0)
        self.T = 0.5

        self.attention_layers = []
        self.graph_layers = []

        self.attention_opt = None
        self.weight_opt = None
        self.attention_sched = None
        self.weight_sched = None
        self.n_epochs = args.n_epochs

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.current_task_num == 0:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            self.attention_opt.zero_grad()
    
            feats1 = self.net1(inputs, returnt='features')
            feats2 = self.net2(inputs, returnt='features')
            feats = torch.cat((feats1, feats2), dim=1)
            out = self.attention_layers[self.current_task_num](feats)
    
            
            outputs = self.head(out)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss = loss.item()
            self.opt.step()
            # print("attention layer learning rate:", self.attention_opt.param_groups[0]['lr'])
            self.attention_opt.step()       
            self.attention_sched.step()

        else:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            self.attention_opt.zero_grad()
            self.weight_opt.zero_grad()
    
            feats1 = self.net1(inputs, returnt='features')
            feats2 = self.net2(inputs, returnt='features')
            feats = torch.cat((feats1, feats2), dim=1)
            
            # 对当前任务的权重进行softmax操作
            weights = torch.softmax(torch.stack(self.weight_list[self.current_task_num]) / self.T, dim=0)
            out = 0
            for i in range(len(self.attention_layers)):
                out += weights[i] * self.attention_layers[i](feats)

            out = self.graph_layers[self.current_task_num](out)
            
            outputs = self.head(out)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss = loss.item()
            self.opt.step()
            # print("attention layer learning rate:", self.attention_opt.param_groups[0]['lr'])
            # print("weight layer learning rate:", self.weight_opt.param_groups[0]['lr'])
            self.attention_opt.step()       
            self.attention_sched.step()
            self.weight_opt.step()
            self.weight_sched.step()
        
        return tot_loss
    
    def new_attention_layer(self, train_loader):
        
        attention_layer = AttentionBlock(input_dim=768 * 2, fc_dim=768, num_heads=1).to(self.device)
        graph_layer = GraphBlock(input_dim=768, fc_dim=768, num_heads=1).to(self.device)
        
        if self.current_task_num == 0:
            weight = [torch.nn.Parameter(torch.ones(1, device=self.device), requires_grad=False)]
        else:
            # 初始化所有的weight为1
            weight = [torch.nn.Parameter(torch.ones(1, device=self.device), requires_grad=True) for _ in range(self.current_task_num + 1)]
            # 将最后一个元素改为2
            weight[-1] = torch.nn.Parameter(torch.tensor([1.0], device=self.device), requires_grad=True)
        # else:
        #     # 使用指数增长的方式初始化权重
        #     weight = [torch.nn.Parameter(torch.exp(torch.tensor(float(i), device=self.device)), requires_grad=True) for i in range(self.current_task_num + 1)]

        self.attention_layers.append(attention_layer)
        self.graph_layers.append(graph_layer)
        self.weight_list.append(weight)

        if self.current_task_num == 0:
            layers_params = self.attention_layers[self.current_task_num].parameters()
        else:
            attention_params = list(self.attention_layers[self.current_task_num].parameters())
            graph_params = list(self.graph_layers[self.current_task_num].parameters())
            layers_params = attention_params + graph_params
        
        self.attention_opt = torch.optim.Adam(layers_params, lr=1e-4)
        self.attention_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.attention_opt, T_max=len(train_loader) * self.n_epochs, eta_min=1e-6)
        
        self.weight_opt = torch.optim.Adam([param for param in self.weight_list[self.current_task_num]], lr=5e-3)
        self.weight_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.weight_opt, T_max=len(train_loader) * self.n_epochs, eta_min=1e-5)
        
        print(f"任务{self.current_task_num}：当前attention长度：{len(self.attention_layers)}；当前graph长度：{len(self.graph_layers)}；创建的weight的长度：{len(weight)}")

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
    
    def print_optimizer_parameters(self):
        """
        打印所有优化器中包含的参数信息，包含参数的形状和是否参与梯度计算 (requires_grad)。
        """
        # 定义要检查的优化器列表（可以根据你的实际情况调整）
        optimizers = {
            'main_optimizer': self.opt,
            'attention_optimizer': self.attention_opt,
            'weight_optimizer': self.weight_opt
        }
        
        # 遍历每个优化器
        for opt_name, optimizer in optimizers.items():
            if optimizer is None:
                print(f"{opt_name} 未定义或未初始化。")
                continue
            
            print(f"\n{opt_name} 包含的参数如下：")
            # 遍历优化器中的每个参数组
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param is not None:
                        print(f"参数形状: {param.shape}, requires_grad: {param.requires_grad}")
                    else:
                        print("None 参数")
    
    def calculate_qkv_ranks(self, attention_layer, threshold=1e-5):
        """
        计算并打印 attention layer 中 Q、K、V 三个矩阵的秩。
        
        Args:
        - attention_layer (nn.Module): 实例化的 AttentionBlock 层。
        - threshold (float): 判定奇异值为 0 的阈值，默认值为 1e-5。
        """
        
        def calculate_matrix_rank(weight_matrix, threshold=threshold):
            # 对权重矩阵进行 SVD 分解
            u, s, v = torch.svd(weight_matrix)
            # 计算大于阈值的奇异值数量，即为矩阵的近似秩
            rank = torch.sum(s > threshold).item()
            return rank, s
        
        # 处理 attention layer 中的 in_proj_weight，即 Q, K, V 三个拼接矩阵
        for name, param in attention_layer.named_parameters():
            if 'in_proj_weight' in name and param.requires_grad:
                weight_matrix = param.data
                embed_dim = weight_matrix.shape[1]
                
                # 将 in_proj_weight 切分为 Q, K, V 三个矩阵
                Q = weight_matrix[:embed_dim, :]    # Q 的形状为 (1536, 1536)
                K = weight_matrix[embed_dim:2*embed_dim, :]  # K 的形状为 (1536, 1536)
                V = weight_matrix[2*embed_dim:, :]  # V 的形状为 (1536, 1536)
                
                print("\n[Attention Layer - Q矩阵] 秩信息:")
                rank_Q, singular_values_Q = calculate_matrix_rank(Q)
                print(f"  形状: {Q.shape}")
                print(f"  近似秩: {rank_Q}")
                print(f"  奇异值（前10个）: {singular_values_Q[:10]}")
    
                print("\n[Attention Layer - K矩阵] 秩信息:")
                rank_K, singular_values_K = calculate_matrix_rank(K)
                print(f"  形状: {K.shape}")
                print(f"  近似秩: {rank_K}")
                print(f"  奇异值（前10个）: {singular_values_K[:10]}")
    
                print("\n[Attention Layer - V矩阵] 秩信息:")
                rank_V, singular_values_V = calculate_matrix_rank(V)
                print(f"  形状: {V.shape}")
                print(f"  近似秩: {rank_V}")
                print(f"  奇异值（前10个）: {singular_values_V[:10]}")
    
        print("\nQ、K、V矩阵的秩计算完成。")
    
    def print_trainable_params(self, optimizer):
        total_params = 0
        for group_idx, group in enumerate(optimizer.param_groups):
            print(f"Param Group {group_idx}:")
            for param_idx, param in enumerate(group['params']):
                if param.requires_grad:
                    param_size = param.numel()
                    print(f"  Param {param_idx}: shape={param.shape}, size={param_size}")
                    total_params += param_size
                else:
                    print(f"  Param {param_idx}: shape={param.shape}, size={param.numel()} (not trainable)")
        print(f"\nTotal trainable parameters: {total_params}")
        return total_params

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
