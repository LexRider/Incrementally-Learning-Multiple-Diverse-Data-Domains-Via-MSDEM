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
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim import Adam
from collections import Counter

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from collections import Counter
import torch.nn.functional as F

class SmallAttentionBlock(nn.Module):
    def __init__(self, input_dim, reduced_dim, fc_dim, num_heads):
        super(SmallAttentionBlock, self).__init__()
        self.input_reduce = nn.Linear(input_dim, reduced_dim)  # 先进行降维
        # self.input_reduce = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, reduced_dim)
        # )
        self.multihead_attention = nn.MultiheadAttention(embed_dim=reduced_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(reduced_dim)
        self.fc = nn.Linear(reduced_dim, fc_dim)

    def forward(self, x):
        x = self.input_reduce(x).unsqueeze(0)  # 降维后再输入到 attention 中
        # x = x.unsqueeze(0)
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.layer_norm(attn_output + x)
        output = self.fc(x.squeeze(0))
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

# class GateBlock(nn.Module):
#     def __init__(self, input_dim, num_experts, noise_stddev=0, max_grad_norm=1.0, hidden_dim=256, temperature=2.0):
#         super(GateBlock, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_experts)
#         )

#         self.noise_stddev = noise_stddev  # 噪声标准差
#         self.max_grad_norm = max_grad_norm  # 梯度裁剪阈值
#         self.temperature = temperature  # Gumbel-Softmax 温度参数

#         for param in self.fc.parameters():
#             param.register_hook(lambda grad: torch.clamp(grad, -self.max_grad_norm, self.max_grad_norm))

#         self.token = True
#         self.start_gate_values = None
#         self.end_gate_values = None

#     def gumbel_softmax_sample(self, logits, temperature):
#         """
#         Gumbel-Softmax 采样函数，通过加入 Gumbel 噪声增强稀疏性。
#         """
#         # 采样 Gumbel 噪声
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
#         # 加入 Gumbel 噪声后再应用 softmax
#         return F.softmax((logits + gumbel_noise) / temperature, dim=-1)

#     def forward(self, x):
#         # 在专家维度 (dim=0) 上取平均，形状变为 (batch_size, input_dim)
#         avg_x = x.mean(dim=0)  # (batch_size, input_dim)
        
#         # 计算门控权重，通过 Gumbel-Softmax 实现稀疏选择
#         gate_values = self.fc(avg_x)  # (batch_size, num_experts)
#         noise = torch.randn_like(gate_values, device=gate_values.device) * self.noise_stddev
#         gate_values = gate_values + noise
#         noisy_gate_values = self.gumbel_softmax_sample(gate_values, self.temperature)  # 使用 Gumbel-Softmax

#         # 记录初始和最终的 gate_values
#         if self.token:
#             self.start_gate_values = noisy_gate_values
#             self.token = False
#         self.end_gate_values = noisy_gate_values
        
#         # 转置并调整 noisy_gate_values 形状，以便与 x 匹配
#         noisy_gate_values = noisy_gate_values.T.unsqueeze(-1)  # (num_experts, batch_size, 1)
        
#         # 使用 noisy_gate_values 对专家输出进行加权组合
#         gated_output = (x * noisy_gate_values).sum(dim=0)  # (batch_size, feature_dim)
#         return gated_output

class GateBlock(nn.Module):
    def __init__(self, input_dim, num_experts, noise_stddev=0, max_grad_norm=1.0, hidden_dim=256):
        super(GateBlock, self).__init__()
        # self.fc = nn.Linear(input_dim, num_experts)  # 门控权重生成层
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        self.noise_stddev = noise_stddev  # 噪声标准差
        self.max_grad_norm = max_grad_norm  # 梯度裁剪阈值

        # 为每个参数注册钩子，用于在计算梯度后进行裁剪
        for param in self.fc.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -self.max_grad_norm, self.max_grad_norm))

        # 初始化其他属性
        self.token = True
        self.start_gate_values = None
        self.end_gate_values = None

    def forward(self, x):
        # 在专家维度 (dim=0) 上取平均，形状变为 (batch_size, input_dim)
        avg_x = x.mean(dim=0)  # (batch_size, input_dim)
        
        # 计算门控权重，在专家维度 (dim=-1) 上进行 softmax 归一化
        gate_values = torch.softmax(self.fc(avg_x), dim=-1)

        # 加入噪声，确保噪声与 gate_values 在同一设备上
        noise = torch.randn_like(gate_values, device=gate_values.device) * self.noise_stddev
        noisy_gate_values = gate_values + noise
        # print("noisy_gate_values.shape before softmax", noisy_gate_values.shape)
        noisy_gate_values = torch.softmax(noisy_gate_values, dim=1)  # 在专家维度归一化

        # 记录初始和最终的 gate_values
        if self.token:
            self.start_gate_values = noisy_gate_values
            self.token = False
        self.end_gate_values = noisy_gate_values
        
        # print("x.shape", x.shape)
        # print("noisy_gate_values.shape after softmax", noisy_gate_values.shape)
        
        # 转置并调整 noisy_gate_values 形状，以便与 x 匹配
        noisy_gate_values = noisy_gate_values.T.unsqueeze(-1)  # (num_experts, batch_size, 1)
        # 使用 noisy_gate_values 对专家输出进行加权组合
        gated_output = (x * noisy_gate_values).sum(dim=0)  # (batch_size, feature_dim)
        # print("gated_output", gated_output.shape)
        # gated_output = (x * noisy_gate_values.unsqueeze(-1)).sum(dim=0)  # (batch_size, feature_dim)
        return gated_output
        
class Dual_Backbone(ContinualModel):
    NAME = 'my-model-2-bbs-1112-upper-bound'
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

        self.input_dim = 768 * 2
        self.reduced_dim=288
        self.fc_dim=288

        self.current_task_num = None
        self.current_eval_num = None

        self.classifier_list = []
        self.gate_list = []
        self.T = 0.5

        self.attention_layers = []
        self.graph_layers = []
        self.attention_opt = None
        self.weight_opt = None
        self.gate_opt = None
        self.attention_sched = None
        self.weight_sched = None
        self.gate_sched = None
        self.opt_sched = None
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
    
            outputs = self.classifier_list[-1](out)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss = loss.item()
            self.opt.step()
            self.opt_sched.step()
            
            self.attention_opt.step()       
            self.attention_sched.step()
            # print("classifier learning rate:", self.opt.param_groups[0]['lr'])
            # print("attention layer learning rate:", self.attention_opt.param_groups[0]['lr'])
            # print("gate learning rate:", self.gate_opt.param_groups[0]['lr'])

        else:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            self.attention_opt.zero_grad()
            self.gate_opt.zero_grad()
    
            feats1 = self.net1(inputs, returnt='features')
            feats2 = self.net2(inputs, returnt='features')
            feats = torch.cat((feats1, feats2), dim=1)
            
            expert_outputs = [att(feats) for att in self.attention_layers]  # 获取所有专家输出
            expert_outputs = torch.stack(expert_outputs)  # 形状 (num_experts, batch_size, feature_dim)
            gated_output = self.gate_list[-1](expert_outputs)
            out = self.graph_layers[-1](gated_output)  # 通过 graph layer 处理
            
            outputs = self.classifier_list[-1](out)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss = loss.item()
            self.opt.step()
            self.opt_sched.step()
            self.gate_opt.step()
            self.gate_sched.step()
            self.attention_opt.step()       
            self.attention_sched.step()
            # print("classifier learning rate:", self.opt.param_groups[0]['lr'])
            # print("attention layer learning rate:", self.attention_opt.param_groups[0]['lr'])
            # print("gate learning rate:", self.gate_opt.param_groups[0]['lr'])

        return tot_loss
    
    def new_attention_layer(self, train_loader=None, lr=None):
        attention_layer = SmallAttentionBlock(input_dim=768 * 2, reduced_dim=768, fc_dim=768, num_heads=1).to(self.device)
        graph_layer = GraphBlock(input_dim=768, fc_dim=768, num_heads=1).to(self.device)
        self.attention_layers.append(attention_layer)
        self.graph_layers.append(graph_layer)

        if self.current_task_num == 0:
            layers_params = self.attention_layers[self.current_task_num].parameters()
        else:
            attention_params = list(self.attention_layers[self.current_task_num].parameters())
            graph_params = list(self.graph_layers[self.current_task_num].parameters())
            layers_params = attention_params + graph_params
        
        self.attention_opt = torch.optim.Adam(layers_params, lr=0.001)
        self.attention_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.attention_opt, T_max=len(train_loader) * self.n_epochs, eta_min=1e-5)
        print(f"任务{self.current_task_num}：当前attention长度：{len(self.attention_layers)}；当前graph长度：{len(self.graph_layers)}")
        print("当前attn学习率：", self.attention_opt.param_groups[0]["lr"])

    def new_classifier(self, train_loader=None, out_dim=None):
        classifier = torch.nn.Linear(768, out_dim).to(self.device)
        self.classifier_list.append(classifier)
        params_to_optimize = list(self.classifier_list[-1].parameters())
        
        # 重新初始化优化器
        self.opt = torch.optim.Adam(params_to_optimize, lr=0.001)
        self.opt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=len(train_loader) * self.n_epochs, eta_min=1e-5)
        print("当前head学习率：", self.opt.param_groups[0]["lr"])
        
        # # 可选: 打印出优化器中包含的参数
        # print("优化器已更新，包含以下参数：")
        # for param in self.opt.param_groups[0]['params']:
        #     print(f"参数形状: {param.shape}, requires_grad: {param.requires_grad}")

    def new_gate(self, train_loader, num_experts):
        if num_experts != 0:
            gate = GateBlock(input_dim=768, num_experts=num_experts).to(self.device)
            self.gate_list.append(gate)
            self.gate_opt = torch.optim.Adam(self.gate_list[-1].parameters(), lr=0.001)
            self.gate_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.gate_opt, T_max=len(train_loader) * self.n_epochs, eta_min=0.00001)
            print("当前gate学习率：", self.gate_opt.param_groups[0]["lr"])
        else:
            gate = GateBlock(input_dim=768, num_experts=1).to(self.device)
            self.gate_list.append(gate)
            print(f"训练第{num_experts}个任务，新建一个占位gate")

    def print_optimizer_parameters(self):
        """
        打印所有优化器中包含的参数信息，包含参数的形状和是否参与梯度计算 (requires_grad)。
        """
        # 定义要检查的优化器列表（可以根据你的实际情况调整）
        optimizers = {
            'main_optimizer': self.opt,
            'attention_optimizer': self.attention_opt,
            'gate_optimizer': self.gate_opt
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
        print(f"Total trainable parameters: {total_params}\n")
        return total_params
          
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
