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
from models.graph_attention_3thmodel_pack.model import build_model

class SmallAttentionBlock(nn.Module):
    def __init__(self, input_dim, reduced_dim, fc_dim, num_heads):
        super(SmallAttentionBlock, self).__init__()
        self.input_reduce = nn.Linear(input_dim, reduced_dim)  # 先进行降维
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
        
class Dual_Backbone(ContinualModel):
    NAME = 'my-model-2-bbs-1112-ub-weight-1+3'
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

        self.net2 = torch.jit.load('/hy-tmp/ViT-L-14.pt', map_location='cpu')
        self.net2 = build_model(self.net2.state_dict())
        self.net2 = self.net2.to(self.device)

        for param in self.net1.parameters():
            param.requires_grad = False

        for param in self.net2.parameters():
            param.requires_grad = False

        self.current_task_num = None
        self.current_eval_num = None

        self.classifier_list = []
        self.gate_list = []
        self.weight_list = []
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
    
            feats1 = self.net1(inputs)
            feats2 = self.net2(inputs)
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
            self.weight_opt.zero_grad()
    
            feats1 = self.net1(inputs)
            feats2 = self.net2(inputs)
            feats = torch.cat((feats1, feats2), dim=1)
            
            weights = torch.softmax(torch.stack(self.weight_list[self.current_task_num]) / self.T, dim=0)
            out = 0
            for i in range(len(self.attention_layers)):
                out += weights[i] * self.attention_layers[i](feats)
                
            out = self.graph_layers[-1](out)  # 通过 graph layer 处理
            outputs = self.classifier_list[-1](out)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss = loss.item()
            self.opt.step()
            self.opt_sched.step()
            self.weight_opt.step()
            self.weight_sched.step()
            self.attention_opt.step()       
            self.attention_sched.step()
            
            # print("classifier learning rate:", self.opt.param_groups[0]['lr'])
            # print("attention layer learning rate:", self.attention_opt.param_groups[0]['lr'])
            # print("gate learning rate:", self.gate_opt.param_groups[0]['lr'])

        return tot_loss
    
    def new_attention_layer(self, train_loader=None, lr=None):
        attention_layer = SmallAttentionBlock(input_dim=768 * 2, reduced_dim=288, fc_dim=288, num_heads=32).to(self.device)
        graph_layer = GraphBlock(input_dim=288, fc_dim=288, num_heads=32).to(self.device)
        self.attention_layers.append(attention_layer)
        self.graph_layers.append(graph_layer)

        if self.current_task_num == 0:
            layers_params = self.attention_layers[self.current_task_num].parameters()
        else:
            attention_params = list(self.attention_layers[self.current_task_num].parameters())
            graph_params = list(self.graph_layers[self.current_task_num].parameters())
            layers_params = attention_params + graph_params
        
        self.attention_opt = torch.optim.Adam(layers_params, lr=0.0005)
        self.attention_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.attention_opt, T_max=len(train_loader) * self.n_epochs, eta_min=1e-5)
        print(f"任务{self.current_task_num}：当前attention长度：{len(self.attention_layers)}；当前graph长度：{len(self.graph_layers)}")
        print("当前attn学习率：", self.attention_opt.param_groups[0]["lr"])

    def new_classifier(self, train_loader=None, out_dim=None):
        classifier = torch.nn.Linear(288, out_dim).to(self.device)
        self.classifier_list.append(classifier)
        params_to_optimize = list(self.classifier_list[-1].parameters())
        
        # 重新初始化优化器
        self.opt = torch.optim.Adam(params_to_optimize, lr=0.0005)
        self.opt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=len(train_loader) * self.n_epochs, eta_min=1e-5)
        print("当前head学习率：", self.opt.param_groups[0]["lr"])
        
        # # 可选: 打印出优化器中包含的参数
        # print("优化器已更新，包含以下参数：")
        # for param in self.opt.param_groups[0]['params']:
        #     print(f"参数形状: {param.shape}, requires_grad: {param.requires_grad}")

    def new_weight(self, train_loader, num_experts):
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
        
        self.weight_list.append(weight)
        self.weight_opt = torch.optim.Adam([param for param in self.weight_list[self.current_task_num]], lr=0.01)
        self.weight_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.weight_opt, T_max=len(train_loader) * self.n_epochs, eta_min=0.00001)
        print("当前weight学习率：", self.weight_opt.param_groups[0]["lr"], "weight_list长度：", len(self.weight_list))

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
