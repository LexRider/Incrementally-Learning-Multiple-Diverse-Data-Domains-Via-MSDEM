#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.utils.lora_utils import LoRALayer


class ClipLinear(nn.Linear, LoRALayer):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def forward(self, x: torch.Tensor, AB: dict = None):

        def T(w):
            return w.transpose(1, 2) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if AB is not None:
            A = None
            if isinstance(AB, dict):
                B = AB['B']
                A = AB.get('A')
            else:
                B = AB
            if A is not None:
                res = (B @ (A @ torch.permute(x, (1, 2, 0)).unsqueeze(1))).sum(1)
                return result + torch.permute(res, (2, 0, 1))
            res = (B @ torch.permute(x, (1, 2, 0)).unsqueeze(1)).sum(1)
            return result + torch.permute(res, (2, 0, 1))

        return result

# class DeepHead(nn.Module):
#     def __init__(self, input_dim=768, hidden_dim=256, output_dim=20):
#         super(DeepHead, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim, bias=True),  # 输入层到第一个隐藏层
#             nn.GELU(),  # 激活函数
#             nn.Linear(hidden_dim, hidden_dim, bias=True),  # 第一个隐藏层到第二个隐藏层
#             nn.GELU(),  # 激活函数
#             nn.Linear(hidden_dim, hidden_dim, bias=True),  # 第二个隐藏层到第三个隐藏层
#             nn.GELU(),  # 激活函数
#             nn.Linear(hidden_dim, output_dim, bias=True)  # 输出层
#         )
#         self.out_features = output_dim

#     def forward(self, x):
#         return self.net(x)

class DeepHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=20):
        super(DeepHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True)  # 输出层
        )
        self.out_features = output_dim

    def forward(self, x):
        return self.net(x)


class IncrementalClassifier(nn.Module):

    def __init__(self, embed_dim: int, nb_classes: int):
        """
        Incremental classifier for continual learning.

        Args:
            embed_dim: int, dimension of the input features.
            nb_classes: int, number of classes to classify.
        """

        super().__init__()

        self.embed_dim = embed_dim        
        self.old_state_dict = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 更换一个更深的分类头
        # heads = [nn.Linear(embed_dim, nb_classes, bias=True)]
        heads = [DeepHead(input_dim=embed_dim, hidden_dim=256, output_dim=nb_classes).to(self.device)]
        self.heads = nn.ModuleList(heads)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update(self, nb_classes: int, freeze_old=True):
        """
        Add a new head to the classifier.

        Args:
            nb_classes, number of classes to add.
            freeze_old: bool, whether to freeze the old heads.
        """
        
        # 更换一个更深的分类头
        _fc = DeepHead(input_dim=self.embed_dim, hidden_dim=256, output_dim=nb_classes).to(self.device)
        # 初始化新的头部的权重与偏置
        for layer in _fc.net:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                    
        # _fc = nn.Linear(self.embed_dim, nb_classes, bias=True).to(self.device)
        # nn.init.trunc_normal_(_fc.weight, std=.02)
        # nn.init.constant_(_fc.bias, 0)

        if freeze_old:
            for param in self.heads.parameters():
                param.requires_grad = False

        self.heads.append(_fc)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Compute the logits for each head and concatenate them.

        Args:
            x: torch.Tensor, input features.
        """
        return torch.cat([h(x) for h in self.heads], dim=1)

    # def forward(self, x: torch.Tensor, selected_head: int = None):
    #     """
    #     Forward pass.

    #     Compute the logits for each head and concatenate them.

    #     Args:
    #         x: torch.Tensor, input features.
    #         selected_head: int, the index of the selected head. If None, return all heads' outputs.
    #     """
    #     outputs = []
    #     for i, head in enumerate(self.heads):
    #         if selected_head is None or i == selected_head:
    #             outputs.append(head(x))  # 保留当前任务的输出
    #         else:
    #             outputs.append(torch.zeros_like(head(x)))  # 将其他任务的输出设为0
    #     return torch.cat(outputs, dim=1)
