import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from models.utils.continual_model import ContinualModel

from utils.buffer import Buffer
from backbone.utils import layers
from backbone.vit import vit_base_patch16_224_prompt_prototype, VisionTransformer
from backbone.resnet34 import BirdResnet

from tqdm import tqdm  # 引入 tqdm 库用于显示进度条
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from backbone import MammothBackbone

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, fc_dim, num_heads):  # 添加self
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

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  # 添加self
        super(Classifier, self).__init__()  # 纠正类名
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),  # 输入层到第一个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, hidden_dim, bias=True),  # 第一个隐藏层到第二个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, hidden_dim, bias=True),  # 第二个隐藏层到第三个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, output_dim, bias=True)  # 输出层
        )
    def forward(self, x):
        return self.net(x)

def unfreeze_blocks(vit_model, n):
    
    if n > len(vit_model.blocks):
        raise ValueError(f"The model has only {len(vit_model.blocks)} blocks, but {n} were requested to unfreeze.")

    # Unfreeze the last n blocks
    for i in range(-n, 0):  # Negative index to access last n blocks
        for param in vit_model.blocks[i].parameters():
            param.requires_grad = True

    # Unfreeze norm and fc_norm layers
    if hasattr(vit_model, 'norm') and vit_model.norm is not None:
        for param in vit_model.norm.parameters():
            param.requires_grad = True

    if hasattr(vit_model, 'fc_norm') and vit_model.fc_norm is not None:
        for param in vit_model.fc_norm.parameters():
            param.requires_grad = True

    print(f"Last {n} blocks, self.norm, and self.fc_norm have been unfrozen.")
    
class DualVit(MammothBackbone):

    def __init__(self, unfreeze_blk=1, output_dim = None):
        super(DualVit, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net1 = VisionTransformer(num_classes=200)
        self.net1.load_state_dict(torch.load('vit_model_weights_in21k_ft_in1k.pth'))
        self.net1 = self.net1.to(self.device)

        self.net2 = VisionTransformer(num_classes=200)
        self.net2.load_state_dict(torch.load('vit_model_weights_in21k.pth'))
        self.net2 = self.net2.to(self.device)
        
        # Freeze all parameters in both models
        for param in self.net1.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False

        # Unfreeze specific blocks
        unfreeze_blocks(self.net1, n=unfreeze_blk)
        unfreeze_blocks(self.net2, n=unfreeze_blk)
        
        self.attention_layer = AttentionBlock(input_dim=768*2, fc_dim=768, num_heads=1)
        self.classifier = Classifier(input_dim=768, hidden_dim=1024, output_dim=output_dim)
    
    def forward(self, x): 
        feats1 = self.net1(x, returnt='features')
        feats2 = self.net2(x, returnt='features')
        feats = torch.cat((feats1, feats2), dim=1)
        out = self.attention_layer(feats)
        out = self.classifier(out)
        return out



        
        