import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from typing import Tuple
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import pandas as pd

from backbone.vit import vit_base_patch16_224_prompt_prototype, VisionTransformer
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from utils import smart_joint
from utils.conf import base_path
from datasets.seq_cub200 import MyCUB200, CUB200, SequentialCUB200
from datasets.seq_tinyimagenet import TinyImagenet, MyTinyImagenet, SequentialTinyImagenet
from datasets.seq_cifar100 import TCIFAR100, MyCIFAR100
from datasets.seq_cifar10 import TCIFAR10, MyCIFAR10
from backbone.ResNetBottleneck import resnet50


class TrainBirdDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        # 加载 .npz 文件中的数据
        data = np.load(npz_file)
        self.data = data['data']  # 图像数据
        self.targets = torch.tensor(data['targets']).long()  # 类别标签，转换为长整型张量
        
        # 如果传入了 transform，保存 transform
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # 从张量中获取图像数据
        label = self.targets[idx]  # 获取对应的标签

        # 将 NumPy 数组转换为 PIL Image
        img = Image.fromarray(img.astype(np.uint8))  # 确保 img 为 uint8 格式
        
        # 如果传入 transform，则对图像进行增强
        if self.transform:
            img = self.transform(img)

        return img, label, img  # 返回增强后的图像、标签和原始图像

class TestBirdDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        # 加载 .npz 文件中的数据
        data = np.load(npz_file)
        self.data = data['data']  # 图像数据
        self.targets = torch.tensor(data['targets']).long()  # 类别标签，转换为长整型张量
        
        # 如果传入了 transform，保存 transform
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # 从张量中获取图像数据
        label = self.targets[idx]  # 获取对应的标签

        # 将 NumPy 数组转换为 PIL Image
        img = Image.fromarray(img.astype(np.uint8))  # 确保 img 为 uint8 格式
        
        # 如果传入 transform，则对图像进行增强
        if self.transform:
            img = self.transform(img)

        return img, label  # 返回增强后的图像和标签


# def is_valid_image(img_path):
#     img_full_path = os.path.join('/hy-tmp/birds525', img_path)  # 构建完整路径
#     try:
#         with Image.open(img_full_path) as img:
#             return img.size[0] == 224 and img.size[1] == 224 and img.mode == 'RGB'
#     except:
#         return False  # 如果图像无法打开，直接认为它无效


# def get_sub_path_df():
#     data_dir = '/hy-tmp/birds525'
#     paths_df = pd.read_csv(os.path.join(data_dir, "birds.csv"))
#     valid_image_mask = paths_df['filepaths'].apply(is_valid_image)
#     filtered_paths_df = paths_df[valid_image_mask]

#     # 过滤出 class id 在 0 到 199 之间的样本
#     sub_paths_df = filtered_paths_df[filtered_paths_df['class id'].between(0, 199)]

#     return sub_paths_df

def print_label_distribution(dataset):
    # 初始化一个 Counter 对象来统计标签数量
    label_counts = Counter()

    # 遍历数据集并统计每个标签的数量
    for _, label, _ in tqdm(dataset):
        label_counts[label] += 1

    # 打印标签分布信息
    total_labels = sum(label_counts.values())
    print("Label Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"Label {label}: {count} ({count / total_labels:.2%})")

class SequentialMutiDomain(ContinualDataset):
    NAME = 'seq-cifar100tiny'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 15
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    
    TINY_MEAN, TINY_STD = [0.4802, 0.4480, 0.3975], [0.2770, 0.2691, 0.2821]
    BIRD_MEAN, BIRD_STD = [0.4758, 0.4685, 0.3870], [0.2376, 0.2282, 0.2475]
    # CIFAR100_MEAN, CIFAR100_STD = [0, 0, 0], [1, 1, 1]
    CIFAR100_MEAN, CIFAR100_STD = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    CIFAR10_MEAN, CIFAR10_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]
    SIZE = (224, 224)

    task_num = -1

    TINY_TRANSFORM = transforms.Compose([
        transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD)])
    TINY_TEST_TRANSFORM = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(TINY_MEAN, TINY_STD)])

    BIRD_TRAIN_TRANSFORM = T.Compose([
        T.RandomCrop(224, padding=4, padding_mode='reflect'),
        T.RandomApply(torch.nn.ModuleList([T.GaussianBlur(kernel_size=3, sigma=(0.2, 5))]), p=0.15),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(BIRD_MEAN, BIRD_STD)])

    BIRD_TEST_TRANSFORM = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(BIRD_MEAN, BIRD_STD)])
    # paths_df = get_sub_path_df()
    # train_df = paths_df[paths_df['data set'] == 'train']
    # test_df = paths_df[paths_df['data set'] == 'test']
    # valid_df = paths_df[paths_df['data set'] == 'valid']

    CIFAR100_TRANSFORM = transforms.Compose(
        [transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )
    CIFAR100_TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])    

    CIFAR10_TRANSFORM = transforms.Compose(
        [transforms.Resize(224),
         transforms.RandomCrop(224, padding=28),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(CIFAR10_MEAN, CIFAR100_STD)])

    CIFAR10_TEST_TRANSFORM = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        self.task_num += 1

        # Load TinyImageNet datasets
        tiny_train_dataset = MyTinyImagenet(base_path() + 'TINYIMG', train=True, download=True, transform=self.TINY_TRANSFORM)
        tiny_test_dataset = TinyImagenet(base_path() + 'TINYIMG', train=False, download=True, transform=self.TINY_TEST_TRANSFORM)

        # # Load BIRD-200 datasets
        # bird_train_dataset = TrainBirdDataset(npz_file='/hy-tmp/data/BIRDS200/train_birds_data.npz', transform=self.BIRD_TRAIN_TRANSFORM)
        # bird_test_dataset = TestBirdDataset(npz_file='/hy-tmp/data/BIRDS200/test_birds_data.npz', transform=self.BIRD_TEST_TRANSFORM)

        # Load CIFAR100 datasets
        cifar100_train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True, download=True, transform=self.CIFAR100_TRANSFORM)
        cifar100_test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False, download=True, transform=self.CIFAR100_TEST_TRANSFORM)

        # # Load CIFAR10 datasets
        # cifar10_train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True, download=True, transform=self.CIFAR10_TRANSFORM)
        # cifar10_test_dataset = TCIFAR10(base_path() + 'CIFAR10', train=False, download=True, transform=self.CIFAR10_TEST_TRANSFORM)
        
        # # 查看训练集中的样本数量
        # print(f"训练集样本数量: {len(bird_train_dataset)}")
        # print(f"测试集样本数量: {len(bird_test_dataset)}")

        # Modify labels of tiny datasets to avoid overlap
        for i in range(len(tiny_train_dataset.targets)):
            tiny_train_dataset.targets[i] += 100
        for i in range(len(tiny_test_dataset.targets)):
            tiny_test_dataset.targets[i] += 100
        
        # # Modify labels of birds-200 datasets to avoid overlap
        # for i in range(len(bird_train_dataset.targets)):
        #     bird_train_dataset.targets[i] += 20
        # for i in range(len(bird_test_dataset.targets)):
        #     bird_test_dataset.targets[i] += 20
        
        # # Modify labels of birds-200 datasets to avoid overlap
        # for i in range(len(bird_train_dataset.targets)):
        #     bird_train_dataset.targets[i] += 20
        # for i in range(len(bird_test_dataset.targets)):
        #     bird_test_dataset.targets[i] += 20

        # # Modify labels of CIFAR100 datasets to avoid overlap
        # for i in range(len(cifar100_train_dataset.targets)):
        #     cifar100_train_dataset.targets[i] += 200
        # for i in range(len(cifar100_test_dataset.targets)):
        #     cifar100_test_dataset.targets[i] += 200

        # # Modify labels of CIFAR10 datasets to avoid overlap
        # for i in range(len(cifar10_train_dataset.targets)):
        #     cifar10_train_dataset.targets[i] += 200
        # for i in range(len(cifar10_test_dataset.targets)):
        #     cifar10_test_dataset.targets[i] += 200

        # start_c, end_c = self.get_offsets(self.task_num)
        # print(f"当前task_idx：{self.task_num}, 标签起始：{start_c}, {end_c}")

        if self.task_num < 5:
            print("从cifar100中取数据")
            # print_label_distribution(tiny_train_dataset)
            # print_label_distribution(tiny_test_dataset)

            train_dataset = cifar100_train_dataset
            test_dataset = cifar100_test_dataset
        
        # elif 10 <= self.task_num < 11:
        #     print("从CIFAR10中取数据")
        #     train_dataset = cifar10_train_dataset
        #     test_dataset = cifar10_test_dataset
        
        else:
            print("从tiny中取数据")
            train_dataset = tiny_train_dataset
            test_dataset = tiny_test_dataset
            
        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)

        return train_loader, test_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        # Use the ViT backbone and load pre-trained weights
        model = VisionTransformer(num_classes=200)
        model.load_state_dict(torch.load('vit_model_weights_in21k.pth'))
        model.head = torch.nn.Linear(768, SequentialMutiDomain.N_CLASSES)
        return model
        # return resnet50(SequentialMutiDomain.N_CLASSES_PER_TASK * SequentialMutiDomain.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

