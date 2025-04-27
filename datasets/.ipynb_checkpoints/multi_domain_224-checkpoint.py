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

from backbone.vit import vit_base_patch16_224_prompt_prototype, VisionTransformer
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from utils import smart_joint
from utils.conf import base_path
from datasets.seq_cub200 import MyCUB200, CUB200, SequentialCUB200
from datasets.seq_tinyimagenet import TinyImagenet, MyTinyImagenet, SequentialTinyImagenet



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

class CombinedDataset(Dataset):
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        """
        Initializes the combined dataset by concatenating data and targets of both datasets.

        Args:
            dataset1 (Dataset): The first dataset.
            dataset2 (Dataset): The second dataset.
        """
        # Concatenate data and targets
        self.data = np.concatenate((dataset1.data, dataset2.data), axis=0)
        self.targets = np.concatenate((dataset1.targets, dataset2.targets), axis=0)

    def __len__(self):
        """
        Returns the total number of samples in the combined dataset.
        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Returns a sample and its target at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (sample, target) corresponding to the index.
        """
        return self.data[idx], self.targets[idx]

def merge_datasets(dataset1: Dataset, dataset2: Dataset, mode: int = 0) -> Dataset:
    """
    Merges two datasets into a single dataset based on the specified mode.

    Args:
        dataset1 (Dataset): The first dataset.
        dataset2 (Dataset): The second dataset.
        mode (int): The merging mode (0, 1, or 2).

    Returns:
        Dataset: The merged dataset.
    """
    def remap_labels(dataset, label_map):
        """
        Remaps the labels of a dataset using a specified mapping.

        Args:
            dataset (Dataset): The dataset whose labels need to be remapped.
            label_map (dict): A dictionary mapping old labels to new labels.

        Returns:
            Dataset: The dataset with remapped labels.
        """
        for i in range(len(dataset.targets)):
            if dataset.targets[i] in label_map:
                dataset.targets[i] = label_map[dataset.targets[i]]
        return dataset

    if mode == 0:
        # Mode 0: Keep dataset1 labels unchanged (0-199), shift dataset2 labels to (200-399)
        label_map_2 = {i: i + 200 for i in range(200)}  # dataset2 label mapping
        dataset2 = remap_labels(dataset2, label_map_2)
        merged_dataset = CombinedDataset(dataset1, dataset2)
    
    elif mode == 1:
        # Mode 1: Remap labels according to specific rules
        # Define label mapping rules
        label_map_1 = {i: i // 20 * 40 + i % 20 for i in range(200)}  # dataset1 label mapping
        label_map_2 = {i: i // 20 * 40 + 20 + i % 20 for i in range(200)}  # dataset2 label mapping

        # Remap labels of both datasets using the defined rules
        dataset1 = remap_labels(dataset1, label_map_1)
        dataset2 = remap_labels(dataset2, label_map_2)

        # Merge the two datasets after remapping
        merged_dataset = CombinedDataset(dataset1, dataset2)

    elif mode == 2:
        # Mode 2: Remap labels with new specified rules
        # Define label mapping rules
        label_map_1 = {i: (i // 10) * 20 + (i % 10) for i in range(200)}  # dataset1 label mapping
        label_map_2 = {i: (i // 10) * 20 + 10 + (i % 10) for i in range(200)}  # dataset2 label mapping

        # Remap labels of both datasets using the defined rules
        dataset1 = remap_labels(dataset1, label_map_1)
        dataset2 = remap_labels(dataset2, label_map_2)

        # Merge the two datasets after remapping
        merged_dataset = CombinedDataset(dataset1, dataset2)

    else:
        raise ValueError("Invalid mode. Mode should be 0, 1, or 2.")

    return merged_dataset



class SequentialMutiDomain(ContinualDataset):
    NAME = 'seq-tinycub'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 20
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    TINY_MEAN, TINY_STD = [0.4802, 0.4480, 0.3975], [0.2770, 0.2691, 0.2821]
    CUB_MEAN, CUB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # SIZE = (MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)
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

    CUB_TRANSFORM = transforms.Compose([
        transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CUB_MEAN, CUB_STD)])
    CUB_TEST_TRANSFORM = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                                             transforms.CenterCrop(MyCUB200.IMG_SIZE),
                                             transforms.ToTensor(),
                                             transforms.Normalize(CUB_MEAN, CUB_STD)])


    # def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
    #     self.task_num += 1

    #     # Load Tiny ImageNet datasets
    #     tiny_train_dataset = MyTinyImagenet(base_path() + 'TINYIMG', train=True, download=True,
    #                                         transform=self.TINY_TRANSFORM)
    #     tiny_test_dataset = TinyImagenet(base_path() + 'TINYIMG', train=False, download=True,
    #                                      transform=self.TINY_TEST_TRANSFORM)

    #     # Load CUB-200 datasets
    #     cub_train_dataset = MyCUB200(base_path() + 'CUB200', train=True, download=True, transform=self.CUB_TRANSFORM)
    #     cub_test_dataset = CUB200(base_path() + 'CUB200', train=False, download=True, transform=self.CUB_TEST_TRANSFORM)

    #     # Combine datasets using the new CombinedDataset class
    #     train_dataset = CombinedTrainDataset(tiny_train_dataset, cub_train_dataset)
    #     test_dataset = CombinedTestDataset(tiny_test_dataset, cub_test_dataset)

    #     # start_c, end_c = self.get_offsets(self.task_num)
    #     # print(f"当前task_idx：{self.task_num}, 标签起始：{start_c}, {end_c}")

        
    #     train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)

    #     return train_loader, test_loader
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        self.task_num += 1

        # Load Tiny ImageNet datasets
        tiny_train_dataset = MyTinyImagenet(base_path() + 'TINYIMG', train=True, download=True,
                                            transform=self.TINY_TRANSFORM)
        tiny_test_dataset = TinyImagenet(base_path() + 'TINYIMG', train=False, download=True,
                                         transform=self.TINY_TEST_TRANSFORM)

        # Load CUB-200 datasets
        cub_train_dataset = MyCUB200(base_path() + 'CUB200', train=True, download=True, transform=self.CUB_TRANSFORM)
        cub_test_dataset = CUB200(base_path() + 'CUB200', train=False, download=True, transform=self.CUB_TEST_TRANSFORM)

        # Modify labels of CUB-200 datasets to avoid overlap
        for i in range(len(cub_train_dataset.targets)):
            cub_train_dataset.targets[i] += 200
        for i in range(len(cub_test_dataset.targets)):
            cub_test_dataset.targets[i] += 200

        # for i in range(len(cub_train_dataset.targets)):
        #     tiny_train_dataset.targets[i] += 200
        # for i in range(len(cub_test_dataset.targets)):
        #     tiny_test_dataset.targets[i] += 200
            
        # # Combine datasets using the new CombinedDataset class
        # train_dataset = CombinedTrainDataset(tiny_train_dataset, cub_train_dataset)
        # test_dataset = CombinedTestDataset(tiny_test_dataset, cub_test_dataset)

        # start_c, end_c = self.get_offsets(self.task_num)
        # print(f"当前task_idx：{self.task_num}, 标签起始：{start_c}, {end_c}")
        

        if self.task_num < 10:
            print("从TinyImageNet中取数据")
            # print_label_distribution(tiny_train_dataset)
            # print_label_distribution(tiny_test_dataset)

            train_dataset = tiny_train_dataset
            test_dataset = tiny_test_dataset
        else:
            print("从CUB200中取数据")
            train_dataset = cub_train_dataset
            test_dataset = cub_test_dataset
            
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
        model = VisionTransformer(num_classes=SequentialTinyImagenet.N_CLASSES)
        model.load_state_dict(torch.load('vit_model_weights_tinyimg.pth'))
        return model

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)

    # @staticmethod
    # def get_denormalization_transform():
    #     transform = DeNormalize(SequentialCUB200.MEAN, SequentialCUB200.STD)
    #     return transform

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    # def get_class_names(self):
    #     if self.class_names is not None:
    #         return self.class_names
    #     classes = fix_class_names_order(CLASS_NAMES, self.args)
    #     self.class_names = classes
    #     return self.class_names

    # TINY_TRANSFORM = transforms.Compose(
    #     [transforms.Resize(SIZE, interpolation=InterpolationMode.BICUBIC),
    #      transforms.RandomHorizontalFlip(),
    #      transforms.ToTensor(),  # 确保转换为 CHW 格式
    #      transforms.Normalize(TINY_MEAN, TINY_STD)]
    # )
    # TINY_TEST_TRANSFORM = transforms.Compose([
    #     transforms.Resize(SIZE, interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),  # 确保转换为 CHW 格式
    #     transforms.Normalize(TINY_MEAN, TINY_STD)
    # ])

    # CUB_TRANSFORM = transforms.Compose([
    #     transforms.Resize(SIZE, interpolation=InterpolationMode.BICUBIC),  # 调整为统一大小
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),  # 确保转换为 CHW 格式
    #     transforms.Normalize(CUB_MEAN, CUB_STD)])
    #
    # CUB_TEST_TRANSFORM = transforms.Compose([
    #     transforms.Resize(SIZE, interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),  # 确保转换为 CHW 格式
    #     transforms.Normalize(CUB_MEAN, CUB_STD)])
