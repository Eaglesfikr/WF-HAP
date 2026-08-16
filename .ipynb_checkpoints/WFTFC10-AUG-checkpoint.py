# 修改版：频域增强改为50%删去+50%增强，输入为已生成的频域文件
# 去掉了时域增强和频域转换，只保留频域增强

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
warnings.filterwarnings('ignore')
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

import tqdm
import pickle
import argparse
import random
import math
import os
import bisect

import dill

from sklearn.utils import shuffle
from DF import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
print(f"Device: {device}")

batch_size = 256
fp16_precision = True
temperature = 0.5
n_views = 2
num_epoches = 100

# === 加载已生成的频域文件（不需要再跑频域转换） ===
data = np.load('./datasets/awf1_freq.npz')
print(data.files)
x_freq = data['x']  # 频域特征
y_train = data['y']
print(x_freq.shape)


class Augmentor():
    def __init__(self):
        self.r_add = 0.1
        self.r_remove = 0.1
        self.lambda_ = 0.8
        self.tau = 0.5
        self.eps = 1e-8

    def augment_frequency(self, log_amp_spec):
        """
        频域增强：对需要修改的比例部分，同时做50%删去 + 50%增强
        不再是"全部增强"或"全部删去"的二选一，而是对同一个序列
        一部分频率点删去（置零），另一部分频率点增强（加噪声）

        输入: log_amp_spec (已经计算好的对数幅度谱)
        输出: 增强后的对数幅度谱
        """
        assert log_amp_spec.shape[0] == 2500, (
            f"频谱长度必须严格为 2500，当前长度为 {log_amp_spec.shape[0]}。"
        )

        # 1. 选择要删去的位置 (r_remove 比例)
        mask_remove = np.random.uniform(0, 1, size=len(log_amp_spec)) < self.r_remove
        # 2. 选择要增强的位置 (r_add 比例)，与删去位置不重叠
        mask_add = np.random.uniform(0, 1, size=len(log_amp_spec)) < self.r_add
        mask_add = mask_add & ~mask_remove  # 排除已被选为删去的位置

        # 3. 执行删去：将选中的频率点置零
        augmented_spec = log_amp_spec.copy().astype(np.float64)
        augmented_spec[mask_remove] = 0.0

        # 4. 执行增强：在选中的频率点上加噪声
        A_max = np.max(log_amp_spec)
        noise_term = mask_add.astype(float) * (self.r_add * A_max)
        augmented_spec += noise_term

        return augmented_spec.astype(np.float32)


# === 主循环：只做强频域增强 ===
print("Initializing WFTFC Augmentor...")
augmentor = Augmentor()

n_samples = len(x_freq)
print(f"Found {n_samples} samples in the dataset.")

freq_len = x_freq.shape[1]
x_freq_aug = np.empty((n_samples, freq_len), dtype=np.float32)
y_train_all = np.empty((n_samples,), dtype=np.float32)

print("Starting frequency augmentation (50% remove + 50% add)...")
for i in tqdm.tqdm(range(n_samples)):
    spec = x_freq[i]
    label = y_train[i]

    # 频域增强（50%删去 + 50%增强）
    spec_aug = augmentor.augment_frequency(spec)

    x_freq_aug[i] = spec_aug
    y_train_all[i] = label

# 保存
print("Saving augmented datasets...")
np.savez_compressed('./datasets/awf1_freq_tfcaugv810.npz', x=x_freq_aug, y=y_train_all)

print("All done.")