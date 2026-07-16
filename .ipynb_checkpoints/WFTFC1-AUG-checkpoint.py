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
print (f"Device: {device}")

batch_size = 256
fp16_precision = True
temperature = 0.5
n_views = 2
num_epoches = 100

data = np.load('./datasets/awf1.npz') 
x_train = data['feature'] #awf1
y_train = data['label']
print(x_train.shape)


def find_bursts(x):
    
    direction = x[0]
    bursts = []
    start = 0
    temp_burst = x[0]
    for i in range(1, len(x)):
        if x[i] == 0.0:
            break
        
        elif x[i] == direction:
            temp_burst += x[i]
            
        else:
            # if temp_burst <= -10 or temp_burst > 0:
            bursts.append((start, i, temp_burst))
            start = i
            temp_burst = x[i]
            direction *= -1
            
    return bursts

class Augmentor():
    def __init__(self):
        self.learning = 5e-4
        self.r_inc_in = 1.0
        self.r_inc_out = 1.0
        self.r_dec_in = 0.5
        self.r_dec_out = 0.5
        self.th_in = 10
        self.th_out = 2
        self.r_add = 0.1
        self.r_remove = 0.1
        self.lambda_ = 0.8
        self.tau = 0.5
        # 固定全局长度参数
        self.trace_len = 5000
        self.eps = 1e-8
            
    def find_bursts(self, x):
        direction = x[0]
        bursts = []
        start = 0
        temp_burst = x[0]
        for i in range(1, len(x)):
            if x[i] == 0.0:
                break

            elif x[i] == direction:
                temp_burst += x[i]

            else:
                # if temp_burst <= -10 or temp_burst > 0:
                bursts.append((start, i, temp_burst))
                start = i
                temp_burst = x[i]
                direction *= -1

        return bursts
        
    def create_trace_from_burst_sizes(self, burst_sizes, length=5000):
        out = []
    
        for size in burst_sizes:
            val = 1 if size > 0 else -1
            out.extend([val] * abs(int(size)))
    
        if len(out) < length:
            out.extend([0] * (length - len(out)))
    
        return np.asarray(out[:length], dtype=np.float32)
    
    
    def augment_time(self, trace):
        """
        WFTFC time-domain augmentation.
    
        Args:
            trace : cell direction sequence
    
        Returns:
            augmented trace
        """
        # 1.bursts extraction
        bursts = find_bursts(trace)
        burst_sizes = [b[2] for b in bursts]
        
        # 2. for incoming burst
        augmented_sizes = []
        for size in burst_sizes:
            p = random.random()
            if p > 0.5:
                    if size < -self.th_in:
                        u = random.random()   # 等价 U(0,1)
                        new_size = size * (1 + u * self.r_inc_in)
                    elif size > self.th_out:
                        u = random.random()
                        new_size = size * (1 + u * self.r_inc_out)
                    else:
                        new_size = size
                        
            else:
                    if size <= -self.th_in:
                        u = random.random()
                        new_size = size * (1 - u * self.r_dec_in)
                    elif size > self.th_out:
                        u = random.random()
                        new_size = size * (1 - u * self.r_dec_out)
                    else:
                        new_size = size
            augmented_sizes.append(new_size)
        augmented_trace = self.create_trace_from_burst_sizes(augmented_sizes)    
        return augmented_trace
        

    def frequency_get(
        self,
        trace,
        target_len=5000,
        fs=1,
        log_scale=True,
        normalize=True
    ):
        """
        时域流量trace → 固定长度5000单边幅度频谱
        1. 先把原始时序截断/补零到 target_len
        2. FFT计算单边幅值谱
        3. 可选对数压缩 + 全局归一化到[0,1]
        :param trace: list / np.ndarray 一维时域流量序列
        :param target_len: 输出频谱固定长度，这里设5000
        :param fs: 采样频率，流量trace默认1
        :param log_scale: 是否20log10对数幅值（流量特征推荐True）
        :param normalize: 是否归一化到0~1
        :return: spec: np.ndarray, shape=(5000,) 固定长度频谱特征
        """
        # 1. 转浮点数组，统一时序长度
        x = np.asarray(trace, dtype=np.float64)
        n_orig = len(x)
        
        if n_orig > target_len:
            # 过长：截断前target_len个点
            x_fixed = x[:target_len]
        else:
            # 过短：末尾补0到target_len
            pad_width = target_len - n_orig
            x_fixed = np.pad(x, pad_width=(0, pad_width), mode="constant", constant_values=0)
    
        # 2. FFT复数结果
        fft_complex = np.fft.fft(x_fixed)
        N = target_len
        # 双边幅值归一化
        amp_full = np.abs(fft_complex) / N
    
        # 3. 截取单边频谱 [0, fs/2]，长度刚好 N//2 = 10000/2=5000
        half_n = N // 2
        spec = amp_full[:half_n]
        spec[1:] *= 2  # 修正幅值能量
    
        # 4. 对数缩放，防止log(0)
        if log_scale:
            spec = 20 * np.log10(spec + 1e-10)
    
        # 5. 归一化到 [0,1]
        if normalize:
            s_min = np.min(spec)
            s_max = np.max(spec)
            if s_max - s_min > 1e-8:
                spec = (spec - s_min) / (s_max - s_min)
        
        # 强制返回固定长度5000频谱
        return spec
        
    def augment_frequency(self, log_amp_spec):
        """
        输入: log_amp_spec (已经计算好的对数幅度谱)
        输出: 增强后的对数幅度谱
        """
        # 随机决定是 Remove 还是 Add
        p = np.random.uniform(0, 1)
        
        if p >= 0.5:
            # --- Remove 模式 ---
            mask = np.random.uniform(0, 1, size=len(log_amp_spec)) < self.r_remove
            # 严格执行伪代码: A_log * mask (保留未被选中的，mask=1代表被选中移除，所以取反)
            # 注意：原伪代码如果是 A_log * mask，且 mask 是稀疏的，那就是只保留选中的。
            # 通常 Remove 是指置零。这里假设 mask=True 表示该位置要被 Remove (置0)。
            augmented_spec = log_amp_spec * (~mask).astype(float)
        else:
            # --- Add 模式 ---
            mask = np.random.uniform(0, 1, size=len(log_amp_spec)) < self.r_add
            # 计算 A_max (当前样本的最大对数幅度)
            A_max = np.max(log_amp_spec)
            noise_term = mask.astype(float) * (self.r_add * A_max)
            augmented_spec = log_amp_spec + noise_term
            
        return augmented_spec

      
    # def augmentor(self, trace):
    #     """
    #     主入口函数
    #     Args:
    #         trace: np.ndarray / list，原始10000长流量轨迹
    #     Returns:
    #         output: np.ndarray shape=(4, 5000)
    #             0: 原始轨迹
    #             1: 时域增强轨迹
    #             2: 原始频谱
    #             3: 增强频谱
    #     """
    #     trace = np.asarray(trace, dtype=np.float32)
    #     # 校验输入长度
    #     assert len(trace) == self.trace_full_len, f"输入轨迹必须是{self.trace_full_len}长度"

    #     # 1. 原轨迹
    #     trace_ori = trace.copy()

    #     # 2. 时域增强轨迹
    #     trace_time_aug = self.augment_time(trace)

    #     # 3. 原始频谱
    #     spec_raw = self.frequency_get(trace)

    #     # 4. 频谱增强
    #     spec_aug = self.freq_augment_spectrum(trace)

    #     # 新增打印，看四条长度
    #     # print("trace_ori len:", len(trace_ori))
    #     # print("trace_time_aug len:", len(trace_time_aug))
    #     # print("spec_raw_10k len:", len(spec_raw_10k))
    #     # print("spec_aug_10k len:", len(spec_aug_10k))
    #     # 拼接4条样本
    #     output = np.stack([trace_ori, trace_time_aug, spec_raw, spec_aug], axis=0)
    #     return output
        


###=======================================================主函数===================================================###

# # 加载数据
# data = np.load('./datasets/awf1.npz')
# x_train = data['feature'] # 假设 shape 是 (N, 5000) 或 (N, 10000)
# y_train = data['label']

print("Initializing WFTFC Augmentor...")
# 修正类名：Augmentor
augmentor = Augmentor()

n_samples = len(x_train)
print(f"Found {n_samples} samples in the dataset.")

# 初始化存储数组
# 注意：rfft 输出的长度是 N/2 + 1，即 2501。
# 如果你坚持要 5000 长度，需要在 frequency_get 里做插值或补零，或者这里改成 2501。
# 这里为了适配 AWF1 原始长度概念，暂时设为 2501 (标准 FFT 一半)，或者你可以手动 pad 到 5000。
# 根据你的代码 frequency_get 返回的是 N//2 (5000)，说明输入是 10000。
# 但你说 AWF1 是 5000。
# 如果输入是 5000，rfft 结果是 2501。
# 为了兼容，这里我们动态获取长度：
dummy_spec = augmentor.frequency_get(x_train[0])
freq_len = len(dummy_spec)

x_time_aug = np.empty((n_samples, augmentor.trace_len), dtype=np.float32)
x_freq = np.empty((n_samples, freq_len), dtype=np.float32)
x_freq_aug = np.empty((n_samples, freq_len), dtype=np.float32)
y_train_all = np.empty((n_samples,), dtype=np.float32)

print("Starting augmentation...")
for i in tqdm.tqdm(range(n_samples)):
    trace = x_train[i]
    label = y_train[i]
    
    # 1. 时域增强
    trace_time_aug = augmentor.augment_time(trace)
    
    # 2. 获取原始频谱 (Log-Amplitude)
    spec = augmentor.frequency_get(trace)
    
    # 3. 频域增强 (直接传入计算好的 spec)
    spec_aug = augmentor.augment_frequency(spec)
    
    # 存入数组
    x_time_aug[i] = trace_time_aug
    x_freq[i] = spec
    x_freq_aug[i] = spec_aug
    y_train_all[i] = label
    
# 保存
print("Saving augmented datasets...")
np.savez_compressed('./datasets/awf1_time_tfcaug.npz', x=x_time_aug, y=y_train_all)
np.savez_compressed('./datasets/awf1_freq.npz', x=x_freq, y=y_train_all)
np.savez_compressed('./datasets/awf1_freq_tfcaug.npz', x=x_freq_aug, y=y_train_all)

print("All done.")

