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

data = np.load('./datasets/proteus_day90_sign.npz') 
x_train = data['X'] #awf1
y_train = data['y']


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
        self.trace_full_len = 10000
        self.spec_raw_len = 5000
            
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
        
    def create_trace_from_burst_sizes(self, burst_sizes, length=10000):
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
        target_len=10000,
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
        
    def freq_augment_spectrum(
        self,
        spec: np.ndarray,
        # 频带分段（单边谱固定长度2500）
        low_cutoff: int = 800,
        mid_cutoff: int = 1600,
        # 低频掩码阈值t
        t: float = 0.3,
        # 公式固定超参（按你要求修改默认值）
        rho: float = 2.0,
        alpha: float = 3.0,
        beta: float = 2.0,
        gamma: float = 2.0,
        eps: float = 5e-3,
        # 实验复现随机种子
        seed: int = None
    ) -> np.ndarray:
        """
        单边频谱频域增强，重点增强中高频、抑制低频
        :param spec: 输入单边幅度谱，shape=(5000,)，值域[0,1]
        :param low_cutoff: 低频段结束下标 [0, low_cutoff) = 低频
        :param mid_cutoff: 中频段结束下标 [low_cutoff, mid_cutoff) = 中频；[mid_cutoff, 2500) = 高频
        :param rho, alpha, beta, gamma, eps: 高斯映射公式超参
        :param seed: 随机种子，实验复现用
        :return: aug_spec: 增强后频谱，shape=(2500,)
        """
        if seed is not None:
            np.random.seed(seed)
        G = spec.copy()
        N = len(G)
        # print(N)
        assert N == 5000, "输入必须是5000长度单边频谱"
    
        # ========== Step1：划分低/中/高频带，分段计算μ、σ ==========
        band_split = [0, low_cutoff, mid_cutoff, N]
        mu = np.zeros(N)
        sigma = np.zeros(N)
        for i in range(3):
            start, end = band_split[i], band_split[i+1]
            band_data = G[start:end]
            band_mu = np.mean(band_data)
            band_std = np.std(band_data)
            mu[start:end] = band_mu
            sigma[start:end] = band_std

        # 因为是频带，已经算好了，不需要自动计算低频阈值 t（仅使用低频段统计量）取03就行
        # low_band = G[:low_cutoff]
        # mu_low = np.mean(low_band)
        # std_low = np.std(low_band)
        # t = mu_low + k * std_low  # 自动生成阈值，不再外部传入
        t = 0.3
        # ========== Step2：构建掩码M，抑制低频分量 ==========
        # M[v] = 1 当频点均值 < t（低频区域），否则0
        M = np.where(mu < t, 1.0, 0.0)
        # G_M = (1-M) 逐元素乘原始频谱，抹除低频，只保留中高频
        G_M = (1 - M) * G
        # print(len(G_M))
    
        # ========== Step3：高斯映射生成平滑频域网格G_H ==========
        numerator = np.square(G_M - alpha * mu)
        denominator = 2 * np.square(beta * sigma)
        exp_term = np.exp(- numerator / (denominator + eps)) #架构eps防止除0
        G_H = np.power(rho * exp_term, gamma) + eps
    
        # ========== Step4：构造协调分布Φ ~ N(1, G_H²)，采样增益 ==========
        # 每个频点独立正态分布，均值1，标准差=G_H[v]
        gain = np.random.normal(loc=1.0, scale=G_H, size=N)
        # 防止增益为负（频谱幅值不能为负）
        gain = np.clip(gain, a_min=eps, a_max=None)
    
        # ========== Step5：原始频谱 × 协调增益，输出增强频谱 ==========
        aug_spec = G * gain
        # 可选：重新归一化回[0,1]，保持输入值域统一
        aug_min = np.min(aug_spec)
        aug_max = np.max(aug_spec)
        if aug_max - aug_min > eps:
            aug_spec = (aug_spec - aug_min) / (aug_max - aug_min)
    
        return aug_spec

    def pad_spec_to_full(self, spec_5k):
        """5000长度频谱后补0至10000"""
        pad = np.zeros(self.trace_full_len - self.spec_raw_len, dtype=np.float32)
        full_spec = np.concatenate([spec_5k, pad])
        return full_spec
        
    def augmentor(self, trace):
        """
        主入口函数
        Args:
            trace: np.ndarray / list，原始10000长流量轨迹
        Returns:
            output: np.ndarray shape=(4, 10000)
                0: 原始轨迹(10000)
                1: 时域增强轨迹(10000)
                2: 原始频谱(5000补0→10000)
                3: 增强频谱(5000补0→10000)
        """
        trace = np.asarray(trace, dtype=np.float32)
        # 校验输入长度
        assert len(trace) == self.trace_full_len, f"输入轨迹必须是{self.trace_full_len}长度"

        # 1. 原轨迹
        trace_ori = trace.copy()

        # 2. 时域增强轨迹
        trace_time_aug = self.augment_time(trace)

        # 3. 原始频谱 + 补0到10000
        spec_raw_5k = self.frequency_get(trace)
        spec_raw_10k = self.pad_spec_to_full(spec_raw_5k)

        # 4. 频谱增强 + 补0到10000
        spec_aug_5k = self.freq_augment_spectrum(spec_raw_5k)
        spec_aug_10k = self.pad_spec_to_full(spec_aug_5k)

        # 新增打印，看四条长度
        # print("trace_ori len:", len(trace_ori))
        # print("trace_time_aug len:", len(trace_time_aug))
        # print("spec_raw_10k len:", len(spec_raw_10k))
        # print("spec_aug_10k len:", len(spec_aug_10k))
        # 拼接4条样本
        output = np.stack([trace_ori, trace_time_aug, spec_raw_10k, spec_aug_10k], axis=0)
        return output
        


###=======================================================主函数===================================================###
# 初始化增强器实例
aug_tool = Augmentor()

# ========== 下面这段burst统计代码完全不用改 ==========
outgoing_burst_sizes = []
x_random = x_train[np.random.choice(range(len(x_train)), size=1000, replace=False)]
for x in x_random:
    bursts = aug_tool.find_bursts(x)  # 这里也要改，调用实例方法
    outgoing_burst_sizes += [b[2] for b in bursts if b[2] > 0]
max_outgoing_burst_size = max(outgoing_burst_sizes)

bins = max(1, int(np.ceil(max_outgoing_burst_size - 1)))
count, bins = np.histogram(outgoing_burst_sizes, bins=bins)
PDF = count/np.sum(count)
OUTGOING_BURST_SIZE_CDF = np.zeros_like(bins)
OUTGOING_BURST_SIZE_CDF[1:] = np.cumsum(PDF)
# ====================================================

print("Generating WFHAP augmented dataset...")
n = len(x_train)

# 每条原始样本生成4条10000长度数据，总样本数 4*n
total_aug_num = 4 * n
# 维度：(4*n, 10000)
x_train_aug = np.empty((total_aug_num, aug_tool.trace_full_len), dtype=np.float32)
y_train_aug = np.empty((total_aug_num,), dtype=np.float32)

for i in tqdm.tqdm(range(n)):
    trace = x_train[i]  # 输入原始10000长轨迹
    label = y_train[i]
    
    # 调用主方法，一次性得到4条样本 shape=(4,10000)
    four_samples = aug_tool.augmentor(trace)
    
    # 填入数组：第4*i、4*i+1、4*i+2、4*i+3 分别对应四类样本
    x_train_aug[4*i]     = four_samples[0]  # 原轨迹 10000
    x_train_aug[4*i + 1] = four_samples[1]  # 时域增强 10000
    x_train_aug[4*i + 2] = four_samples[2]  # 原始频谱(5k补0→10k)
    x_train_aug[4*i + 3] = four_samples[3]  # 频谱增强(5k补0→10k)
    
    # 标签全部相同
    y_train_aug[4*i : 4*i + 4] = label

print("Augmented train shape:", x_train_aug.shape)
print("Augmented label shape:", y_train_aug.shape)

np.savez_compressed(
    "datasets/protues_day90_WFHAPaug.npz",
    x_train=x_train_aug,
    y_train=y_train_aug
)

print("Saved to datasets/protues_day90_WFHAPaug.npz")
print(x_train_aug.dtype)
print(x_train_aug.shape)
print("WFCMemory:",
      x_train_aug.nbytes / 1024 / 1024,
      "MB")

##====================================================================test====================================================================##
# A = Augmentor()
# data = np.load('./datasets/protues_WFHAPaug.npz')
# print(data.files)
# np.set_printoptions(threshold=np.inf)
# x1 = data['x_train'][0]
# x1_f = A.frequency_get(x1)
# print(x1_f)
# print(len(x1_f))
# x1_f_aug = A.freq_augment_spectrum(x1_f)
# print(x1_f_aug)