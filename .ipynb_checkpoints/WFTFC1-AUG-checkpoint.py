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
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
print (f"Device: {device}")

batch_size = 256
fp16_precision = True
temperature = 0.5
n_views = 2
num_epoches = 100

data = np.load('./datasets/awf1.npz') 
x_train = data['feature'] #awf1
y_train = data['label']


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

def create_trace_from_burst_sizes(burst_sizes, length=5000):
    out = []

    for size in burst_sizes:
        val = 1 if size > 0 else -1
        out.extend([val] * abs(int(size)))

    if len(out) < length:
        out.extend([0] * (length - len(out)))

    return np.asarray(out[:length], dtype=np.float32)


def augment_time(trace,
                 r_in=0.3,
                 r_out=0.3):
    """
    WFTFC time-domain augmentation.

    Args:
        trace : cell direction sequence
        r_in  : maximum perturbation ratio of incoming bursts
        r_out : maximum perturbation ratio of outgoing bursts

    Returns:
        augmented trace
    """

    bursts = find_bursts(trace)
    burst_sizes = [b[2] for b in bursts]

    # p ~ U(0,1)
    p = random.random()

    new_sizes = []

    for size in burst_sizes:

        if size > 0:
            ratio = r_out
        else:
            ratio = r_in

        if p > 0.5:
            # increase
            delta = random.uniform(0.0, ratio)
            new_size = int(round(size * (1 + delta)))
        else:
            # decrease
            delta = random.uniform(0.0, ratio)
            new_size = int(round(size * (1 - delta)))

        # 保证至少保留一个 cell
        if size > 0:
            new_size = max(1, new_size)
        else:
            new_size = min(-1, new_size)

        new_sizes.append(new_size)

    return create_trace_from_burst_sizes(new_sizes)
    

def augment_frequency(trace,
                      remove_ratio=0.05,
                      add_ratio=0.05,
                      noise_scale=0.1):
    """
    WFTFC frequency-domain augmentation.

    Args:
        trace : cell direction sequence
        remove_ratio : ratio of removed frequency components
        add_ratio    : ratio of added frequency components
        noise_scale  : maximum amplitude perturbation

    Returns:
        augmented trace
    """

    # FFT
    spectrum = np.fft.fft(trace)

    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    N = len(spectrum)

    # 只选择正频率（不包含DC和Nyquist）
    if N % 2 == 0:
        candidates = np.arange(1, N // 2)
    else:
        candidates = np.arange(1, (N + 1) // 2)

    p = random.random()

    if p > 0.5:
        # ---------- Add ----------
        num = max(1, int(add_ratio * len(candidates)))

        idx = np.random.choice(candidates,
                               size=num,
                               replace=False)

        for i in idx:
            j = N - i

            delta = random.uniform(0.0, noise_scale)

            magnitude[i] *= (1.0 + delta)
            magnitude[j] *= (1.0 + delta)

    else:
        # ---------- Remove ----------
        num = max(1, int(remove_ratio * len(candidates)))

        idx = np.random.choice(candidates,
                               size=num,
                               replace=False)

        for i in idx:
            j = N - i

            magnitude[i] = 0.0
            magnitude[j] = 0.0

    # 重建频谱
    spectrum_new = magnitude * np.exp(1j * phase)

    # IFFT
    augmented = np.fft.ifft(spectrum_new).real

    return augmented.astype(trace.dtype)
    


###=======================================================主函数===================================================###
outgoing_burst_sizes = []
x_random = x_train[np.random.choice(range(len(x_train)), size=1000, replace=False)]
for x in x_random:
    bursts = find_bursts(x)
    outgoing_burst_sizes += [x[2] for x in bursts if x[2] > 0]
max_outgoing_burst_size = max(outgoing_burst_sizes)


bins = max(1, int(np.ceil(max_outgoing_burst_size - 1)))
count, bins = np.histogram(outgoing_burst_sizes, bins=bins)
PDF = count/np.sum(count)
OUTGOING_BURST_SIZE_CDF = np.zeros_like(bins)
OUTGOING_BURST_SIZE_CDF[1:] = np.cumsum(PDF)

print("Generating WFTFC augmented dataset...")

n = len(x_train)

x_train_aug = np.empty((2 * n, 5000), dtype=x_train.dtype)
y_train_aug = np.empty((2 * n,), dtype=y_train.dtype)

for i in tqdm.tqdm(range(n)):
    trace = x_train[i]
    label = y_train[i]

    x_train_aug[2 * i] = augment_time(trace) # 时域增强
    x_train_aug[2 * i + 1] = augment_frequency(trace) #频域增强p
    # breakpoint() #断点测试c

    y_train_aug[2 * i] = label
    y_train_aug[2 * i + 1] = label

print("Augmented train shape:", x_train_aug.shape)
print("Augmented label shape:", y_train_aug.shape)

np.savez_compressed(
    "datasets/awf1_WFTFCaug.npz",
    x_train=x_train_aug,
    y_train=y_train_aug
)

print("Saved to datasets/awf1_WFTFCaug.npz")
print(x_train_aug.dtype)
print(x_train_aug.shape)
print("WFCMemory:",
      x_train_aug.nbytes / 1024 / 1024,
      "MB")