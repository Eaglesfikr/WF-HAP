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

data = np.load('./datasets/awf1.npz') # AWF-PT-sup
x_train = data['feature']
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

class Augmentor():
    def __init__(self):
        methods = {
            'merge downstream burst',
            'change downstream burst sizes',
            'merge downstream and upstream bursts',
            'add upstream bursts',
            'remove upstrean bursts',
            'divide bursts'
        }
        
        
        self.large_burst_threshold = 10
        
        # changing the content
        self.upsample_rate = 1.0
        self.downsample_rate = 0.5
        
        # merging bursts
        self.num_bursts_to_merge = 5
        self.merge_burst_rate = 0.1
        
        # add incoming bursts
        self.add_outgoing_burst_rate = 0.3
        # self.outgoing_burst_sizes = list(range(max_outgoing_burst_size))
        self.outgoing_burst_sizes = list(range(max(1, int(np.floor(max_outgoing_burst_size)))))
        
        # shift
        self.shift_param = 10
        
        
        
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
        
        
    # representing the change of contents of a website
    def increase_incoming_bursts(self, burst_sizes):
        out = []
        for i, size in enumerate(burst_sizes):
            if size <= -self.large_burst_threshold:
                up_sample_rate = random.random()*self.upsample_rate
                new_size = int(size * (1+up_sample_rate))
                out.append(new_size)
            else:
                out.append(size)
                
        return out
        
        
    def decrease_incoming_bursts(self, burst_sizes):
        out = []
        for i, size in enumerate(burst_sizes):
            if size <= -self.large_burst_threshold:
                up_sample_rate = random.random()*self.downsample_rate
                new_size = int(size * (1-up_sample_rate))
                out.append(new_size)
            else:
                out.append(size)
                
        return out
        
        
    def change_content(self, trace):
        bursts = self.find_bursts(trace)
        burst_sizes = [x[2] for x in bursts]
        
        if len(trace) < 1000:
            new_burst_sizes = self.increase_incoming_bursts(burst_sizes)
            
        elif len(trace) > 4000:
            new_burst_sizes = self.decrease_incoming_bursts(burst_sizes)
            
        else:
            p = random.random()
            if p >= 0.5:
                new_burst_sizes = self.increase_incoming_bursts(burst_sizes)
                
            else:
                new_burst_sizes = self.decrease_incoming_bursts(burst_sizes)
                
                
        return new_burst_sizes
    
    
    def merge_incoming_bursts(self, burst_sizes):
        
        out = []
        
        # skipping first 20 cells
        i = 0
        num_cells = 0
        while i < len(burst_sizes) and num_cells < 20:
            num_cells += abs(burst_sizes[i])
            out.append(burst_sizes[i])
            i += 1
            
        
        while i < len(burst_sizes) - self.num_bursts_to_merge:
            prob = random.random()
            
            # ignore outgoing bursts
            if burst_sizes[i] > 0:
                out.append(burst_sizes[i])
                i+= 1
                continue
            
            if prob < self.merge_burst_rate:
                num_merges = random.randint(2, self.num_bursts_to_merge)
                merged_size = 0
                
                # merging the incoming bursts
                while i < len(burst_sizes) and num_merges > 0:
                    if burst_sizes[i] < 0:
                        merged_size += burst_sizes[i]
                        num_merges -= 1
                    i += 1     
                out.append(merged_size)
                    
            else:
                out.append(burst_sizes[i])
                i += 1
                
        return out
    
    
    def add_outgoing_burst(self, burst_sizes):
        
        out = []
        
        i = 0
        num_cells = 0
        while i < len(burst_sizes) and num_cells < 20:
            num_cells += abs(burst_sizes[i])
            out.append(burst_sizes[i])
            i += 1
            
        
        for size in burst_sizes[i:]:
            if size > -10 :
                out.append(size)
                continue
            
            prob = random.random()
            
            if prob < self.add_outgoing_burst_rate:
                
                index = len(outgoing_burst_sizes)
                while index >= len(outgoing_burst_sizes):
                    outgoing_burst_prob = random.random()
                    index = bisect.bisect_left(OUTGOING_BURST_SIZE_CDF, outgoing_burst_prob)
                    
                outgoing_burst_size = self.outgoing_burst_sizes[index]
                divide_place = random.randint(3, abs(size) - 3)
                
                out += [-divide_place, outgoing_burst_size, -(abs(size) - divide_place)]
                
            else:
                out.append(size)
                
        return out
                
        
    def create_trace_from_burst_sizes(self, burst_sizes):
        out = []
        
        for size in burst_sizes:
            val = 1 if size > 0 else -1
            
            out += [val]*(int(abs(size)))
            
        if len(out) < 5000:
            out += [0]*(5000 - len(out))
            
        return np.array(out)[:5000]
    
    def shift(self, x):
        pad = np.random.randint(0, 2, size = (self.shift_param, ))
        pad = 2*pad-1
        zpad = np.zeros_like(pad)
        
        shift_val = np.random.randint(-self.shift_param, self.shift_param+1, 1)[0]
        shifted = np.concatenate((x, zpad, pad), axis=-1)
        shifted = np.roll(shifted, shift_val, axis=-1)
        shifted = shifted[:5000]
        
        return shifted
        
    
    def augment(self, trace):
        
        mapping = {
            0: self.change_content,
            1: self.merge_incoming_bursts,
            2: self.add_outgoing_burst
        }
        
        bursts = self.find_bursts(trace)
        
        burst_sizes = [x[2] for x in bursts]
        
        
        aug_method = mapping[random.randint(0, len(mapping)-1)]
        
        augmented_sizes = aug_method(burst_sizes)
        
        augmented_trace = self.create_trace_from_burst_sizes(augmented_sizes)
        
        return self.shift(augmented_trace)



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

augmentor = Augmentor()

# train_dataset = TrainData(x_train, y_train, augmentor, 2)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print("Generating 2x augmented dataset...")

x_train_aug = []
y_train_aug = []

for i in tqdm.tqdm(range(len(x_train))):
    print(f"Processing {i}/{len(x_train)-1}")
    trace = x_train[i]
    label = y_train[i]

    # 第一次增强
    aug1 = augmentor.augment(trace)
    
    # 第二次增强（随机增强，与第一次不同）
    aug2 = augmentor.augment(trace)

    x_train_aug.append(aug1)
    y_train_aug.append(label)

    x_train_aug.append(aug2)
    y_train_aug.append(label)

x_train_aug = np.asarray(x_train_aug, dtype=x_train.dtype)
y_train_aug = np.asarray(y_train_aug, dtype=y_train.dtype)

print("Augmented train shape:", x_train_aug.shape)
print("Augmented label shape:", y_train_aug.shape)
np.savez(
    "datasets/awf1_aug2x.npz",
    x_train=x_train_aug,
    y_train=y_train_aug
)

print("Saved to datasets/AWF/awf1_aug2x.npz")