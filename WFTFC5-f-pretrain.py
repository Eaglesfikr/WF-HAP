from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import tqdm
import os

# ==========================================
# 1. 设备配置与参数设置
# ==========================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
print(f"Device: {device}")

# 超参数
batch_size = 256
fp16_precision = True
temperature = 0.5
num_epoches = 100

# ==========================================
# 2. 数据加载 (修改部分)
# ==========================================
# 加载你生成的两个频域文件
print("Loading frequency domain datasets...")
freq_data_orig = np.load('./datasets/awf1_freq.npz')
freq_data_aug = np.load('./datasets/awf1_freq_tfcaug.npz')

x_train_orig = freq_data_orig['x'] # 原始频域特征
x_train_aug = freq_data_aug['x']   # 增强后的频域特征
y_train = freq_data_orig['y']      # 标签 (假设两个文件的标签顺序一致)

print(f"Original freq data shape: {x_train_orig.shape}")
print(f"Augmented freq data shape: {x_train_aug.shape}")

num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

# ==========================================
# 3. 模型定义 (Backbone & Projection Head)
# ==========================================
class FCN(nn.Module):
    def __init__(self,
                 n_channels=1,
                 out_channels=128):
        super().__init__()

        kernel_size = 8

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=kernel_size//2,
                      bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,2,padding=1),
            nn.Dropout(0.35)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32,64,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=kernel_size//2,
                      bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,2,padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64,out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=kernel_size//2,
                      bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2,2,padding=1)
        )

        # 不固定长度
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.feature_dim = out_channels

    def forward(self,x):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        return x

class FCNSimCLR(nn.Module):

    def __init__(self, backbone, out_dim=128):
        super().__init__()

        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(backbone.feature_dim, backbone.feature_dim),
            nn.BatchNorm1d(backbone.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone.feature_dim, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)      # (B,256)
        z = self.projector(h)     # (B,128)
        return z

# ==========================================
# 4. 数据加载器 (修改部分)
# ==========================================
class FreqTrainData(Dataset):
    def __init__(self, x_orig, x_aug, y):
        self.x_orig = x_orig
        self.x_aug = x_aug
        self.y = y

    def __getitem__(self, index):
        # 返回一对数据：原始频谱 和 增强后的频谱
        view1 = self.x_orig[index]
        view2 = self.x_aug[index]
        label = self.y[index]
        return [view1, view2], label

    def __len__(self):
        return len(self.x_orig)

# ==========================================
# 5. NetCLR 训练逻辑
# ==========================================
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class NetCLR(object):
    def __init__(self, **args):
        self.model = args['model']
        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']
        self.fp16_precision = args['fp16_precision']
        self.num_epoches = args['num_epoches']
        self.batch_size = args['batch_size']
        self.device = args['device']
        self.temperature = args['temperature']
        self.n_views = 2
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.log_every_n_step = 100
        
    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        
        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        logits = logits / self.temperature
        return logits, labels
        
    def train(self, train_loader):
        best_acc = 0
        scaler = GradScaler(enabled=self.fp16_precision)

        n_iter = 0
        print ("Start SimCLR training for %d number of epoches"%self.num_epoches)
        
        for epoch_counter in range(self.num_epoches+1):
            with tqdm.tqdm(train_loader, unit='batch') as tepoch:
                for data, _ in tepoch:
                    tepoch.set_description(f"Epoch {epoch_counter}")
                    
                    self.model.train()
                    # data 是一个包含两个张量的列表 [view1, view2]
                    # 将它们拼接在一起
                    data = torch.cat(data, dim=0)
                    
                    # --- 修复开始 ---
                    # 关键修复：将数据从 (Batch, Length) 变为 (Batch, Channels, Length)
                    data = data.view(data.size(0), 1, data.size(1))
                    # --- 修复结束 ---

                    data = data.float().to(self.device)

                    with autocast(enabled=self.fp16_precision):
                        features = self.model(data)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    if n_iter%self.log_every_n_step == 0:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        tepoch.set_postfix(loss=loss.item(), accuracy = top1.item())
                    n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()
            
            # 保存模型
            if epoch_counter % 20 == 0 and epoch_counter > 0:
                os.makedirs('./checkpoints/WFTFC/', exist_ok=True)
                torch.save(self.model.state_dict(), f'./checkpoints/WFTFC/WFTFC_freq_FCN_epoch_{epoch_counter}.pth.tar')

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 创建数据集和数据加载器
    train_dataset = FreqTrainData(x_train_orig, x_train_aug, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 获取频域特征的长度，例如 2501
    input_feature_dim = x_train_orig.shape[1]
    print("input feature:",input_feature_dim)

    # 初始化模型
    backbone = FCN(
        n_channels=1,
        out_channels=256
    )
    
    model = FCNSimCLR(
        backbone,
        out_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    netclr = NetCLR(
               model = model,
               optimizer = optimizer,
               scheduler = scheduler,
               fp16_precision = fp16_precision,
               device = device,
               temperature = temperature,
               n_views = 2,
               num_epoches = 101,
               batch_size = batch_size)
    netclr.train(train_loader)