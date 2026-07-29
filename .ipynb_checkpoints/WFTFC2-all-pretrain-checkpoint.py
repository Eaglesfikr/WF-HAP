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
batch_size = 128
fp16_precision = True
temperature = 0.5
num_epoches = 100
alpha = 0.8  # 时域和频域自监督损失的权重
beta = 0.2   # 时频一致性损失的权重

# ==========================================
# 2. 数据加载
# ==========================================
print("Loading Time and Frequency domain datasets...")

# --- 加载时域数据 ---
time_data_orig = np.load('./datasets/awf1.npz')
time_data_aug = np.load('./datasets/awf1_aug2x.npz')
x_time_orig = time_data_orig['feature']
x_time_aug_full = time_data_aug['X']
# 从增强数据中隔行取样
x_time_aug = x_time_aug_full[::2]

# --- 加载频域数据 ---
freq_data_orig = np.load('./datasets/awf1_freq.npz')
freq_data_aug = np.load('./datasets/awf1_freq_tfcaug.npz')
x_freq_orig = freq_data_orig['x']
x_freq_aug = freq_data_aug['x']

# --- 加载标签 ---
y_train = time_data_orig['label']

print(f"Original time data shape: {x_time_orig.shape}")
print(f"Augmented time data shape: {x_time_aug.shape}")
print(f"Original freq data shape: {x_freq_orig.shape}")
print(f"Augmented freq data shape: {x_freq_aug.shape}")

num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

# ==========================================
# 3. 模型定义 (Backbone & Projection Head)
# ==========================================
class DFNet(nn.Module):
    def __init__(self, out_dim, input_feature_dim):
        super(DFNet, self).__init__()
        kernel_size = 8
        conv_stride = 1
        pool_stride = 4
        pool_size = 8
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size, stride=conv_stride)
        self.conv1_1 = nn.Conv1d(32, 32, kernel_size, stride=conv_stride)
        self.conv2 = nn.Conv1d(32, 64, kernel_size, stride=conv_stride)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, stride=conv_stride)
        self.conv3 = nn.Conv1d(64, 128, kernel_size, stride=conv_stride)
        self.conv3_3 = nn.Conv1d(128, 128, kernel_size, stride=conv_stride)
        self.conv4 = nn.Conv1d(128, 256, kernel_size, stride=conv_stride)
        self.conv4_4 = nn.Conv1d(256, 256, kernel_size, stride=conv_stride)
        
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)
        
        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        
        # 动态计算展平后的维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_feature_dim)
            out = self._forward_features(dummy_input)
            flattened_dim = out.view(1, -1).size(1)
        
        self.fc = nn.Linear(flattened_dim, out_dim)
        self.weight_init()

    def _forward_features(self, x):
        # 提取卷积层特征，用于计算全连接层输入维度
        x = F.pad(x, (3,4))
        x = F.elu((self.conv1(x)))
        x = F.pad(x, (3,4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)

        x = F.pad(x, (3,4))
        x = F.relu((self.conv2(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_2(x)
        x = self.dropout2(x)

        x = F.pad(x, (3,4))
        x = F.relu((self.conv3(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_3(x)
        x = self.dropout3(x)

        x = F.pad(x, (3,4))
        x = F.relu((self.conv4(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_4(x)
        x = self.dropout4(x)
        return x

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inp):
        x = self._forward_features(inp)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, out_dim)
        )

    def forward(self, x):
        return self.projector(x)

class DualDomainModel(nn.Module):
    def __init__(self, time_input_dim, freq_input_dim, out_dim=128):
        super(DualDomainModel, self).__init__()
        # 时域特征提取器 (Backbone)
        self.time_encoder = DFNet(out_dim=512, input_feature_dim=time_input_dim)
        # 频域特征提取器 (Backbone)
        self.freq_encoder = DFNet(out_dim=512, input_feature_dim=freq_input_dim)
        
        # 投影头 (Projection Head)
        self.time_projector = ProjectionHead(input_dim=512, out_dim=out_dim)
        self.freq_projector = ProjectionHead(input_dim=512, out_dim=out_dim)

    def forward(self, x_time, x_freq):
        # 1. 分别通过各自的Backbone提取特征
        h_time = self.time_encoder(x_time)
        h_freq = self.freq_encoder(x_freq)
        
        # 2. 分别通过各自的投影头得到128维向量
        z_time = self.time_projector(h_time)
        z_freq = self.freq_projector(h_freq)
        
        return z_time, z_freq

# ==========================================
# 4. 数据加载器
# ==========================================
class DualDomainDataset(Dataset):
    def __init__(self, x_time_orig, x_time_aug, x_freq_orig, x_freq_aug, y):
        self.x_time_orig = x_time_orig
        self.x_time_aug = x_time_aug
        self.x_freq_orig = x_freq_orig
        self.x_freq_aug = x_freq_aug
        self.y = y

    def __getitem__(self, index):
        view_time_orig = self.x_time_orig[index]
        view_time_aug = self.x_time_aug[index]
        view_freq_orig = self.x_freq_orig[index]
        view_freq_aug = self.x_freq_aug[index]
        label = self.y[index]
        return [view_time_orig, view_time_aug, view_freq_orig, view_freq_aug], label

    def __len__(self):
        return len(self.x_time_orig)

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
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.log_every_n_step = 100

    def info_nce_loss(self, features):
        batch_size = features.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
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
        print (f"Start Dual-Domain SimCLR training for {self.num_epoches} epochs")
        
        for epoch_counter in range(self.num_epoches + 1):
            with tqdm.tqdm(train_loader, unit='batch') as tepoch:
                for data, _ in tepoch:
                    tepoch.set_description(f"Epoch {epoch_counter}")
                    self.model.train()
                    
                    # 解包数据
                    time_orig, time_aug, freq_orig, freq_aug = data
                    
                    # 拼接时域和频域的原始与增强视图
                    time_data = torch.cat([time_orig, time_aug], dim=0)
                    freq_data = torch.cat([freq_orig, freq_aug], dim=0)
                    
                    # 增加通道维度 (Batch, Length) -> (Batch, Channels=1, Length)
                    time_data = time_data.view(time_data.size(0), 1, time_data.size(1)).float().to(self.device)
                    freq_data = freq_data.view(freq_data.size(0), 1, freq_data.size(1)).float().to(self.device)

                    with autocast(enabled=self.fp16_precision):
                        # 前向传播，获取时域和频域的投影特征 (128维)
                        z_time, z_freq = self.model(time_data, freq_data)
                        
                        # 计算三个损失
                        # 1. 时域损失 (基于时域投影向量)
                        logits_time, labels_time = self.info_nce_loss(z_time)
                        loss_time = self.criterion(logits_time, labels_time)
                        
                        # 2. 频域损失 (基于频域投影向量)
                        logits_freq, labels_freq = self.info_nce_loss(z_freq)
                        loss_freq = self.criterion(logits_freq, labels_freq)
                        
                        # 3. 时频一致性损失 (基于原始时域和原始频域的投影向量)
                        z_consistency = torch.cat([z_time[:self.batch_size], z_freq[:self.batch_size]], dim=0)
                        logits_consistency, labels_consistency = self.info_nce_loss(z_consistency)
                        loss_consistency = self.criterion(logits_consistency, labels_consistency)
                        
                        # 计算总损失
                        loss = alpha * (loss_time + loss_freq) + beta * loss_consistency

                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if n_iter % self.log_every_n_step == 0:
                        top1, _ = accuracy(logits_time, labels_time, topk=(1, 5))
                        tepoch.set_postfix(loss=loss.item(), loss_time=loss_time.item(), loss_freq=loss_freq.item(), loss_cons=loss_consistency.item(), acc=top1.item())
                    
                    n_iter += 1
            
            # 学习率调度
            if epoch_counter >= 10:
                self.scheduler.step()
            
            # 保存模型
            if epoch_counter % 20 == 0 and epoch_counter > 0:
                os.makedirs('./checkpoints/WFTFC/', exist_ok=True)
                torch.save(self.model.state_dict(), f'./checkpoints/WFTFC/WFTFC_dual_domain_epoch_{epoch_counter}.pth.tar')

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 创建数据集和数据加载器
    train_dataset = DualDomainDataset(x_time_orig, x_time_aug, x_freq_orig, x_freq_aug, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    time_input_dim = x_time_orig.shape[1]
    freq_input_dim = x_freq_orig.shape[1]
    print(f"Time input feature dim: {time_input_dim}")
    print(f"Freq input feature dim: {freq_input_dim}")

    # 初始化双域模型
    model = DualDomainModel(time_input_dim, freq_input_dim, out_dim=128).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    
    netclr = NetCLR(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        fp16_precision=fp16_precision,
        device=device,
        temperature=temperature,
        num_epoches=num_epoches,
        batch_size=batch_size
    )
    netclr.train(train_loader)