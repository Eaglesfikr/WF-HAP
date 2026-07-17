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
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 设备配置与参数设置
# ==========================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
print(f"Device: {device}")

# 微调超参数
batch_size = 128
num_epoches = 100
learning_rate = 0.0005
pretrained_model_path = './checkpoints/WFTFC/WFTFC_dual_domain_epoch_100.pth.tar' # 指向你预训练好的模型

# ==========================================
# 2. 模型定义 (Backbone & 新的分类头)
# ==========================================
# --- 这里需要复制你之前训练代码中的 DFNet, ProjectionHead, DualDomainModel 定义 ---
# 为了代码完整性，这里假设它们已经定义好了。
# 请确保这里的 DualDomainModel 定义与预训练时完全一致。

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
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_feature_dim)
            out = self._forward_features(dummy_input)
            flattened_dim = out.view(1, -1).size(1)
        
        self.fc = nn.Linear(flattened_dim, out_dim)
        self.weight_init()

    def _forward_features(self, x):
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
        self.time_encoder = DFNet(out_dim=512, input_feature_dim=time_input_dim)
        self.freq_encoder = DFNet(out_dim=512, input_feature_dim=freq_input_dim)
        self.time_projector = ProjectionHead(input_dim=512, out_dim=out_dim)
        self.freq_projector = ProjectionHead(input_dim=512, out_dim=out_dim)

    def forward(self, x_time, x_freq):
        h_time = self.time_encoder(x_time)
        h_freq = self.freq_encoder(x_freq)
        z_time = self.time_projector(h_time)
        z_freq = self.freq_projector(h_freq)
        return z_time, z_freq

# ==========================================
# 3. 数据加载
# ==========================================
print("Loading Fine-tuning Datasets...")

# 加载用于微调的数据 (这里以 AWF 数据集为例)
time_data = np.load('./datasets/awf2.npz') # 时域数据
freq_data = np.load('./datasets/awf2_freq.npz') # 频域数据

x_time_total = time_data['data']
y_total = time_data['labels']
x_freq_total = freq_data['x_freq']

# 划分训练集和测试集
x_time_train, x_time_test, x_freq_train, x_freq_test, y_train, y_test = train_test_split(
    x_time_total, x_freq_total, y_total, test_size=0.2, random_state=42, stratify=y_total)

num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

class FineTuneDataset(Dataset):
    def __init__(self, x_time, x_freq, y):
        self.x_time = x_time
        self.x_freq = x_freq
        self.y = y

    def __getitem__(self, index):
        return self.x_time[index], self.x_freq[index], self.y[index]

    def __len__(self):
        return len(self.x_time)

# ==========================================
# ★★★ 新增：少样本采样函数 ★★★
# ==========================================
def sample_few_shot(x_time, x_freq, y, n_samples_per_class):
    """
    从每个类别中随机抽取 n_samples_per_class 个样本。
    """
    selected_indices = []
    # 获取所有类别的标签
    classes = np.unique(y)
    
    for c in classes:
        # 找到属于当前类别 c 的所有样本的索引
        class_indices = np.where(y == c)[0]
        
        # 随机选择 n_samples_per_class 个索引
        # 如果该类别的样本数少于 n_samples_per_class，则全部选中
        n = min(n_samples_per_class, len(class_indices))
        selected_class_indices = np.random.choice(class_indices, n, replace=False)
        
        selected_indices.extend(selected_class_indices)
    
    # 转换为 numpy 数组并打乱顺序
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    
    # 返回采样后的数据
    return x_time[selected_indices], x_freq[selected_indices], y[selected_indices]

# ==========================================
# ★★★ 新增：执行采样 ★★★
# ==========================================
N = 5  # 每个类别取 5 条样本
print(f"Performing few-shot sampling: N={N} samples per class.")

# 在划分出训练集后，立即进行采样
x_time_train, x_freq_train, y_train = sample_few_shot(x_time_train, x_freq_train, y_train, N)

print(f"Training data shape after sampling: {x_time_train.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

train_dataset = FineTuneDataset(x_time_train, x_freq_train, y_train)
test_dataset = FineTuneDataset(x_time_test, x_freq_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 4. 加载预训练模型并替换分类头
# ==========================================
time_input_dim = x_time_train.shape[1]
freq_input_dim = x_freq_train.shape[1]

# 1. 实例化模型
model = DualDomainModel(time_input_dim, freq_input_dim, out_dim=128).to(device)

# 2. 加载预训练权重
checkpoint = torch.load(pretrained_model_path)
model.load_state_dict(checkpoint)
print(f"Loaded pretrained model from {pretrained_model_path}")

# 3. 替换投影头为分类头
# 我们将两个分支的128维输出拼接起来，得到256维的向量，然后进行分类
model.time_projector = nn.Identity() # 移除时域投影头
model.freq_projector = nn.Identity() # 移除频域投影头

# --- 修改这一行 ---
# 将输入维度从 128 + 128 改为 512 + 512
model.classifier = nn.Linear(512 + 512, num_classes).to(device) 

print("Replaced projection heads with a new classifier.")

# ==========================================
# 5. 微调训练与测试
# ==========================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)
scaler = GradScaler()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for time_data, freq_data, labels in tqdm.tqdm(loader, desc="Training"):
        time_data = time_data.view(time_data.size(0), 1, time_data.size(1)).float().to(device)
        freq_data = freq_data.view(freq_data.size(0), 1, freq_data.size(1)).float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            # 前向传播
            z_time, z_freq = model(time_data, freq_data)
            # 拼接两个域的特征
            features = torch.cat([z_time, z_freq], dim=1)
            # 分类
            outputs = model.classifier(features)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for time_data, freq_data, labels in loader:
            time_data = time_data.view(time_data.size(0), 1, time_data.size(1)).float().to(device)
            freq_data = freq_data.view(freq_data.size(0), 1, freq_data.size(1)).float().to(device)
            labels = labels.to(device)

            z_time, z_freq = model(time_data, freq_data)
            features = torch.cat([z_time, z_freq], dim=1)
            outputs = model.classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

print("Start Fine-tuning...")
for epoch in range(num_epoches):
    loss = train_one_epoch(model, train_loader, optimizer, criterion)
    acc = test(model, test_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epoches}], Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

# 保存微调后的模型
torch.save(model.state_dict(), './checkpoints/WFTFC/WFTFC_dual_domain_finetuned.pth.tar')
print("Fine-tuning completed and model saved.")