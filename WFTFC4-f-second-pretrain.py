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
start_epoch = 100
end_epoch = 400

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
class DFNet(nn.Module):
    def __init__(self, out_dim):
        super(DFNet, self).__init__()
        kernel_size = 8
        channels = [1, 32, 64, 128, 256]
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
        
        self.fc = nn.Linear(2560, out_dim)

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()
                
    def forward(self, inp):
        x = inp
        # ==== first block ====
        x = F.pad(x, (3,4))
        x = F.elu((self.conv1(x)))
        x = F.pad(x, (3,4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)
        
        # ==== second block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv2(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_2(x)
        x = self.dropout2(x)
        
        # ==== third block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv3(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_3(x)
        x = self.dropout3(x)
        
        # ==== fourth block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv4(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x 

class DFsimCLR(nn.Module):
    def __init__(self, df, out_dim):
        super(DFsimCLR, self).__init__()
        
        self.backbone = df
        self.backbone.weight_init()
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )
        
    def forward(self, inp):
        out = self.backbone(inp)
        return out

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
        
    def train(
            self,
            train_loader,
            start_epoch,
            end_epoch
    ):
        best_acc = 0
        scaler = GradScaler(enabled=self.fp16_precision)

        n_iter = 0
        print ("Start SimCLR training for %d number of epoches"%self.num_epoches)
        
        for epoch_counter in range(start_epoch, end_epoch):
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
                torch.save(self.model.state_dict(), f'./checkpoints/WFTFC/WFTFC_freq_epoch_{epoch_counter}.pth.tar')

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
    df = DFNet(out_dim=512)
    model = DFsimCLR(df, out_dim=128).to(device)
    checkpoint = torch.load(
        "./checkpoints/WFTFC/WFTFC_freq_epoch_100.pth.tar",
        map_location=device
    )

    model.load_state_dict(checkpoint)

    print("Loaded checkpoint: epoch100")
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
    netclr.train(
        train_loader,
        start_epoch,
        end_epoch
    )