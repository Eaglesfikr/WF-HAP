from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore')

# 基础库
import os
import math
import bisect
import pickle
import random
import argparse

# 第三方库
import numpy as np
import dill
import tqdm
from tqdm import tqdm

# PyTorch 核心
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# PyTorch 数据与优化器
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# 自定义模块
from DF import *

# ====================== 自定义数据集 ======================
class WFHAPAugDataset(Dataset):
    def __init__(self, data_npz_path):
        data = np.load(data_npz_path)
        self.x_all = data["x_train"].astype(np.float32)
        self.total_origin_num = self.x_all.shape[0] // 4  # 每4行对应1个原始样本

    def __len__(self):
        return self.total_origin_num

    def __getitem__(self, idx):
        # 取出一组4条样本
        ori_t = self.x_all[4 * idx]      # z_i^T 原时序
        aug_t = self.x_all[4 * idx + 1]  # \bar{z}_i^T 时序增强
        ori_f = self.x_all[4 * idx + 2]  # z_i^F 原始频谱
        aug_f = self.x_all[4 * idx + 3]  # \bar{z}_i^F 频谱增强
        return ori_t, aug_t, ori_f, aug_f

# ====================== 工具函数 accuracy (参考代码配套) ======================
def accuracy(output, target, topk=(1,)):
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

# ====================== 训练器类（对齐参考NetCLR结构） ======================
class MultiModalDFCLR(object):
    def __init__(self, **args):
        # 两路模型
        self.model_t = args['model_t']
        self.model_f = args['model_f']
        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']
        self.fp16_precision = args['fp16_precision']
        self.num_epoches = args['num_epoches']
        self.batch_size = args['batch_size']
        self.device = args['device']
        self.temperature = args['temperature']
        self.log_every_n_step = 100
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _single_modal_info_nce(self, feat_q, feat_k):
        """单模态双视图InfoNCE：输入query、aug key，返回loss, top1"""
        B = feat_q.shape[0]
        feats = torch.cat([feat_q, feat_k], dim=0)
        feats = F.normalize(feats, dim=1)
        sim_mat = torch.matmul(feats, feats.T) / self.temperature

        labels = torch.cat([torch.arange(B) for _ in range(2)], dim=0).to(self.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_mat = sim_mat[~mask].view(sim_mat.shape[0], -1)

        pos = sim_mat[labels.bool()].view(labels.shape[0], -1)
        neg = sim_mat[~labels.bool()].view(sim_mat.shape[0], -1)
        logits = torch.cat([pos, neg], dim=1)
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        loss = self.criterion(logits, target)
        top1, _ = accuracy(logits, target, topk=(1,5))
        return loss, top1.item()

    def _cross_modal_info_nce(self, feat_t, feat_f):
        """跨模态T-F InfoNCE：query=T, positive=F，负例全部batch F"""
        B = feat_t.shape[0]
        t_norm = F.normalize(feat_t, dim=1)
        f_norm = F.normalize(feat_f, dim=1)
        sim = torch.matmul(t_norm, f_norm.T) / self.temperature

        pos = sim[torch.arange(B), torch.arange(B)].unsqueeze(1)
        mask = torch.eye(B, dtype=torch.bool, device=self.device)
        neg = sim[~mask].view(B, -1)
        logits = torch.cat([pos, neg], dim=1)
        target = torch.zeros(B, dtype=torch.long, device=self.device)
        loss = self.criterion(logits, target)
        top1, _ = accuracy(logits, target, topk=(1,5))
        return loss, top1.item()

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.fp16_precision)
        n_iter = 0
        print(f"Start Multi-Modal DF SimCLR training for {self.num_epoches} epochs")

        for epoch_counter in range(1, self.num_epoches + 1):
            self.model_t.train()
            self.model_f.train()
            with tqdm(train_loader, unit="batch") as tepoch:
                for ori_t, aug_t, ori_f, aug_f in tepoch:
                    tepoch.set_description(f"Epoch {epoch_counter}")
                    B = ori_t.shape[0]
                    # 扩充通道 [B,1,L]
                    ori_t = ori_t.unsqueeze(1).to(self.device)
                    aug_t = aug_t.unsqueeze(1).to(self.device)
                    ori_f = ori_f.unsqueeze(1).to(self.device)
                    aug_f = aug_f.unsqueeze(1).to(self.device)

                    with autocast(enabled=self.fp16_precision):
                        # 时域前向
                        z_t = self.model_t(ori_t)
                        z_t_aug = self.model_t(aug_t)
                        # 频域前向
                        z_f = self.model_f(ori_f)
                        z_f_aug = self.model_f(aug_f)

                        # 三类损失
                        loss_T, acc_T = self._single_modal_info_nce(z_t, z_t_aug)
                        loss_F, acc_F = self._single_modal_info_nce(z_f, z_f_aug)
                        loss_C, acc_C = self._cross_modal_info_nce(z_t, z_f)
                        total_loss = 0.8*(loss_T + loss_F) + 0.2*loss_C

                    self.optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    # 日志打印
                    if n_iter % self.log_every_n_step == 0:
                        tepoch.set_postfix(
                            total_loss=f"{total_loss.item():.4f}",
                            lossT=f"{loss_T.item():.3f}", accT=f"{acc_T:.1f}",
                            lossF=f"{loss_F.item():.3f}", accF=f"{acc_F:.1f}",
                            lossC=f"{loss_C.item():.3f}", accC=f"{acc_C:.1f}"
                        )
                    n_iter += 1
            # 学习率调度
            if epoch_counter >= 10:
                self.scheduler.step()
            # 保存权重
            if epoch_counter % 50 == 0:
                torch.save({
                    "epoch": epoch_counter,
                    "model_t": self.model_t.state_dict(),
                    "model_f": self.model_f.state_dict(),
                    "opt": self.optimizer.state_dict(),
                    "sched": self.scheduler.state_dict()
                }, f"./checkpoints/DF_MultiSimCLR_epoch_{epoch_counter}.pth.tar")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "./datasets/protues_train_WFHAPaug.npz"
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 5e-4
    TAU = 0.5
    PROJ_DIM = 512
    FP16 = True
    # 构建两路独立DFSimCLR
    backbone_t = DFNet(out_dim=10240)
    model_t = DFsimCLR(backbone_t, out_dim=PROJ_DIM).to(device)

    backbone_f = DFNet(out_dim=10240)
    model_f = DFsimCLR(backbone_f, out_dim=PROJ_DIM).to(device)

    # 优化器 & 调度器
    params = list(model_t.parameters()) + list(model_f.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-10)

    # Dataset & Loader
    train_ds = WFHAPAugDataset(DATA_PATH)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )

    # 初始化训练器（对齐参考NetCLR传参方式）
    train_args = {
        "model_t": model_t,
        "model_f": model_f,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "fp16_precision": FP16,
        "num_epoches": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": device,
        "temperature": TAU
    }
    trainer = MultiModalDFCLR(**train_args)
    trainer.train(train_loader)