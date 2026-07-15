import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ====================== 超参数（自行修改） ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 20          # 替换为你的网站类别总数
BATCH_SIZE = 32
LABELED_PER_CLASS = 5     # 每类少量标注样本数量 N
REPEAT_EXP = 1            # 重复5轮实验消除采样随机性
TRAIN_EPOCH = 100
LR = 5e-4
PROJ_DIM = 512            # SimCLR训练时投影输出维度，保持一致
CKPT_PATH = "./checkpoints/DF_MultiSimCLR_epoch_100.pth.tar"
TARGET_NPZ = "./datasets/protues_day90_WFHAPaug.npz"  # 仅目标域day90数据集

# ====================== DFNet主干（与预训练完全一致） ======================
class DFNet(nn.Module):
    def __init__(self, out_dim):
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

        self.fc = nn.Linear(10240, out_dim)

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inp):
        x = inp
        # Block 1
        x = F.pad(x, (3, 4))
        x = F.elu(self.conv1(x))
        x = F.pad(x, (3, 4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)
        # Block 2
        x = F.pad(x, (3, 4))
        x = F.relu(self.conv2(x))
        x = F.pad(x, (3, 4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_2(x)
        x = self.dropout2(x)
        # Block 3
        x = F.pad(x, (3, 4))
        x = F.relu(self.conv3(x))
        x = F.pad(x, (3, 4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_3(x)
        x = self.dropout3(x)
        # Block 4
        x = F.pad(x, (3, 4))
        x = F.relu(self.conv4(x))
        x = F.pad(x, (3, 4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)
        return x

# ====================== SimCLR预训练包装器（仅用于加载权重） ======================
class DFsimCLR(nn.Module):
    def __init__(self, df_backbone, out_dim):
        super().__init__()
        self.backbone = df_backbone
        self.backbone.weight_init()
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )
    def forward(self, x):
        return self.backbone(x)

# ====================== 论文规定：两层FC分类头微调包装模型 ======================
class FinetuneModel(nn.Module):
    def __init__(self, backbone, feat_dim, num_cls):
        super().__init__()
        self.backbone = backbone
        # 论文描述：两个全连接层作为分类头
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_cls)
        )
    def forward(self, x):
        feat = self.backbone(x)
        logits = self.cls_head(feat)
        return logits

# ====================== 工具函数1：每类无放回采样N条标注样本 ======================
def sample_traces(all_index, y_label, N_per_cls):
    train_idx = []
    for cls in range(NUM_CLASSES):
        cls_all_idx = np.where(y_label == cls)[0]
        select_num = min(N_per_cls, len(cls_all_idx))
        selected = np.random.choice(cls_all_idx, select_num, replace=False)
        train_idx.extend(selected)
    train_idx = np.array(train_idx)
    np.random.shuffle(train_idx)
    return train_idx

# ====================== 工具函数2：数据集（同时返回时序T、频谱F、标签） ======================
class DualViewDataset(Dataset):
    def __init__(self, full_T, full_F, full_y, select_index):
        self.T_data = full_T[select_index]
        self.F_data = full_F[select_index]
        self.label = full_y[select_index]
    def __getitem__(self, idx):
        return self.T_data[idx], self.F_data[idx], self.label[idx]
    def __len__(self):
        return len(self.label)

# ====================== 工具函数3：加载目标域全部原始T/F（丢弃增强样本） ======================
def load_target_dataset(npz_path):
    data = np.load(npz_path)
    all_x = data["x_train"]
    all_y = data["y_train"]
    group_count = all_x.shape[0] // 4
    T_list = []
    F_list = []
    y_list = []
    for i in range(group_count):
        T_list.append(all_x[4 * i])       # 原始时序
        F_list.append(all_x[4 * i + 2])   # 原始频谱
        y_list.append(all_y[4 * i])
    T_arr = np.array(T_list, dtype=np.float32)
    F_arr = np.array(F_list, dtype=np.float32)
    y_arr = np.array(y_list)
    return T_arr, F_arr, y_arr

# ====================== 工具函数4：加载预训练权重，丢弃投影头，拼接两层分类头 ======================
def get_finetune_model(ckpt_dict_key):
    # 重建预训练SimCLR结构
    raw_backbone = DFNet(out_dim=10240)
    simclr_model = DFsimCLR(raw_backbone, out_dim=PROJ_DIM)
    # 加载checkpoint
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    state_dict = checkpoint[ckpt_dict_key]

    # 只保留卷积主干参数，抛弃整套SimCLR投影头 backbone.fc
    backbone_state = {}
    for key in state_dict.keys():
        if key.startswith("backbone.") and not key.startswith("backbone.fc"):
            new_key = key[len("backbone."):]
            backbone_state[new_key] = state_dict[key]
    # 空主干加载预训练卷积权重
    pure_backbone = DFNet(out_dim=10240)
    missing_keys, _ = pure_backbone.load_state_dict(backbone_state, strict=False)
    assert set(missing_keys) == {"fc.weight", "fc.bias"}, "仅缺失投影头fc为正常现象"
    # 拼接论文规定两层分类头，构建完整微调网络
    finetune_net = FinetuneModel(pure_backbone, feat_dim=10240, num_cls=NUM_CLASSES).to(device)
    return finetune_net

# ====================== 训练、测试单分支函数 ======================
def train_branch(model, dataloader, optimizer):
    model.train()
    for batch_idx, (T_input, F_input, label) in enumerate(dataloader):
        optimizer.zero_grad()
        # 区分时域/频域输入
        if hasattr(model, "branch_type") and model.branch_type == "time":
            in_tensor = T_input.unsqueeze(1).float().to(device)
        else:
            in_tensor = F_input.unsqueeze(1).float().to(device)
        label_tensor = label.to(device).long()
        logits = model(in_tensor)
        loss = F.cross_entropy(logits, label_tensor)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Batch Loss: {loss.item():.6f}")

def test_branch(model, dataloader):
    model.eval()
    total_correct = 0
    total_sample = len(dataloader.dataset)
    with torch.no_grad():
        for T_input, F_input, label in dataloader:
            if hasattr(model, "branch_type") and model.branch_type == "time":
                in_tensor = T_input.unsqueeze(1).float().to(device)
            else:
                in_tensor = F_input.unsqueeze(1).float().to(device)
            label_tensor = label.to(device).long()
            logits = model(in_tensor)
            pred = logits.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(label_tensor.view_as(pred)).sum().item()
    acc = total_correct / total_sample
    return acc

# ====================== 主实验入口 ======================
if __name__ == "__main__":
    # 一次性加载目标域全部原始时序、频谱、标签
    all_T, all_F, all_y = load_target_dataset(TARGET_NPZ)
    all_sample_index = np.arange(len(all_y))

    time_best_acc_record = []
    freq_best_acc_record = []

    for exp_round in range(REPEAT_EXP):
        print(f"\n==================== Experiment {exp_round + 1}/{REPEAT_EXP} ====================")
        # 1. 划分少量标注训练集 + 剩余全量测试集
        train_index = sample_traces(all_sample_index, all_y, LABELED_PER_CLASS)
        test_index = np.setdiff1d(all_sample_index, train_index)
        # 构建数据集与加载器（T/F同时存在，两套分支共用）
        train_dataset = DualViewDataset(all_T, all_F, all_y, train_index)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
        test_dataset = DualViewDataset(all_T, all_F, all_y, test_index)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=0)

        # 2. 初始化时域、频域微调网络（预训练主干+两层分类头）
        model_time = get_finetune_model("model_t")
        model_time.branch_type = "time"
        model_freq = get_finetune_model("model_f")
        model_freq.branch_type = "freq"
        # 全部参数联合微调，无冻结
        opt_time = optim.Adam(model_time.parameters(), lr=LR)
        opt_freq = optim.Adam(model_freq.parameters(), lr=LR)

        best_time_acc = 0.0
        best_freq_acc = 0.0
        # 3. 完整epoch训练循环
        for epoch in range(TRAIN_EPOCH):
            print(f"Epoch {epoch}")
            # 同时训练两个分支，共用一份dataloader
            train_branch(model_time, train_loader, opt_time)
            train_branch(model_freq, train_loader, opt_freq)
            # 测试当前精度
            cur_time_acc = test_branch(model_time, test_loader)
            cur_freq_acc = test_branch(model_freq, test_loader)
            # 更新全局最优精度
            best_time_acc = max(best_time_acc, cur_time_acc)
            best_freq_acc = max(best_freq_acc, cur_freq_acc)
            # 每10轮打印一次测试精度
            if epoch % 10 == 0:
                print(f"Time Acc: {cur_time_acc * 100:.2f}% | Freq Acc: {cur_freq_acc * 100:.2f}%")
        # 保存本轮实验最优精度
        time_best_acc_record.append(best_time_acc)
        freq_best_acc_record.append(best_freq_acc)
        print(f"Round{exp_round+1} Best Result -> Time:{best_time_acc*100:.2f}% Freq:{best_freq_acc*100:.2f}%")
        print("-" * 70)

    # 4. 统计5次实验均值、标准差
    time_acc_arr = np.array(time_best_acc_record)
    freq_acc_arr = np.array(freq_best_acc_record)
    print("\n==================== Final Average Result ====================")
    print(f"Time Branch | Mean Acc: {np.mean(time_acc_arr)*100:.1f} %, Std: {np.std(time_acc_arr)*100:.1f}")
    print(f"Freq Branch | Mean Acc: {np.mean(freq_acc_arr)*100:.1f} %, Std: {np.std(freq_acc_arr)*100:.1f}")