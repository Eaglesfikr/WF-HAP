import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from DF import *

device = "cuda"

#############################
# 读取测试集
#############################
test = np.load("awf2_high.npz")

x_test = test["feature"].astype(np.float32)
y_test = test["label"].astype(np.int64)

print(x_test.shape)
print(y_test.shape)

x_test = torch.from_numpy(x_test).unsqueeze(1)
y_test = torch.from_numpy(y_test)

testset = TensorDataset(x_test, y_test)

testloader = DataLoader(
    testset,
    batch_size=128,
    shuffle=False
)

#############################
# 加载模型
#############################

model = DFNet(out_dim=100)

model.load_state_dict(torch.load(
    "checkpoints/awf1_high.pth",
    map_location=device
))

model = model.to(device)

model.eval()

#############################
# 测试
#############################

correct = 0
total = 0

with torch.no_grad():

    pbar = tqdm(testloader)

    for x, y in pbar:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(
            Acc=f"{100.0*correct/total:.2f}%"
        )

acc = 100.0 * correct / total

print(f"\nTest Accuracy = {acc:.2f}%")