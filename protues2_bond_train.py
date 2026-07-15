import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from DF import *
from tqdm import tqdm
import os
train = np.load("./datasets/proteus_train_sign.npz")
device = "cuda"

x_train = train["X"].astype(np.float32)
y_train = train["y"].astype(np.int64)

print(x_train.shape)
print(y_train.shape)

x_train = torch.from_numpy(x_train).unsqueeze(1).to(device)
y_train = torch.from_numpy(y_train).to(device)

trainset = TensorDataset(x_train, y_train)

trainloader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True
)



model = DFNet(out_dim=102)

model.weight_init()

model = model.to(device)


## 优化器
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

## 训练
epochs = 50

epochs = 50

for epoch in range(epochs):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        trainloader,
        desc=f"Epoch [{epoch+1}/{epochs}]",
        leave=True
    )

    for x, y in pbar:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({
            "Loss": f"{total_loss/(pbar.n+1):.4f}",
            "Acc": f"{100.0*correct/total:.2f}%"
        })

    print(
        f"Epoch {epoch+1:03d}: "
        f"Loss={total_loss/len(trainloader):.4f}, "
        f"Acc={100.0*correct/total:.2f}%"
    )

os.makedirs("checkpoints", exist_ok=True)
torch.save(
        model.state_dict(),
        "checkpoints/protues_all.pth"
    )