import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    print(data.files)
    X = data["data"]
    y = data["labels"]

    X = np.array(X)
    y = np.array(y)

    return X, y


# ===== load =====
X_src, y_src = load_npz("./datasets/tor_100w_2500tr.npz")
X_tgt, y_tgt = load_npz("./datasets/tor_200w-100w_2500.npz")

# 标签要从str映射为整数
y_src = np.array(y_src).astype(str)
y_tgt = np.array(y_tgt).astype(str)
# ⭐关键：用 union，而不是只用 source
all_classes = set(y_src) | set(y_tgt)

cls2id = {c: i for i, c in enumerate(sorted(all_classes))}

y_src = np.array([cls2id[c] for c in y_src])
y_tgt = np.array([cls2id[c] for c in y_tgt])