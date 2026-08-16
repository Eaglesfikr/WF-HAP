import numpy as np
import tqdm
import numpy as np
from scipy.ndimage import uniform_filter1d  # 用于平滑


class FreqAugTool:
    def freq_augment_spectrum(
            self,
            spec: np.ndarray,
            low_cutoff: int = 800,
            rho: float = 1.0,      # 建议降低默认值
            alpha: float = 3.0,
            beta: float = 2.0,
            gamma: float = 1.0,    # 建议降低默认值，线性化响应
            eps: float = 1e-6,
            seed: int = None,
            smooth_window: int = 50, # 新增：平滑窗口大小
            max_gain: float = 1.5,   # 新增：最大增益限制
            min_gain: float = 0.5    # 新增：最小增益限制
    ):
        if seed is not None:
            np.random.seed(seed)

        G = spec.copy()
        N = len(G)

        
        
        # 1. 全局统计量
        global_mu = np.mean(G)
        global_sigma = np.std(G)

        # 2. 掩码逻辑 (保持原样)
        M_suppress = np.ones_like(G)
        M_suppress[low_cutoff:] = 0.0
        G_M = (1 - M_suppress) * G

        # 3. 映射函数
        numerator = np.square(G_M - alpha * global_mu)
        denominator = 2 * np.square(beta * global_sigma)
        denominator = np.maximum(denominator, eps)

        exp_term = np.exp(- numerator / denominator)
        G_H = np.power(rho * exp_term, gamma) + eps

        # --- 关键修改点 1: 平滑 noise_scale ---
        # 在应用随机噪声前，先让标准差曲线变平滑，避免相邻频率点扰动差异过大
        # 仅对中高频部分进行平滑
        noise_scale = G_H.copy()
        if smooth_window > 1:
            # 对整体做平滑，或者只对 high_freq 部分做
            # 这里简单处理：对整个 noise_scale 做滑动平均
            noise_scale = uniform_filter1d(noise_scale, size=smooth_window)

        # 强制低频部分无扰动
        noise_scale[:low_cutoff] = 0.0

        # 4. 生成增益
        gain = np.random.normal(loc=1.0, scale=noise_scale, size=N)

        # --- 关键修改点 2: 限制增益范围 (Clipping) ---
        # 防止出现 5倍、10倍 这种破坏性的极端值
        gain = np.clip(gain, a_min=min_gain, a_max=max_gain)

        aug_spec = G * gain

        return aug_spec


if __name__ == "__main__":
    aug_tool = FreqAugTool()
    # 读取你已生成好的原始频谱文件
    data = np.load("./datasets/awf1_freq.npz")
    x_ori_freq = data["x"]   # shape (N, 5000)
    y_label = data["y"]
    print(f"原始频谱数据集 shape: {x_ori_freq.shape}")

    num_sample = x_ori_freq.shape[0]
    aug_freq_data = np.empty_like(x_ori_freq, dtype=np.float32)

    # 批量增强所有频谱样本
    print("开始频域增强处理...")
    for idx in tqdm.tqdm(range(num_sample)):
        single_spec = x_ori_freq[idx]
        aug_spec = aug_tool.freq_augment_spectrum(single_spec)
        aug_freq_data[idx] = aug_spec

    # 保存增强后的频谱文件
    save_path = "./datasets/awf1_freq_hapaug_v2.npz"
    np.savez_compressed(save_path, x=aug_freq_data, y=y_label)
    print(f"增强频谱已保存至: {save_path}")
    print(f"输出增强频谱 shape: {aug_freq_data.shape}")