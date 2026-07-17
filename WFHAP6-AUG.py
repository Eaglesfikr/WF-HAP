import numpy as np
import tqdm

class FreqAugTool:
    def freq_augment_spectrum(
            self,
            spec: np.ndarray,
            low_cutoff: int = 800,
            mid_cutoff: int = 1600,
            t: float = 0.3,
            rho: float = 2.0,
            alpha: float = 3.0,
            beta: float = 2.0,
            gamma: float = 2.0,
            eps: float = 5e-3,
            seed: int = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        G = spec.copy()
        N = len(G)
        assert N == 2500, "输入频谱必须长度=5000"

        band_split = [0, low_cutoff, mid_cutoff, N]
        mu = np.zeros(N)
        sigma = np.zeros(N)
        for i in range(3):
            start, end = band_split[i], band_split[i+1]
            band_data = G[start:end]
            band_mu = np.mean(band_data)
            band_std = np.std(band_data)
            mu[start:end] = band_mu
            sigma[start:end] = band_std

        M = np.where(mu < t, 1.0, 0.0)
        G_M = (1 - M) * G

        numerator = np.square(G_M - alpha * mu)
        denominator = 2 * np.square(beta * sigma)
        exp_term = np.exp(- numerator / (denominator + eps))
        G_H = np.power(rho * exp_term, gamma) + eps

        gain = np.random.normal(loc=1.0, scale=G_H, size=N)
        gain = np.clip(gain, a_min=eps, a_max=None)

        aug_spec = G * gain
        aug_min = np.min(aug_spec)
        aug_max = np.max(aug_spec)
        if aug_max - aug_min > eps:
            aug_spec = (aug_spec - aug_min) / (aug_max - aug_min)
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
    save_path = "./datasets/awf1_freq_hapaug.npz"
    np.savez_compressed(save_path, x=aug_freq_data, y=y_label)
    print(f"增强频谱已保存至: {save_path}")
    print(f"输出增强频谱 shape: {aug_freq_data.shape}")