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
            eps: float = 1e-6,
            seed: int = None
    ):
        if seed is not None:
        np.random.seed(seed)
    
        G = spec.copy()
        N = len(G)
        
        # 1. 计算全局统计量 (关键点1：使用全局而非局部)
        global_mu = np.mean(G)
        global_sigma = np.std(G)
        
        # 2. 构造针对性的掩码 (关键点2：显式保留中高频)
        # 假设我们要增强 [low_cutoff, N] 的部分
        mask_indices = np.arange(N)
        # 创建一个全1的掩码，把不想增强的地方(低频)置0
        # 注意：这里逻辑是 1=保留/增强, 0=不处理
        enhancement_mask = np.zeros_like(G)
        enhancement_mask[low_cutoff:] = 1.0 
        
        # 如果原论文逻辑是 G_M = (1-M)*G，那这里的 M 应该是“抑制掩码”
        # 即：低频 M=1 (被抑制), 中高频 M=0 (被保留)
        M_suppress = np.ones_like(G)
        M_suppress[low_cutoff:] = 0.0
        
        G_M = (1 - M_suppress) * G # 现在 G_M 只有中高频有值，低频为0
        
        # 3. 应用映射函数
        # 注意：对于 G_M 中为 0 的部分（低频），代入公式会得到一个常数背景值
        # exp(-(0 - alpha*mu)^2 / ...) 
        # 为了让低频完全不受影响，建议在计算 gain 时强行将低频 gain 设为 1
        
        numerator = np.square(G_M - alpha * global_mu)
        denominator = 2 * np.square(beta * global_sigma)
        
        # 防止除零
        denominator = np.maximum(denominator, eps)
        
        exp_term = np.exp(- numerator / denominator)
        G_H = np.power(rho * exp_term, gamma) + eps
        
        # 4. 生成增益
        # G_H 现在在中高频区域有特定的响应值，在低频区域是另一个值
        noise_scale = G_H 
        
        # 关键修正：强制低频部分的增益为 1.0 (即无扰动)
        # 或者你可以让低频也有微小扰动，但为了突出中高频，建议直接置1
        noise_scale[:low_cutoff] = 0.0 # scale为0意味着 gain 恒为 1 (loc=1.0)
        
        gain = np.random.normal(loc=1.0, scale=noise_scale, size=N)
        gain = np.clip(gain, a_min=eps, a_max=None)
        
        aug_spec = G * gain
        
        # 归一化回原始范围通常是个好习惯，防止幅值爆炸
        # 但要注意不要破坏相对关系，这里简单的 min-max 可能会导致相位或相对幅值失真
        # 如果只是做数据增强，通常不需要严格归一化回 0-1，保持相对变化即可
        # 如果你的下游模型对幅值敏感，请保留归一化；如果是分类任务，通常影响不大
        
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