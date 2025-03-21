import numpy as np

class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        clip_sample: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample

        # 设置噪声调度
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"不支持的beta调度: {beta_schedule}")
        
        # 计算 alphas 及其累积乘积（注意索引对应训练过程中的时间步，0～num_train_timesteps-1）
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # 记录最后一个时间步的alpha累积乘积
        self.final_alpha_cumprod = self.alphas_cumprod[0]
        
        # 初始噪声标准差
        self.init_noise_sigma = 1.0
        
        # 初始化时间步数组（存储训练时间步的下标），按降序排列
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        
    def set_timesteps(self, num_inference_steps: int):
        """设置推理时间步，选取训练时间步的子集，并按降序排列"""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = np.asarray(list(range(0, self.num_train_timesteps, step_ratio)))
        timesteps = np.flip(timesteps).copy()
        self.timesteps = timesteps
        
    def scale_model_input(self, sample, timestep):
        """
        根据时间步缩放模型输入
        注意：这里传入的 timestep 应该是训练时的时间步下标（例如 self.timesteps 中的某个值）
        """
        # 直接使用训练时间步作为索引
        sigma = self.alphas_cumprod[timestep] ** 0.5
        return sample / ((sigma**2 + 1) ** 0.5)
    
    def step(
        self,
        model_output,
        timestep: int,
        sample,
        eta: float = 0.0,
    ):
        """
        执行DDIM采样步骤

        参数:
          model_output: 模型输出
          timestep: 当前训练时间步（应为 self.timesteps 中的一个值）
          sample: 当前样本
          eta: 随机性控制参数

        返回:
          一个对象，其属性 prev_sample 为前一步的样本
        """
        # 通过 self.timesteps 确定当前 timestep 在采样序列中的位置
        step_index = np.where(self.timesteps == timestep)[0][0]
        
        # 根据采样序列确定前一个训练时间步
        if step_index < len(self.timesteps) - 1:
            prev_timestep = self.timesteps[step_index + 1]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        else:
            alpha_prod_t_prev = self.final_alpha_cumprod
        
        # 当前时间步对应的alpha累积乘积，注意直接用传入的 timestep 作为索引
        alpha_prod_t = self.alphas_cumprod[timestep]
        
        # 计算方差
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # 计算噪声尺度 sigma_t（当 eta=0 时为0，从而采样过程确定）
        sigma_t = eta * np.sqrt(variance)
        
        # 计算预测的原始样本
        pred_original_sample = (sample - np.sqrt(1 - alpha_prod_t) * model_output) / np.sqrt(alpha_prod_t)
        
        # 计算从 x_t 到 x_{t-1} 的方向项，注意 eta 对 variance 的影响应为 eta**2 * variance
        pred_sample_direction = (np.sqrt(1 - alpha_prod_t_prev - (eta ** 2) * variance)) * model_output

        # 如果不是最后一步，则添加随机噪声项
        if step_index < len(self.timesteps) - 1:
            noise = np.random.randn(*sample.shape)
        else:
            noise = 0

        # 计算前一个样本
        prev_sample = np.sqrt(alpha_prod_t_prev) * pred_original_sample + np.sqrt(1 - alpha_prod_t_prev - sigma_t**2) * pred_sample_direction + sigma_t * noise
        
        # 裁剪样本（如果需要）
        if self.clip_sample:
            prev_sample = np.clip(prev_sample, -1, 1)
            
        return type('obj', (object,), {'prev_sample': prev_sample})