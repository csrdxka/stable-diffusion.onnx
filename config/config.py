class Config:
    def __init__(self):
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda'
        self.dtype = 'float32'
        self.batch_size = 1
        self.unet_channels = 4
        self.latent_height = 512
        self.latent_width = 512
        self.guidance = 7.5
        self.num_inference_steps = 50
        self.vae_scale_factor = 0.18215

        self.model_path = './models'