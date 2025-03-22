import os
import torch
from transformers import CLIPTokenizer
from scheduler.ddim_scheduler import DDIMScheduler
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from typing import List, Optional, Tuple, Union
from diffusers import DDIMScheduler as DiffusersDDIMScheduler

class StableDiffusionPipeline:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dtype = np.float32

        self.generator = np.random

        # Initialize providers
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.config.model_path, 'tokenizer'))
        self.text_encoder = ort.InferenceSession(os.path.join(self.config.model_path, 'text_encoder/model.onnx'), providers=providers)
        self.unet = ort.InferenceSession(os.path.join(self.config.model_path, 'unet/model.onnx'), providers=providers)
        self.vae_decoder = ort.InferenceSession(os.path.join(self.config.model_path, 'vae_decoder/model.onnx'), providers=['CPUExecutionProvider'])

        # Custom DDIM Scheduler
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False
        )

        # 使用 Diffusers 的调度器
        # self.scheduler = DiffusersDDIMScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     num_train_timesteps=1000,
        #     clip_sample=False
        # )
    
    def set_random_seed(self, seed):
        if isinstance(seed, int):
            self.generator.seed(seed)
        else:
            self.generator.seed(None)
    
    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width):
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = np.random.normal(0, 1, latents_shape).astype(np.float32)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def encode_prompt(self, prompt, negative_prompt, do_classifier_free_guidance=True):
        def tokenize(prompt):
            text_input_ids = (
                self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="np",
                )
                .input_ids.astype(np.int32)
            )
            output_name = self.text_encoder.get_outputs()[0].name
            text_embeddings = self.text_encoder.run(
                output_names=[output_name],
                input_feed={
                    "input_ids": text_input_ids,
                }
            )[0]
            return text_embeddings
        
        text_embeddings = tokenize(prompt).copy()
        uncond_embeddings = tokenize(negative_prompt)
        text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    def denoise_latent(self, latents, text_embeddings, timesteps, guidance):
        def check_variable_type(x):
            if isinstance(x, np.ndarray):
                print("变量是一个 np.array")
            elif isinstance(x, torch.Tensor):
                print("变量是一个 torch.Tensor")
            else:
                print("变量既不是 np.array, 也不是 torch.Tensor")
        do_classifier_free_guidance = guidance > 1.0

        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.scheduler.timesteps

        for timestep in tqdm(timesteps):
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

            params = {
                "sample": latent_model_input.astype(np.float32),
                "timestep": np.array([timestep], dtype=np.float32),
                "encoder_hidden_states": text_embeddings.astype(np.float32),
            }

            noise_pred = self.unet.run(
                None,
                params
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

            # 使用自定义的调度器
            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
            # 使用 Diffusers 的调度器
            # latents = self.scheduler.step(torch.from_numpy(noise_pred), timestep, torch.from_numpy(latents)).prev_sample.numpy()
        return latents
    def decode_latent(self, latents):
        images = self.vae_decoder.run(None, {"latent_sample": latents.astype(np.float32)})[0]
        return images
    
    @staticmethod
    def numpy_to_pil(images):
        # 将范围 [-1, 1] 转换为 [0, 255]
        images = (images + 1) * 255 / 2
        # 裁剪超出 [0, 255] 的数值
        images = np.clip(images, 0, 255)
        # 四舍五入后转换为 uint8 类型
        images = np.round(images).astype(np.uint8)
        # 调整维度从 (N, C, H, W) 到 (N, H, W, C)
        images = np.transpose(images, (0, 2, 3, 1))
        # 将每一张图转换为 PIL.Image 对象
        return [Image.fromarray(img) for img in images]
    
    def run(self, prompt, negative_prompt, seed=None):
        self.set_random_seed(seed)
        
        timesteps = None
        latents = self.initialize_latents(
            batch_size=self.config.batch_size,
            unet_channels=self.config.unet_channels,
            latent_height=(self.config.latent_height // 8),
            latent_width=(self.config.latent_width // 8)
        )
        do_classifier_free_guidance = self.config.guidance > 1.0
        text_embeddings = self.encode_prompt(prompt, negative_prompt, do_classifier_free_guidance=do_classifier_free_guidance)
        latents = self.denoise_latent(latents, text_embeddings, timesteps, self.config.guidance)

        images = self.decode_latent(latents / self.config.vae_scale_factor)
        images = self.numpy_to_pil(images)
        return images