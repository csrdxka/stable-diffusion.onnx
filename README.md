# Stable Diffusion ONNX

这是一个基于ONNX Runtime的Stable Diffusion实现，旨在提供高效、易于部署的文本到图像生成解决方案。

## 项目概述

本项目将Stable Diffusion模型转换为ONNX格式，并提供了一个简洁的Python接口用于文本到图像的生成。主要特点包括：

- 使用ONNX Runtime进行高效推理
- 支持CPU和CUDA加速
- 实现了DDIM采样器
- 简单易用的API接口

## 安装要求

- Python 3.8+
- PyTorch
- ONNX Runtime (CPU或GPU版本)
- Transformers
- NumPy
- Pillow
- tqdm

## 模型下载

在运行之前，您需要下载转换好的ONNX模型文件。模型文件应放置在`./models`目录下

## 快速开始

### 基本用法

```python
from pipeline import StableDiffusionPipeline
from config import Config

# 初始化配置
config = Config()

# 初始化pipeline
pipeline = StableDiffusionPipeline(config)

# 设置提示词和负向提示词
prompt = "a photo of a small cat"
negative_prompt = ""

# 生成图像
images = pipeline.run(prompt=prompt, negative_prompt=negative_prompt)

# 保存图像
images[0].save("output.png")
```

### 使用示例脚本

项目提供了一个简单的示例脚本`txt2img.py`，可以直接运行：

```bash
python txt2img.py
```

## 配置参数

在`config/config.py`中可以修改以下配置参数：

- `device`: 使用的设备 ('cuda' 或 'cpu')
- `batch_size`: 批处理大小
- `latent_height/width`: 生成图像的高度和宽度
- `guidance`: 分类器自由引导比例
- `num_inference_steps`: 推理步数
- `model_path`: 模型文件路径

## 项目结构

```
stable-diffusion.onnx/
├── config/
│   ├── __init__.py
│   └── config.py
├── pipeline/
│   ├── __init__.py
│   └── stable_diffusion_pipeline.py
├── scheduler/
│   ├── __init__.py
│   └── ddim_scheduler.py
├── models/
│   └── (模型文件)
├── txt2img.py
└── README.md
```

## 自定义采样器

本项目实现了DDIM采样器，位于`scheduler/ddim_scheduler.py`。您可以通过修改采样器参数来调整生成过程：

```python
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False
)
```

## 高级用法

### 设置随机种子

```python
# 使用固定种子以获得可重复的结果
images = pipeline.run(prompt=prompt, negative_prompt=negative_prompt, seed=42)
```

### 调整生成参数

您可以通过修改Config对象来调整生成参数：

```python
config = Config()
config.guidance = 9.0  # 增加引导比例
config.num_inference_steps = 30  # 减少推理步数以加快生成
```

## 性能优化

- 对于大多数用户，推荐使用CUDA加速
- VAE解码器默认在CPU上运行，因为它通常不是性能瓶颈
- 可以通过减少`num_inference_steps`来加快生成速度，但可能会影响图像质量

## 许可证

[添加您的许可证信息]

## 致谢

本项目基于Stable Diffusion模型，感谢原始模型的开发者和贡献者。
```

这份README文件提供了项目的基本介绍、安装要求、使用方法、配置参数以及项目结构等信息。您可以根据需要进一步完善，特别是添加许可证信息和其他特定的项目细节。