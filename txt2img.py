from pipeline import StableDiffusionPipeline
from config import Config
import os

def main():
    config = Config()
    
    # 初始化StableDiffusionPipeline
    pipeline = StableDiffusionPipeline(config)

    # 设置提示词和负向提示词
    prompt = "a photo of a small cute dog"
    negative_prompt = ""

    # 运行pipeline
    images = pipeline.run(prompt=prompt, negative_prompt=negative_prompt)

    # 保存生成的图像
    os.makedirs("output", exist_ok=True)
    images[0].save("output/output.png")
    print("Image generated and saved as output/output.png")

if __name__ == "__main__":
    main()
