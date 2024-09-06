from diffusers import DiffusionPipeline
from PIL import Image
import imageio
import numpy as np
import torch

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 替换为你的图像文件路径
image_path = "/Users/skylife/Documents/huggingface/test.jpg"
image = Image.open(image_path)

# 确保图像转换为模型所需的格式
image = image.convert("RGB")

# 加载预训练的图像到视频扩散模型，并将其移动到GPU
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt")
pipeline.to(device)

# 生成视频帧
num_frames = 120  # 生成的帧数
video_frames = []

for _ in range(num_frames):
    frame = pipeline(image, num_inference_steps=50).images[0]  # 你可以调整num_inference_steps
    video_frames.append(frame)

# 创建一个视频写入器对象，指定输出文件和帧率
fps = 24  # 指定帧率
writer = imageio.get_writer('output_video.mp4', fps=fps)

# 将每一帧添加到视频文件
for frame in video_frames:
    # 将PIL图像转换为numpy数组
    frame_array = np.array(frame)
    writer.append_data(frame_array)

# 关闭写入器，完成视频保存
writer.close()