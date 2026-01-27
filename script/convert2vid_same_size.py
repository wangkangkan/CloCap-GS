import cv2
import os
import numpy as np
from PIL import Image

def resize_and_pad(image_path, target_size, idx = None):
    # 打开图片
    image = Image.open(image_path)
    # img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # x, y, w, h = boundbox[10*300+idx,:]
    # img = img[y:y + h, x:x + w]
    # image = Image.fromarray(img)
    
    # 保持比例缩放图片
    if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
        image.thumbnail(target_size, Image.LANCZOS)
    # 创建黑色背景图
    background = Image.new('RGB', target_size, (0, 0, 0))
    
    # 计算图像在背景图的中心位置
    image_position = (
        (target_size[0] - image.size[0]) // 2,
        (target_size[1] - image.size[1]) // 2
    )
    
    # 将缩放后的图像粘贴到背景图的中心位置
    background.paste(image, image_position)
    
    return background

def images_to_video(image_folder, output_video_path, fps=30):
    # 获取图像文件列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # 确保按顺序处理图像

    # 获取图像尺寸
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    size = (width, height)

    # 创建视频写入对象
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (450, 600))

    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        # img = cv2.imread(img_path)
        
        img = np.array(resize_and_pad(image_path=img_path, target_size=(450, 600), idx = i))
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    # 释放视频写入对象
    out.release()

# 示例用法
boundbox = np.loadtxt('data/result/if_nerf/final_3/allboundbox_700.txt').astype(np.int)
image_folder = 'data/result/if_nerf/final_1/comparison/4'  # 替换为你的图像文件夹路径
output_video_path = 'paper_low_4.mp4'  # 替换为你想要保存的视频路径
fps = 30  # 设置帧率

images_to_video(image_folder, output_video_path, fps)
