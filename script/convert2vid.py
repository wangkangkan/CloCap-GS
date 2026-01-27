import cv2
import os

def images_to_video(image_folder, output_video_path, fps=30):
    # 获取图像文件列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # 确保按顺序处理图像

    # 获取图像尺寸
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    size = (width, height)

    # 创建视频写入对象
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    # 释放视频写入对象
    out.release()

# 示例用法
image_folder = 'data/result/if_nerf/nocorrd2/comparison/2'  # 替换为你的图像文件夹路径
output_video_path = 'gt_8.mp4'  # 替换为你想要保存的视频路径
fps = 30  # 设置帧率

images_to_video(image_folder, output_video_path, fps)
