import cv2
import numpy as np
import os

path = './image_video1/res_img'
img = cv2.imread(os.path.join(path,'01.jpg'))
# 获取图片尺寸
imgInfo = img.shape
size = (imgInfo[1],imgInfo[0])
print(size)
filelist = os.listdir(path)
filelist.sort()
fps = 4  # 视频每秒组成的原始帧数，由于之前是抽帧提取图片的，我想在这里设小一点
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')				# 设置视频编码格式
fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
video = cv2.VideoWriter('./image_video1/video1.mp4', fourcc, fps, size)
print('a')
# 视频保存在当前目录下
for item in filelist:
    if item.endswith('.jpg') or item.endswith('.JPG'):
        print(item)
        item = os.path.join(path,item)
        # 路径中若存在为中文名
        # img = cv2.imdecode(np.fromfile(item, dtype=np.uint8), 1)
        # 路径为英文名
        img = cv2.imread(item)
        print(img.shape)
        video.write(img)

video.release()
