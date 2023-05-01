import cv2
import numpy as np

# 读取图像
img = cv2.imread('image_video1/image/6.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 阈值处理
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 形态学操作
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

# 距离变换
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# 背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 找到未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记分水岭算法
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [255,0,0]
img[markers == 0] = [0,255,0]
img[markers == 1] = [100,100,255]
img[markers == 2] = [100,100,100]
img[markers == 3] = [200,100,50]
# 显示图像
cv2.imshow('Input Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # 读取图像
# img = cv2.imread('image_video1/image/1.jpg')

# # 转为灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 使用Canny边缘检测算子进行边缘检测
# edges = cv2.Canny(gray, 50, 150)

# # 进行形态学操作
# kernel = np.ones((3,3),np.uint8)
# closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# # 进行分水岭算法处理
# dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(closing,sure_fg)
# ret, markers = cv2.connectedComponents(sure_fg)
# markers = markers+1
# markers[unknown==255] = 0
# markers = cv2.watershed(img,markers)
# # img[markers == -1] = [255,0,0]
# img[markers == 1] = [222, 244, 32]
# img[markers == 2] = [53, 235, 26 ]
# img[markers == 3] = [16, 130, 241 ]
# img[markers == 4] = [103, 54, 243 ]
# img[markers == 5] = [192, 29, 245 ]
# img[markers == 6] = [237, 18, 247 ]
# # 显示结果
# cv2.imshow('result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
