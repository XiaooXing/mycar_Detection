import cv2
import numpy as np
from sklearn import svm

# 加载图像
img = cv2.imread('image.jpg')

# 进行分水岭分割
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv2.dilate(opening,kernel,iterations=3)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

11111111111111111111111111

# 获取分割后的图像区域
num_objects = len(np.unique(markers)) - 1
object_imgs = []
for i in range(1, num_objects+1):
    object_img = np.zeros(img.shape, dtype=np.uint8)
    object_img[markers == i] = img[markers == i]
    object_imgs.append(object_img)

# 对每个图像区域进行分类
clf = svm.SVC()
X = []
y = []
for object_img in object_imgs:
    # 提取特征
    feature = extract_features(object_img)
    X.append(feature)
    # 获取标签
    label = get_label(object_img)
    y.append(label)
clf.fit(X, y)

# 对新图像进行分类
new_img = cv2.imread('new_image.jpg')
new_object_imgs = segment_image(new_img)
for new_object_img in new_object_imgs:
    feature = extract_features(new_object_img)
    label = clf.predict(feature)
    print('Object:', label)
