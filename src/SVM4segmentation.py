import cv2
import numpy as np
from sklearn.svm import SVC

# 加载图像
img = cv2.imread("image_video1/image/1.jpg")
# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 预处理图像以提取特征
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 100, 200)

# 为SVM分类器准备数据
X = []
Y = []
for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        X.append([i, j])
        Y.append(edges[i, j])

# 创建SVM分类器
clf = SVC(kernel='linear')

# 使用数据训练SVM分类器
clf.fit(X, Y)

# 创建一个新的图像来保存分割结果
segmented = np.zeros_like(gray)

# 对图像中的每个像素进行分类
for i in range(segmented.shape[0]):
    for j in range(segmented.shape[1]):
        # 获取当前像素的特征向量
        feature_vector = np.array([i, j]).reshape(1, -1)
        # 使用SVM分类器对像素进行分类
        prediction = clf.predict(feature_vector)
        # 如果像素被分类为边缘，则将其标记为白色
        if prediction == 1:
            segmented[i, j] = 255

# 显示分割结果
cv2.imshow("Segmented Image", segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
