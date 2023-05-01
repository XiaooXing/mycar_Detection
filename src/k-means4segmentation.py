# import numpy as np
# import cv2

# # 加载图像
# img = cv2.imread('image_video1/image/1.jpg')

# # 转换为RGB颜色空间
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # 将图像转换为一维向量
# pixels = img.reshape((-1, 3))

# # 执行k-means聚类算法
# k = 3
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# flags = cv2.KMEANS_RANDOM_CENTERS
# compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)

# # 将每个像素的标签转换为对应的颜色
# segmented_image = centers[labels.flatten()].reshape(img.shape)

# # 显示分割后的图像
# cv2.imshow('Segmented Image', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import numpy as np
import cv2 as cv
img = cv.imread('image_video1/image/1.jpg')
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 6
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imshow('res2',res2)
cv.waitKey(0)
cv.destroyAllWindows()
