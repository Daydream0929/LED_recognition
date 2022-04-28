import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 通过将图像与低通滤波器内核进行卷积来实现图像模糊。这对于消除噪音很有用。
# 它实际上从图像中消除了高频部分（例如噪声，边缘）。因此，在此操作中边缘有些模糊。

# 1. Averaging 平均
# 2. Gaussian Blurring  高斯模糊
# 3. Median Blurring 中卫模糊
# 4. Bilateral Filtering 双边滤波

# 这里我们采用高斯模糊

img = cv.imread('C:\\Users\\941917\\Desktop\\Mycode\\Seagate_ImageProcessing\\original.png')
blur = cv.GaussianBlur(img,(3,3),0)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()