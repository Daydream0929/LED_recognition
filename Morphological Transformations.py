### 形态转换 ：腐蚀 膨胀 开 闭 形态梯度 顶帽 黑帽
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('C:\\Users\\941917\\Desktop\\Mycode\\Seagate_ImageProcessing\\original.png')
blur = cv.blur(img,(5,5))



plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
