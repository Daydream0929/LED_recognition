# 主函数
from calendar import c
from cgi import test
from tkinter import image_names
import cv2 as cv
import numpy as np
import sys
import inspect, re
import matplotlib.pyplot as plt


def varname(p):  # 获取变量p的名字
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)

# 通过Find_color寻找LED的范围
Color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([27, 99, 77]), 'Upper': np.array([42, 246, 189])},
             }


def read_picture(img): # 读入图片img
    if(img is None):
        return
    img = cv.imread('C:\\Users\\941917\\Desktop\\Mycode\\Seagate_ImageProcessing\\' + img + '.png')
    if img is None:
        sys.exit("Could not read the image!")
    return img

def show_picture(img): # 显示图片img (按任意键退出)
    if(img is None):
        return
    cv.imshow("Display Window (Press any key to exit!)", img)
    k = cv.waitKey(0)
    if k == ord("s"):
        return

def save_picture(img, name):  #将图片img保存为'img.png'
    if(img is None):
        return
    print(name)
    cv.imwrite(' ' + name + '.png', img)

def get_ROI(img):  # 获取图片中感兴趣的区域
    if(img is None):
        return
    ROI = img[200:300, 300:400]  #########################重点研究对象
    if img is None:
        sys.exit("Could not find the ROI!")
    return ROI

def BGR_to_HSV(img):  # 将BGR转化为HSV
    if(img is None):
        return
    Hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return Hsv

def BGR_to_GRAY(img): # 将BGR转化为GRAY
    if(img is None):
        return
    Gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return Gray
    
def BGR_to_BINARY(img):   # 二值化处理
    if(img is None):
        return
    Gray1 = BGR_to_GRAY(img)
    # 简单阈值法
    ret1,threshold1 = cv.threshold(Gray1, 141, 255, cv.THRESH_BINARY) # 这里的141 是通过下边的OTSU二值化返回出来的阈值
    ret2,threshold2 = cv.threshold(Gray1, 141, 255, cv.THRESH_BINARY_INV)
    ret3,threshold3 = cv.threshold(Gray1, 141, 255, cv.THRESH_TRUNC)
    ret4,threshold4 = cv.threshold(Gray1, 141, 255, cv.THRESH_TOZERO)
    ret5,threshold5 = cv.threshold(Gray1, 141, 255, cv.THRESH_TOZERO_INV)

    Gray2 = cv.medianBlur(Gray1, 5)
    # 自适应阈值
    threshold6 = cv.adaptiveThreshold(Gray2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    threshold7 =cv.adaptiveThreshold(Gray2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    
    Gray3 = BGR_to_GRAY(img)
    # OTSU二值化
    blur = cv.GaussianBlur(Gray3, (5, 5), 0)  # 先将其转化为高斯滤波
    ret8,threshold8 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return threshold8

def Bitwise_negation(img):  # 按位取反 将图片反转
    if(img is None):
        return
    Bitwise = cv.bitwise_not(img)
    return Bitwise


def Image_enlarge(img, x):   #将一副图像放大
    if(img is None):
        return
    h, w = img.shape[:2]
    x = int(x)
    res = cv.resize(img, (x * h, x * w), interpolation = cv.INTER_CUBIC)
    return res

def Image_shrink(img, x):  #将一副图像缩小
    if(img is None):
        return 
    h, w = img.shape[:2]
    x = int(x)
    res = cv.resize(img, ((int(h / x)), (int(w / x))), interpolation = cv.INTER_AREA)
    return res
    

def Image_segmentation(img):    #获取图片的五个部分 前提是原图要为类似"Int.png"这种比较整齐的图片
    if(img is None):
        return
    h = int(img.shape[0])
    w = (int)(img.shape[1])
    p1 = img[0 : (int)(h/4) , 0 : (int)(w/2)]
    p2 = img[0 : (int)(h/4) , (int)(w/2) : w]
    p3 = img[(int)(h/3) : (int)(h/3) * 2 , 0 : (int)(w/2)]
    p4 = img[(int)(h/3) : (int)(h/3) * 2 , (int)(w/2) : w]
    p5 = img[(int)(h/4) * 3 : h , (int)(w/4) : (int)(w/4) * 3]
    return [p1,p2,p3,p4,p5]



def draw_cont(img):  # 获取图像的轮廓
    if(img is None):
        return
    img_source = img
    img_hsv = BGR_to_HSV(img)
    img_Gauss = cv.GaussianBlur(img_hsv, (5, 5), 0)
    img_Gray = BGR_to_GRAY(img_Gauss)
    # img_canny = cv.Canny(img_Gray,100,200)
    ret, thresh = cv.threshold(img_Gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    p = cv.drawContours(img, contours, -1, (65,65,65), 3)

    '''
    mask = np.zeros(img_Gray.shape,np.uint8)
    mean_val = cv.mean(p, mask = mask)
    print(mean_val)
    print(len(contours))
    x = []
    y = []
    n = len(contours[0])
    for i in range(n):
        x.append(contours[0][i][0][0])
        y.append(contours[0][i][0][1])
    plt.scatter(x, y)
    plt.show()
    '''

    return p

    

Int = read_picture("Int")
p0 = Image_segmentation(Int)[0]   #mean_Val = [(0.6161473087818697, 254.3172804532578, 21.220963172804534, 0.0)]
p1 = Image_segmentation(Int)[1]
p2 = Image_segmentation(Int)[2]   #mean_val = [(0.974152785755313, 252.944859276278, 136.33773693279724, 0.0)]
p3 = Image_segmentation(Int)[3]
p4 = Image_segmentation(Int)[4]


pe = Image_enlarge(p0, 7)  ## 将源图像放大
p = draw_cont(pe)   ## 源图像的轮廓

show_picture(p)




'''
Int = read_picture("shuzi6")
Int_Gauss = cv.GaussianBlur(Int, (5, 5), 0)
#Int_edge = cv.Canny(Int_Gauss, 100, 200)
Int_Gray = BGR_to_GRAY(Int_Gauss)

ret, thresh = cv.threshold(Int_Gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
'''
'''
sz = len(contours)
for i in range (sz):
    p = cv.drawContours(Int_Gauss, contours, i, (0,255,0), 3)
    show_picture(p)
'''
# TTTTTTTTTTTTTTTTTTTTTTest
'''
p1 = read_picture("original")
guass = cv.GaussianBlur(p1,(5,5),0)
hsv = BGR_to_HSV(guass)
hsv_erode = cv.erode(hsv, None, iterations=2)  #膨胀


mask = cv.inRange(hsv_erode, Color_dist['green']['Lower'], Color_dist['green']['Upper']) #提取绿色特征
masked_green = cv.bitwise_and(p1, p1, mask=mask)  # 取反
mask2 = Bitwise_negation(mask)

show_picture(mask)



hsv = BGR_to_HSV(p1)
show_picture(hsv)


flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print( flags )


gray = BGR_to_GRAY(p1)  
show_picture(gray)

Bitwise = Bitwise_negation(p1)
show_picture(Bitwise)

Threshold1 = BGR_to_BINARY(p1)
show_picture(Threshold1)

'''



