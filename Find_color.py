# 点击图片即可获得其像素点的 BGR GRAY HSV 的值

import cv2 as cv

img = cv.imread('C:\\Users\\941917\\Desktop\\Mycode\\Seagate_ImageProcessing\\Int.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gauss = cv.GaussianBlur(img,(5,5),0)
hsv = cv.cvtColor(gauss, cv.COLOR_BGR2HSV)
hsv_erode = cv.erode(hsv, None, iterations=2)  #膨胀

Lower_color = [266, 266, 266]
Upper_color = [-1, -1, -1]

def mouse_click(event, x, y, flags, para):  
    if event == cv.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        print('PIX:', x, y)
        print("BGR:", img[y, x])
        print("GRAY:", gray[y, x])
        print("HSV:", hsv_erode[y, x])
        for i in range(3):
            if(hsv[y, x][i] > Upper_color[i]):
                Upper_color[i] = hsv[y, x][i]
            if(hsv[y, x][i] < Lower_color[i]):
                Lower_color[i] = hsv[y, x][i]
            
    

if __name__ == '__main__':
    cv.namedWindow("img")
    cv.setMouseCallback("img", mouse_click)
    while True:
        cv.imshow('img', img)
        if cv.waitKey() == ord('q'): #键入q退出
            print("Upper_color: ", Upper_color)
            print("Lower_color: ", Lower_color)
            break
    cv.destroyAllWindows()

## 全局绿色部分的BGR范围
## 第一次 'Lower': np.array([31, 61, 137]), 'Upper': np.array([47, 206, 254])


## 左下角绿色部分的BGR范围
## 第一次 'Lower': np.array([34, 131, 143]), 'Upper': np.array([37, 203, 185])
## 第二次 'Lower': np.array([27, 99, 77]), 'Upper': np.array([42, 246, 189])

## 左下角红色部门的BGR范围

