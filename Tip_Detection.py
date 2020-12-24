# -*- coding: utf-8 -*-
"""
Created on Tue 28 Apr 2020
@author: Hilbert YU
"""
import cv2
import cv2 as cv
import numpy as np
import time
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import heapq

def GetImageFromDir(targetPath):
    # 获取文件夹中图像并排序
    ImageName = os.listdir(targetPath)
    ImageName.sort(key=lambda x: int(x[:-4]))
    # 将图像文件存储在ImageNames里
    ImageNames = []
    for filename in ImageName:
        ImageNames.append(os.path.join(targetPath, filename))
    return ImageNames

if __name__ == '__main__':
    # Step 1: input the Image path, you can choose different
    targetPath = '/home/hilbert/AI-Practice-Project/AI-Aero/Goldwind/DataSet/Frame_1_Normal/'
    #targetPath  = '/home/hilbert/AI-Practice-Project/AI-Aero/Goldwind/DataSet/Frame_2_LightWeak/'
    #targetPath = '/home/hilbert/AI-Practice-Project/AI-Aero/Goldwind/DataSet/Frame_3_HeavyWeak/'
    ImageNames= GetImageFromDir(targetPath)
    # Step 2：利用帧差法求风机轮廓区域
    #第一帧
    currentframe  = cv.imread(ImageNames[0])
    currentframegry = cv2.cvtColor(currentframe, cv2.COLOR_BGR2GRAY)
    #第二帧
    nextframe = cv.imread(ImageNames[1])
    nextframegry = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
    #第三帧
    afternextframe = cv.imread(ImageNames[2])
    afternextframegry      = cv2.cvtColor(afternextframe, cv2.COLOR_BGR2GRAY)

    count = 0
    for ImageName in ImageNames:
        count = count + 1
        print((count, ImageName))
        #下一帧
        frameTempbuffer        = cv.imread(ImageName)
        frameTempbuffergry     = cv2.cvtColor(frameTempbuffer, cv2.COLOR_BGR2GRAY)
        #前后两帧作差
        currentdiffone = cv2.absdiff(currentframegry, nextframegry)
        currentdifftwo = cv2.absdiff(nextframegry, afternextframegry)

        #currentdifftwo = cv2.absdiff(afternextframegry, currentframegry)
        #二值化
        binaryone = cv2.adaptiveThreshold(currentdiffone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                          5, 5)
        binarytwo = cv2.adaptiveThreshold(currentdifftwo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                         5, 5)
        #取交集得到afternextframe风机区域二值图
        binary = cv2.bitwise_and(binaryone, binarytwo)
        # cv.imshow("binary.jpg", binary)
        # cv.waitKey(0)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = []
        CenterX = []
        CenterY = []
        # 计算第k条轮廓的各阶矩
        X = []
        for k in range(len(contours)):
            M = cv2.moments(contours[k])
            if M["m00"] > 0:
               center_x = int(M["m10"] / M["m00"])
               center_y = int(M["m01"] / M["m00"])
               CenterX.append(center_x)
               CenterY.append(center_y)
        print(CenterX)
        print(CenterY)
        CenterData = pd.DataFrame([CenterX, CenterY])
        CenterData = pd.DataFrame(CenterData.values.T,columns=['X','Y'])

        # Step 3: 利用DBSCAN算法对轮廓中心数据进行数据处理
        if CenterX: #
           #plt.plot(CenterX, CenterY, color='m', linestyle='', marker='o', label=u'原始点')
           # 不同光照条件下的 eps 与 min_samples设置不同，选择不同文件夹时，我初步设置了如下，仅供参考，需要优化
           #dbscan = DBSCAN(eps=15, min_samples=6)    #   Frame_1_Normal
           dbscan = DBSCAN(eps=150, min_samples=15)  #   Frame_2_LightWeak
           #dbscan = DBSCAN(eps = 150, min_samples = 15)  #   Frame_3_HeavyWeak eps适当越小越好（150-300），min_sample(10-25)适当越大越好
           dbscan.fit(CenterData)
           label_pred = dbscan.labels_
           CenterData['label'] = label_pred
           if max(label_pred) == -1:             #如果只有异常点，不进行处理
               TipX = np.mean(CenterData['X'])
               TipY = np.mean(CenterData['Y'])
           else:
              CenterData = CenterData[~(CenterData['label'].isin([-1]))] #删除噪声点
              #CenterData = CenterData[(CenterData['label'] == 0)]        # 指定边缘点位“0”类只适用于第三种情况，需要打开
              denoisey   =  list(CenterData['Y'])
              minIndexy  = list(map(denoisey.index, heapq.nsmallest(1, denoisey))) #提取Y值最小的5个点

              smallest_ten_value = CenterData.iloc[minIndexy,0:2]
              #对此五个点坐标求平均，代表Tip点的坐标
              TipX = np.mean(smallest_ten_value['X'])
              TipY = np.mean(smallest_ten_value['Y'])

              #plt.plot(TipX, TipY, 'ro')
              #plt.plot(CenterData['X'], CenterData['Y'], 'bo')
              #plt.show()

        # Step 4： Tip 点
        #cv2.drawContours(afternextframe, contours, -1, (0, 255, 0), 3)
        cv2.circle(afternextframe,(int(TipX),int(TipY)),5,(0,0,255),10)
        cv.imshow("afternextframe.jpg", afternextframe)
        cv.waitKey(1000)
        currentframegry, nextframegry, afternextframegry = nextframegry, afternextframegry, frameTempbuffergry
        currentframe, nextframe, afternextframe = nextframe, afternextframe, frameTempbuffer

