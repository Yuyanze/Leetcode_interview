# -*- coding: utf-8 -*-
"""
Created on Tue 28 Apr 2020
@author: Hilbert YU
"""
import cv2
import cv2 as cv
import numpy as np
from time import *
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import heapq


def GetImageFromDir(targetPath):
    ImageName = os.listdir(targetPath)
    ImageName.sort(key=lambda x: int(x[:-4]))
    ImageNames = []
    for filename in ImageName:
        ImageNames.append(os.path.join(targetPath, filename))
    return ImageNames

def DenoisebyDBSCAN(CenterData, EPS, MIN_SAMPLES):

    CenterData = CenterData[CenterData['Y'] >= 200].copy()
    # CenterX = CenterData['X']
    # CenterY = CenterData['Y']
    #plt.plot(CenterX, CenterY, color='m', linestyle='', marker='o', label=u'原始点')  # Time is very long
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)  # Frame_2_LightWeak
    #dbscan = DBSCAN(eps = 150, min_samples = 15)  #
    dbscan.fit(CenterData)
    label_pred = dbscan.labels_
    CenterData['label'] = label_pred

    if max(label_pred)== -1:
        TipX = np.mean(CenterData['X'])
        TipY = np.mean(CenterData['Y'])
    else:
        CenterData = CenterData[~(CenterData['label'].isin([-1]))]
        # CenterData = CenterData[(CenterData['label'] == 0)]
        denoisey = list(CenterData['Y'])
        minIndexy = list(map(denoisey.index, heapq.nsmallest(10, denoisey)))
        smallest_ten_value = CenterData.iloc[minIndexy, 0:2]
        TipX = np.mean(smallest_ten_value['X'])
        #TipY = np.mean(smallest_ten_value['Y'])
        TipY = CenterData.iloc[minIndexy[0], 1]
        # plt.plot(TipX, TipY, 'ro')
        # plt.plot(CenterData['X'], CenterData['Y'], 'bo')
        # plt.show()
    return TipX, TipY, CenterData

# def EstimateBladeTipClearance(ImagePath):
#     # Step 1: input the Image path, you can choose different
#     ImageNames = GetImageFromDir(ImagePath)
#     # Step 2：
#     currentframe = cv.imread(ImageNames[0])
#     currentframe = cv2.medianBlur(currentframe, 9)
#     currentframegry = cv2.cvtColor(currentframe, cv2.COLOR_BGR2GRAY)
#
#     nextframe = cv.imread(ImageNames[1])
#     nextframe = cv2.medianBlur(nextframe, 9)
#     nextframegry = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
#
#     afternextframe = cv.imread(ImageNames[2])
#     afternextframe = cv2.medianBlur(afternextframe, 9)
#     afternextframegry = cv2.cvtColor(afternextframe, cv2.COLOR_BGR2GRAY)
#     count = 0
#
#     for ImageName in ImageNames[3:]:
#         count = count + 1
#         print((count, ImageName))
#
#         frameTempbuffer = cv.imread(ImageName)
#         frameTempbuffergry = cv2.cvtColor(frameTempbuffer, cv2.COLOR_BGR2GRAY)
#
#         currentdiffone = cv2.absdiff(currentframegry, nextframegry)
#         currentdifftwo = cv2.absdiff(nextframegry, afternextframegry)
#         # 二值化
#         binaryone = cv2.adaptiveThreshold(currentdiffone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#                                           9, 9)
#         binarytwo = cv2.adaptiveThreshold(currentdifftwo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#                                           9, 9)
#
#         binary = cv2.bitwise_and(binaryone, binarytwo)
#         #binary = sm.binary_dilation(binary, selem=sm.disk(5)).astype(np.uint8)
#         #morphologyone = sm.binary_dilation(binaryone, selem=sm.disk(5)).astype(np.uint8)
#         # binaryResize = cv2.resize(binary, (int(width / 2), int(height / 2)))
#         #cv.imshow("binary.jpg", binary)
#         #cv.waitKey(50)
#
#         # cv.imshow("binary.jpg", binaryResize)
#         # cv.waitKey(5)
#
#         contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(nextframe, contours, -1, (0, 255, 0), 5)
#         area    = []
#         CenterX = []
#         CenterY = []
#         X = []
#         if contours:         #If the contours is not NULL
#             for k in range(len(contours)):
#                 M = cv2.moments(contours[k])
#                 if M["m00"] > 0:
#                     center_x = int(M["m10"] / M["m00"])
#                     center_y = int(M["m01"] / M["m00"])
#                 else:
#                     #print(contours[k])
#                     center_x = contours[k][0][0][0]
#                     center_y = contours[k][0][0][1]
#                 CenterX.append(center_x)
#                 CenterY.append(center_y)
#                 cv2.circle(nextframe, (int(center_x), int(center_y)), 5, (255, 0, 255), 2)
#             # print(CenterX)
#             # print(CenterY)
#             CenterData = pd.DataFrame([CenterX, CenterY])
#             CenterData = pd.DataFrame(CenterData.values.T, columns=['X', 'Y'])
#             if len(CenterX)>40:
#                # Step 3: 利用DBSCAN算法对轮廓中心数据进行数据处理()
#                TipX, TipY, CenterData = DenoisebyDBSCAN(CenterData,150,15)   # First denoise
#                #print(len(CenterX))
#                if len(CenterX) > 200:# Pointer number is lower than 150, 150 dependence on binary_kernel
#                #if len(CenterX) > 150:# Pointer number is lower than 150, 150 dependence on binary_kernel
#                   TipX, TipY, CenterData = DenoisebyDBSCAN(CenterData,10,6)    #  Second denoise
#             else:
#                 TipX = -1
#                 TipY = -1
#             # Step 4： Tip 点
#             cv2.circle(nextframe, (int(TipX), int(TipY)), 5, (0, 255, 255), 5)
#             cv.imshow("nextframegry.jpg", nextframe)
#             cv.waitKey(1000)
#         currentframegry, nextframegry, afternextframegry = nextframegry, afternextframegry, frameTempbuffergry
#         currentframe, nextframe, afternextframe = nextframe, afternextframe, frameTempbuffer
#     return TipX, TipY

def EstimateBladeTipClearance(firstImage, secondImage, ThirdImage):
    currentframe = cv.imread(firstImage)
    currentframe = cv2.medianBlur(currentframe, 3)
    nextframe    = cv.imread(secondImage)
    nextframe = cv2.medianBlur(nextframe, 3)
    afternextframe = cv.imread(ThirdImage)
    afternextframe = cv2.medianBlur(afternextframe, 3)

    currentframegry = cv2.cvtColor(currentframe, cv2.COLOR_BGR2GRAY)
    nextframegry = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
    afternextframergry = cv2.cvtColor(afternextframe, cv2.COLOR_BGR2GRAY)


    currentdiffone = cv2.absdiff(currentframegry, nextframegry)
    currentdifftwo = cv2.absdiff(nextframegry, afternextframergry)

    binaryone = cv2.adaptiveThreshold(currentdiffone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      9, 9)
    binarytwo = cv2.adaptiveThreshold(currentdifftwo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      9, 9)

    binary = cv2.bitwise_and(binaryone, binarytwo)
    #binary = sm.binary_dilation(binary, selem=sm.disk(5)).astype(np.uint8)
    #morphologyone = sm.binary_dilation(binaryone, selem=sm.disk(5)).astype(np.uint8)
    # binaryResize = cv2.resize(binary, (int(width / 2), int(height / 2)))
    # cv.imshow("binary.jpg", binary)
    # cv.waitKey(1000)

    # cv.imshow("binary.jpg", binaryResize)
    # cv.waitKey(5)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(nextframe, contours, -1, (0, 255, 0), 3)
    CenterX = []
    CenterY = []

    TipX=-1
    TipY=-1
    if contours:
        for k in range(len(contours)):
            M = cv2.moments(contours[k])
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                #print(contours[k])
                center_x = contours[k][0][0][0]
                center_y = contours[k][0][0][1]
            CenterX.append(center_x)
            CenterY.append(center_y)
            cv2.circle(nextframe, (int(center_x), int(center_y)), 5, (255, 0, 255), 2)
        # print(CenterX)
        # print(CenterY)
        CenterData = pd.DataFrame([CenterX, CenterY])
        CenterData = pd.DataFrame(CenterData.values.T, columns=['X', 'Y'])
        if len(CenterX)>40:
           TipX, TipY, CenterData= DenoisebyDBSCAN(CenterData,150,15)   # First denoise
        if len(CenterX)>150: # Pointer number is lower than 150, 150 dependence on binary_kernel
           TipX, TipY, CenterData = DenoisebyDBSCAN(CenterData,10,6)    #  Second denoise

        # Step 4： Tip
        # cv2.circle(nextframe, (int(TipX), int(TipY)), 5, (0, 255, 255), 5)
        # cv.imshow("nextframegry.jpg", nextframe)
        # cv.waitKey(1000)
        # firstImage, secondImage = nextframe, afternextframe
        # currentframegry, nextframegry = nextframegry, afternextframergry
    return TipX, TipY, nextframe

if __name__ == '__main__':

    targetPath = 'C:/HoneyWell/Aero_AI/Goldwind/GW_0424/code/DataSet/Frame_1_Normal/'
    #targetPath = 'C:/HoneyWell/Aero_AI/Goldwind/GW_0424/code/DataSet/Frame_2_LightWeak/'
    #targetPath = 'C:/HoneyWell/Aero_AI/Goldwind/GW_0424/code/DataSet/Frame_3_HeavyWeak/'
    #targetPath  = 'C:/HoneyWell/Aero_AI/Goldwind/GW_0424/code/DataSet/Frame_5_Strong/'
    #targetPath = 'C:/HoneyWell/Aero_AI/Goldwind/GW_0424/code/DataSet/Frame_5_Strong_Test/'
    #targetPath = 'C:/HoneyWell/Aero_AI/Goldwind/GW_0424/code/DataSet/Frame_6_blur/'
    ImageNames = GetImageFromDir(targetPath)
    firstImage = ImageNames[118]
    secondImage = ImageNames[119]
    ThirdImage = ImageNames[120]
    #TipX, TipY  = EstimateBladeTipClearance(targetPath)
    TipX, TipY, nextframe = EstimateBladeTipClearance(firstImage,secondImage,ThirdImage)
    print(int(TipX), int(TipY))



