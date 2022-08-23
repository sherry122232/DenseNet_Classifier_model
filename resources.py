# -*- coding=GBK -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

filename = os.listdir("C:/Users/Administrator/Desktop/wsy/jiantou/data/validate/you/")
base_dir = "C:/Users/Administrator/Desktop/wsy/jiantou/data/validate/you/"
save = "C:/Users/Administrator/Desktop/wsy/jiantou/data/validate/you/"

# 求出图像均值作为阈值来二值化
def custom_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    img_norm = cv.normalize(gray, dst=None, alpha=200, beta=25, norm_type=cv.NORM_MINMAX)
    # cv.imshow("origin", img_norm)


    h, w = img_norm.shape[:2]
    m = np.reshape(img_norm, [1, w * h])  # 化为一维数组
    mean = m.sum() / (w * h)-1
    # print("mean: ", mean)
    ret, binary = cv.threshold(img_norm, mean, 255, cv.THRESH_BINARY)
    # cv.imshow(save + img, binary)
    cv.imwrite(save + img, binary)

for img in filename:
    image = Image.open(base_dir + img)
    src = cv.imread(base_dir + img)
    custom_image(src)
    # cv.waitKey(0)


    # image_size.save(base_dir+ img)
    # cv.destroyAllWindows()
