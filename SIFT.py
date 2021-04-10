import cv2
import numpy as np


def getSIFT(img, mask):
    # img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    # print("mask sum", np.sum(mask))
    kp, des = orb.detectAndCompute(img, mask)
    imgKP = cv2.drawKeypoints(img, kp, None)
    cv2.imwrite("image.jpg", imgKP)
    return kp, des


def getMatch(img1, img2, k=2, mask1=None, mask2=None):
    kp1, des1 = getSIFT(img1, mask1)
    kp2, des2 = getSIFT(img2, mask2)
    bf = cv2.BFMatcher()
    print(len(des1), len(des2))
    matches = bf.knnMatch(des1, des2, k)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    list_kp1 = []
    list_kp2 = []
    # print("good:", len(matches))
    for mats in matches:
        mat = mats[0]
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
    return list_kp1, list_kp2
