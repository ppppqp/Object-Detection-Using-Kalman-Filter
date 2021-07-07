import cv2
import numpy as np


def getSIFT(img, mask, imgNo=0):
    # img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    orb = cv2.BRISK_create()
    # print("mask sum", np.sum(mask))
    kp, des = orb.detectAndCompute(img, mask)
    imgKP = cv2.drawKeypoints(img, kp, None)
    # if(imgNo > 0):
    #     cv2.imwrite("./images/"+str(imgNo) + "_kp.jpg", imgKP)
    if len(kp) == 0:
        return kp, []
    return kp, des


def getMatch(img1, img2, k, mask1=None, mask2=None):
    kp1, des1 = getSIFT(img1, mask1, 0)
    kp2, des2 = getSIFT(img2, mask2, 0)
    bf = cv2.BFMatcher()
    if(len(des1) == 0 or len(des2) == 0):
        return [], [], []
    pre_matches = bf.knnMatch(des1, des2, k)
    # print(len(des2))
    if(len(des2) <= 2):
        return [], [], []
    # matches = bf.knnMatch(des1, des2, k)
    goodMatch = []
    if(len(pre_matches) <= 4):
        return [], [], []
    for m, n in pre_matches:
        if m.distance < 0.75 * n.distance:
            goodMatch.append([m])

    list_kp1 = []
    list_kp2 = []
    goodDes2 = []
    for mats in goodMatch:
        mat = mats[0]
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        goodDes2.append(des2[img2_idx].tolist())
        list_kp1.append([int(x1), int(y1)])
        list_kp2.append([int(x2), int(y2)])

    # if(len(matches) <= 4):
    #     return [], []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append([m])

    # if(len(good) < 2):
    #     return [], []
    # for mats in good:
    #     mat = mats[0]
    #     img1_idx = mat.queryIdx
    #     img2_idx = mat.trainIdx
    #     (x1, y1) = kp1[img1_idx].pt
    #     (x2, y2) = kp2[img2_idx].pt
    #     list_kp1.append((int(x1), int(y1)))
    #     list_kp2.append((int(x2), int(y2)))
    return list_kp1, list_kp2, goodDes2
