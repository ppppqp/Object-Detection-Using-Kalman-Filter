from SIFT import getSIFT, getMatch
import cv2
import numpy as np
# img = cv2.imread("./Vid_A_ball/img0002.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("img", img)
img1Path = "./Vid_A_ball/img0003.jpg"
img2Path = "./Vid_A_ball/img0004.jpg"
img1 = cv2.imread(img1Path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2Path, cv2.IMREAD_GRAYSCALE)
mask = np.zeros((img1.shape[0], img1.shape[1])).astype(np.uint8)
[x, y, w, h] = [200, 113, 45, 45]
mask[y:y+h, x:x+w] = 1
list1, list2 = getMatch(img1, img2, 2, mask)
# matchRes = cv2.drawMatchesKnn(img1, kp1, img2, kp2, match, None, flags=2)
# cv2.imshow("img3", matchRes)
cv2.waitKey(5000)
cv2.destroyAllWindows()
# [<DMatch 0x7fe7d3794a90>],
