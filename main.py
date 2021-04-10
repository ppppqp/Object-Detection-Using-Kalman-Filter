import numpy as np
import matplotlib.pyplot as plt
from SIFT import getSIFT, getMatch
from PIL import Image
from kf import KF
import cv2
plt.ion()
plt.figure()
position = [1, 1, 0, 0, 1, 1, 0, 0]
velocity = [0, 0, 1, 1, 0, 0, 1, 1]
frames = []
meas_variance = np.random.rand(4, 4)*0.01

initial_state = np.array([200, 113, 0, 0, 45, 45, 0, 0]).T
accel_variance = np.ones((8, 8))  # *np.random.rand()*0.1
kf = KF(initial_state=initial_state, accel_variance=accel_variance)

# [real_X, real_Y, real_W, real_H] = initial_state[position]

DT = 0.008
NUM_STEPS = 200
# MEAS_EVERY_STEPS = 20

mus = []
covs = []
real_xs = []
real_vs = []
im = Image.open("./image.jpg")
prevImg = cv2.imread("./Vid_A_ball/img0002.jpg")
prevMask = np.zeros((prevImg.shape[0], prevImg.shape[1])).astype(np.uint8)
x = 200
y = 113
w = 45
h = 45
prevMask[y:y+h, x:x+w] = 1
for step in range(NUM_STEPS):
    # if step > 500:
    #     real_v *= 0.9
    picNum = step + 3
    img = cv2.imread("./Vid_A_ball/img0" + str(picNum//100) +
                     str((picNum % 100)//10) + str(picNum % 10) + ".jpg")
    # covs.append(kf.cov)
    # mus.append(kf.mean)
    kf.predict(dt=DT)

    mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    pos = kf.pos.astype(np.int)
    x = pos[0, 0]
    y = pos[1, 0]
    w = pos[2, 0]
    h = pos[3, 0]
    print(x, y, w, h)

    rect = cv2.rectangle(prevImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
    imgPath = "img"+str(picNum)+".jpg"
    cv2.imwrite(imgPath, rect)
    im2 = Image.open(imgPath)
    frames.append(im2)
    mask[y:y+h, x:x+w] = 1
    list_kp1, list_kp2 = getMatch(prevImg, img, 2, prevMask, mask)
    new_x = np.min(np.array([item[0] for item in list_kp2])) - 5
    new_y = np.min(np.array([item[1] for item in list_kp2]))-5
    new_w = np.max(np.array([item[0] for item in list_kp2])) - new_x + 10
    new_h = np.max(np.array([item[1] for item in list_kp2])) - new_y + 10
    # if step != 0 and step % MEAS_EVERY_STEPS == 0:
    kf.update(meas_value=np.array([new_x, new_y, new_w, new_h]).T,  # real_x + np.random.randn() * np.sqrt(meas_variance),
              meas_variance=meas_variance)
    prevImg = img
    prevMask = mask

im.save("Gif_test.gif",
        save_all=True, append_images=frames)
