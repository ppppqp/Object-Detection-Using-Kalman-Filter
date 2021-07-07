import numpy as np
import matplotlib.pyplot as plt
from SIFT import getSIFT, getMatch
from PIL import Image
from kf import KF
from LS import LeastSquare
import cv2
plt.ion()
plt.figure()
position = [1, 1, 0, 0, 1, 1, 0, 0]
velocity = [0, 0, 1, 1, 0, 0, 1, 1]
# Blue: updates
# Green: measurement
# R: prediction
frames = []
testName = "dudek-face"
# testName = "girl"
imgName = "img"
if(testName == "girl"):
    zero = ""
    Format = ".png"
else:
    if testName == "OccludedFace2":
        zero = "00"
        Format = ".png"
    else:
        zero = "0"
        Format = ".jpg"
if(testName == "dudek-face"):
    zero = "0"
    imgName = "frame"
    Format = ".png"
if(testName == "faceocc1"):
    zero = "1"
# meas_variance = np.random.randn(4, 4)*0.1
meas_variance = np.array([[0.05, 0, 0, 0],
                          [0, 0.05, 0, 0],
                          [0, 0, 0.01, 0],
                          [0, 0, 0, 0.01]])
# 135,67,91,97
# initial_state = np.array([200, 113, 0, 0, 45, 45, 0, 0]).T  # ball
# initial_state = np.array([135, 67, 0, 0, 91, 97, 0, 0]).T  # girl
# initial_state = np.array([282, 167, 0, 0, 525, 684, 0, 0]).T # mytest1
# initial_state = np.array([324, 127, 0, 0, 560, 665, 0, 0]).T  # mytest2
# initial_state = np.array([388, 148, 0, 0, 544, 674, 0, 0])
# initial_state = np.array([113, 74, 0, 0, 82, 86, 0, 0])
initial_state = np.array([128, 145, 0, 0, 122, 119, 0, 0])  # dudek-face
print(128, 145, 122, 119)
accel_variance = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0.2, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0.2, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0.2, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0.2]])*0.01
kf = KF(initial_state=initial_state,
        accel_variance=accel_variance, lam=1, gamma=1)

# [real_X, real_Y, real_W, real_H] = initial_state[position]

DT = 0.5
NUM_STEPS = 480
# MEAS_EVERY_STEPS = 20

im = Image.open("./"+testName+"/"+imgName+zero + "00" + "5"+Format)
prevImg = cv2.imread("./"+testName+"/"+imgName+zero + "00" + "5"+Format)
init_Img = prevImg.copy()
prevMask = np.zeros((prevImg.shape[0], prevImg.shape[1])).astype(np.uint8)
init_Mask = prevMask
x = initial_state[0]
y = initial_state[1]
w = initial_state[4]
h = initial_state[5]
prev_w = w
prev_h = h
prev_x = x
prev_y = y
prevMask[y:y+h, x:x+w] = 1
stdkp, stddes = getSIFT(prevImg, prevMask, 0)
prev_desc = stddes

for step in range(NUM_STEPS):
    picNum = step + 1
    ############### Read a new image ################
    img = cv2.imread("./"+testName+"/" + imgName + zero + str(picNum//100) +
                     str((picNum % 100)//10) + str(picNum % 10) + Format)
    originImg = img.copy()

    # Get measurements based on the previous state
    mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    mask[prev_y:prev_y+prev_h, prev_x:prev_x+prev_w] = 1
    list_kp1_1, list_kp2_1, goodDes2_1 = getMatch(
        init_Img, originImg, 2, init_Mask, mask)

    ############### Make prediction #############

    kf.predict(dt=DT)
    # newMask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)

    # predicted position
    pos = kf.pos.astype(np.int)
    x = pos[0, 0]
    y = pos[1, 0]
    w = pos[2, 0]
    h = pos[3, 0]
    vel = kf.vel.astype(np.int)

    cv2.rectangle(img, (x, y),
                  (x+w, y+h), (0, 0, 255), 2)
    text = "predicted Vx:" + str(vel[0, 0]) + " Vy:" + str(vel[1, 0])
    # cv2.putText(img, text, (100, 200),
    #             cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 5)
    cv2.putText(img, text, (50, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    ############## Measurements ############

    # mask: based on the prediction
    mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    mask[y:y+h, x:x+w] = 1

    list_kp1_2, list_kp2_2, goodDes2_2 = getMatch(
        init_Img, originImg, 2, init_Mask, mask)

    # use mask to get match
    # list_kp1_old, list_kp2_old = getMatch(prevImg, img, 2, prevMask, prevMask)

    # predict the center of the image based on the predicted mask
    # centerX = np.sum(np.array([item[0] for item in list_kp2]))/len(list_kp2)
    # centerY = np.sum(np.array([item[1] for item in list_kp2]))/len(list_kp2)

    # mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    # maskX = (centerX-w/2).astype(np.int)
    # maskY = (centerY-h/2).astype(np.int)
    # mask[maskY-10:maskY+h+10, maskX-10:maskX+w+10] = 1

    # list_kp1, list_kp2 = getMatch(prevImg, img, 2, prevMask, mask, picNum)
    for point in list_kp2_2:
        point_t = (point[0], point[1])
        cv2.circle(img, point_t, 5, (0, 0, 255), 0)
    # cv2.imwrite("./images/"+str(picNum) + "_matched.jpg", img)
    # cv.circle(img, (160, 160), 60, point_color, 0)

    ############# Select the measurements to use ###############

    if(len(list_kp2_2) > 0 and len(list_kp2_2) > 0.7 * len(list_kp2_1)):

        # new_x = np.min(np.array([item[0]
        #                          for item in list_kp2_2])).astype(np.int) - 5
        # new_y = np.min(np.array([item[1]
        #                          for item in list_kp2_2])).astype(np.int) - 5

        centerX = np.sum(np.array([item[0]
                                   for item in list_kp2_2]))/len(list_kp2_2)
        centerY = np.sum(np.array([item[1]
                                   for item in list_kp2_2]))/len(list_kp2_2)

    # mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
        new_x = (centerX-w/2).astype(np.int)
        new_y = (centerY-h/2).astype(np.int)

        # new_w = np.max(np.array([item[0] for item in list_kp2_2])).astype(
        #     np.int) - new_x + 10
        # new_h = np.max(np.array([item[1] for item in list_kp2_2])).astype(
        #     np.int) - new_y + 10

        # new_w = prev_w
        # new_h = prev_h
        # print("prev_desc 1", type(prev_desc[0, 0]))

        # list_kp_old, list_kp_new, junk = getMatch(
        #     prevImg, originImg, 2, prevMask, mask)
        # list_kp_old = np.array(list_kp_old)
        # list_kp_new = np.array(list_kp_new)

        # relative_x = (list_kp2_2[:, 0]-prev_x)/prev_w
        # relative_y = (list_kp2_2[:, 1]-prev_y)/prev_h
        # relative_x = (list_kp_old[:, 0]-prev_x)/prev_w
        # relative_y = (list_kp_old[:, 1]-prev_y)/prev_h
        # relative = np.column_stack((relative_x, relative_y))
        # new_x, new_y, new_w, new_h = LeastSquare(list_kp_new, relative)

        # print("relx_out", relative_x)
        # print("rely_out", relative_y)
        # print("kp1", list_kp1_2)
        # print("kp2", list_kp2_2)
        new_x = int(new_x)
        new_y = int(new_y)
        new_w = prev_w
        new_h = prev_h
        prev_desc = np.array(goodDes2_2).astype(np.uint8)
        # print("fitted", new_x, new_y, new_w, new_h)

    else:
        vel = kf.vel.astype(np.int)
        # print("vel x", vel[0, 0])
        # new_x = int(prev_x + vel[0, 0]*DT)
        # new_y = int(prev_y+vel[1, 0]*DT)
        new_x = prev_x
        new_y = prev_y
        new_w = prev_w
        new_h = prev_h
        prev_desc = goodDes2_1

    # new_w = np.max(np.array([item[0] for item in list_kp2])).astype(
    #     np.int) - new_x + 10
    # new_h = np.max(np.array([item[1] for item in list_kp2])).astype(
    #     np.int) - new_y + 10

    cv2.rectangle(img, (new_x, new_y),
                  (new_x+new_w, new_y+new_h), (0, 255, 0), 2)

    ########### Situation 1: x, y change,  w and h do not change ############
    # use the center and previous w/h to get the new_x, new_y
    # new_x = centerX - prev_w / 2
    # np.min(np.array([item[1] for item in new_kp]))-5
    # new_y = centerY - prev_h / 2

    ########### Situation 2: x, y do not change, w and h do change ############
    # get new_w, new_h based on the sift with the mask of the prediction
    # new_w = np.max(np.array([item[0] for item in list_kp2])) - \
    # np.min(np.array([item[0] for item in list_kp2])) + 10
    # new_h = np.max(np.array([item[1] for item in list_kp2])) - \
    # np.min(np.array([item[0] for item in list_kp2])) + 10

    # To see which situation is true:
    # newMask1: based on the updated x, y
    # newMask2: based on the updated x, y, w, h
    # newMask1[new_y: new_y+h: new_x:new_x+w] = 1
    # newMask2[new_y: new_y+new_h: new_x:new_x+new_w] = 1
    # list_kp1_new1, list_kp2_new1 = getMatch(
    #     prevImage, img, 2, prevMask, newMask1)
    # list_kp1_new2, list_kp2_new2 = getMatch(
    #     prevImage, img, 2, prevMask, newMask2)
    # selectedMask = newMask2

    prev_w = new_w
    prev_h = new_h
    prev_x = new_x
    prev_y = new_y
    kf.update(meas_value=np.array([new_x, new_y, new_w, new_h]).T,  # real_x + np.random.randn() * np.sqrt(meas_variance),
              meas_variance=meas_variance)
    # get updated position
    pos = kf.pos.astype(np.int)
    x = pos[0, 0]
    y = pos[1, 0]
    w = pos[2, 0]
    h = pos[3, 0]
    prevMask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    prevMask[y:y+h, x:x+w] = 1
    ########### Save the image to the file #########
    vel = kf.vel.astype(np.int)
    text = "updated Vx:" + str(vel[0, 0]) + " Vy:" + str(vel[1, 0])
    # cv2.putText(img, text, (100, 100),
    #             cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 0, 0), 5)
    cv2.putText(img, text, (50, 100),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(img, (x, y),
                  (x+w, y+h), (255, 0, 0), 2)

    print(x, y, w, h)
    ########### Update the previous image ##########
    prevImg = originImg
    imgPath = "./images/"+str(picNum)+".jpg"
    cv2.imwrite(imgPath, img)
    im2 = Image.open(imgPath)
    frames.append(im2)
im.save("Gif_test_"+testName + "2.gif",
        save_all=True, append_images=frames)
