import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os

from kalman import Kalman


def SIFT(imgname1="1.jpg", imgname2="2.jpg"):
    sift = cv2.xfeatures2d.SIFT_create()

    img1 = plt.imread(imgname1)
    kp1, des1 = sift.detectAndCompute(img1, None)
    img2 = plt.imread(imgname2)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    origin_point = []
    match_point = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            x1 = kp1[m.queryIdx].pt[0]
            y1 = kp1[m.queryIdx].pt[1]
            x2 = kp2[m.trainIdx].pt[0]
            y2 = kp2[m.trainIdx].pt[1]
            if 120 < x1 < 180 and 30 < y1 < 150 and 120 < x2 < 180 and 30 < y2 < 150:
                origin_point.append((x1, y1))
                match_point.append((x2, y2))
                good.append([m])
    img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img)
    plt.axis(False)
    plt.show()
    return origin_point, match_point


def SIFT_features(img, X, Y, W, H):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    features = []
    keypoints = []
    descriptions = []
    for idx in range(len(kp)):
        if X < kp[idx].pt[0] < X + W and Y < kp[idx].pt[1] < Y + H:
            features.append((kp[idx].pt[0], kp[idx].pt[1]))
            keypoints.append(kp[idx])
            descriptions.append(des[idx])
    return features, keypoints, np.array(descriptions)


def SIFT_matching(kp1, des1, img2, coef):
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    match_point = []
    kp1_match = []
    kp2_match = []
    for m, n in matches:
        if m.distance < coef * n.distance:
            kp1_match.append(kp1[m.queryIdx])
            kp2_match.append(kp2[m.trainIdx])
            match_point.append([m])
    return match_point, kp2, kp1_match, kp2_match


def Pixel2Relative(pixel, X, Y, W, H):
    relative = []
    for p in pixel:
        relative.append(((p[0] - X) / W, (p[1] - Y) / H))
    return relative


def LeastSquare(pixel, relative):
    N = len(pixel)
    pixel_arr = np.array(pixel)
    relative_arr = np.array(relative)
    pixel_x = np.expand_dims(pixel_arr[:, 0], axis=1)
    pixel_y = np.expand_dims(pixel_arr[:, 1], axis=1)
    relative_x = np.hstack((np.ones((N, 1)), np.expand_dims(relative_arr[:, 0], axis=1)))
    relative_y = np.hstack((np.ones((N, 1)), np.expand_dims(relative_arr[:, 1], axis=1)))
    X, W = np.squeeze(
        np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(relative_x), relative_x)), np.transpose(relative_x)),
                  pixel_x))
    Y, H = np.squeeze(
        np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(relative_y), relative_y)), np.transpose(relative_y)),
                  pixel_y))
    return X, Y, W, H


def get_directory(i, png=True, num_length=4, prefix="img"):
    string = "/" + prefix
    for num in range(num_length - len(str(i))):
        string = string + "0"
    if png:
        return string + str(i) + ".png"
    else:
        return string + str(i) + ".jpg"


if __name__ == '__main__':
    coef = 0.3
    X, Y, W, H = 113,74,82,86  # OccludedFace2: 113,74,82,86 # mydata: 435, 250, 410, 570 # Vid_A_ball: 200.34,113.73,45.13,45.59 # dudek-face: 128,145,122,119
    folder_name = "OccludedFace2"
    prefix = "img"
    png = True
    num_length = 5
    begin = 5
    end = 819
    resize_factor = 1.0

    kalman_XY = Kalman()
    kalman_WH = Kalman()
    gif = []
    output_path = ".\\" + folder_name + "_coef_" + str(coef)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    X, Y, W, H = resize_factor * X, resize_factor * Y, resize_factor * W, resize_factor * H

    for i in range(begin, end):
        img1 = plt.imread("./" + folder_name + get_directory(i, png, num_length, prefix))
        img2 = plt.imread("./" + folder_name + get_directory(i + 1, png, num_length, prefix))
        if png:
            img1 = (img1 * 255).astype("uint8")
            img2 = (img2 * 255).astype("uint8")
        img1 = cv2.resize(img1, (0, 0), fx=resize_factor, fy=resize_factor)
        img2 = cv2.resize(img2, (0, 0), fx=resize_factor, fy=resize_factor)

        if i == begin:
            output_img = cv2.rectangle(img1, (int(X), int(Y)), (int(X + W), int(Y + H)), (255, 0, 0), 2)
            gif.append(output_img)
            plt.imsave(output_path + "\\" + get_directory(i, png, num_length, prefix), output_img)
            print(X / resize_factor, ",", Y / resize_factor, ",", W / resize_factor, ",", H / resize_factor)


        features, kp1, des1 = SIFT_features(img1, X, Y, W, H)

        # plt.imshow(cv2.drawKeypoints(img1, keypoints, None))
        # plt.axis(False)
        # plt.show()
        # print(features)
        # print(descriptors)
        match_point, kp2, kp1_match, kp2_match = SIFT_matching(kp1, des1, img2, coef)

        # plt.axis(False)
        # plt.show()
        # print("i =", i, ",", len(kp1), len(kp2), len(kp1_match), len(kp2_match), X, Y, W, H)
        pixel1 = []
        pixel2 = []
        if len(kp1_match) > 0:
            for j in range(len(kp1_match)):
                pixel1.append(kp1_match[j].pt)
                pixel2.append(kp2_match[j].pt)
            relative = Pixel2Relative(pixel1, X, Y, W, H)
            X_update, Y_update, W_update, H_update = LeastSquare(pixel2, relative)
            if X_update > 0 and Y_update > 0 and W_update > 0 and H_update > 0 and X_update + W_update < np.shape(img2)[
                1] and Y_update + H_update < np.shape(img2)[0]:
                X, Y, W, H = X_update, Y_update, W_update, H_update
        kalman_XY.update(X, Y)
        kalman_WH.update(W, H)
        # print("i =", i)
        # print("\tX =", X, ", Y =", Y, ", W =", W, ", H =", H)
        # print("\tKalman_XY =", kalman_XY.getCorrection())
        # print("\tKalman_WH =", kalman_WH.getCorrection())
        if i > begin + 10:
            X, Y, W, H = kalman_XY.getCorrection()[0], kalman_XY.getCorrection()[1], kalman_WH.getCorrection()[0], \
                         kalman_WH.getCorrection()[1]

        output_img = cv2.rectangle(img2, (int(X), int(Y)), (int(X + W), int(Y + H)), (255, 0, 0), 2)
        print(X / resize_factor, ",", Y / resize_factor, ",", W / resize_factor, ",", H / resize_factor)
        gif.append(output_img)
        plt.imsave(output_path + "\\" + get_directory(i + 1, png, num_length, prefix), output_img)
        # if i > 63:
        #     print(match_point)
        #     # print("i =", i, ",", X, Y, W, H)
        #     plt.imshow(cv2.drawMatchesKnn(img1, kp1, img2, kp2, match_point, None, flags=2))
        #     # plt.imshow(cv2.drawKeypoints(img1, kp1_match, None))
        #     plt.axis(False)
        #     plt.show()
    imageio.mimsave(output_path + "\\gif_result.gif", gif, fps=50)
