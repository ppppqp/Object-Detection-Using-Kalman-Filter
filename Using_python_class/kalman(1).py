import numpy as np
import cv2


class Kalman(object):
    def __init__(self, processNoiseCovariance=0.00001, measurementNoiseCovariance=0.00001, errorCovariancePost=0.01):#0.1 1
        self.kalman = cv2.KalmanFilter(4, 2, 0)

        self.kalman_measurement = np.zeros((2, 1), np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.array([[0, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * processNoiseCovariance
        self.kalman.measurementNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * measurementNoiseCovariance
        self.kalman.errorCovPost = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * errorCovariancePost

        self.predicted = None
        self.corrected = None

    def update(self, x, y):
        self.kalman_measurement[0][0] = x
        self.kalman_measurement[1][0] = y
        self.predicted = self.kalman.predict()
        self.corrected = self.kalman.correct(self.kalman_measurement)

    def getCorrection(self):
        return self.corrected[0][0], self.corrected[1][0]

    def getPrediction(self):
        return self.predicted[0][0], self.predicted[1][0]
