import numpy as np

# offsets of each variable in the state vector
[iX, iY, iVx, iVy, iW, iH, iVw, iVh] = range(8)
position = [0, 1, 4, 5]
velocity = [2, 3, 6, 7]
NUMVARS = iVh + 1


class KF:
    def __init__(self, initial_state, accel_variance, gamma, lam
                 ) -> None:
        # mean of state GRV
        '''
        Usage: 
        initial_state: a 8*1 numpy matrix [x,y,vx,vy,w,h,vw,vh]
        accel_variance: a 8*1 numpy matrix specifying the variance of each term above
        '''

        self._x = initial_state.reshape(-1, 1)
        self._accel_variance = accel_variance
        self.lam = lam
        self.gamma = gamma
        # covariance of state GRV
        self._P = np.eye(NUMVARS)
        # print("P is initialized as", self._P)

    def predict(self, dt: float) -> None:
        '''
        Predict the next state
        '''
        # x = F x
        # P = F P Ft + G Gt a
        # print("enter predict")
        F = np.eye(NUMVARS)
        F[iX, iVx] = dt
        F[iY, iVy] = dt
        F[iW, iVw] = dt
        F[iH, iVh] = dt
        # print(F)
        new_x = F.dot(self._x)

        G = np.zeros((NUMVARS, 1))
        G[position, 0] = 0.5 * dt**2
        G[velocity, 0] = dt
        # + G.dot(self._accel_variance).dot(G.T)
        # print("g shape", G.shape)
        # print("accel shape", self._accel_variance.shape)
        # print(G.dot(self._accel_variance).dot(G.T).shape)
        # print("P is", self._P)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T).dot(self._accel_variance)
        # print(F.dot(self._P))
        self._P = new_P
        self._x = new_x

    def update(self, meas_value, meas_variance):
        '''
        Using the measurement to adjust the prediction
        Usage: meas_value: a 8*1 numpy matrix of the position and size 
               feature [currently only considering single point]
        '''
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P
        # print("enter update")
        H = np.zeros((4, NUMVARS))

        H[0, position[0]] = 1
        H[1, position[1]] = 1
        H[2, position[2]] = 1
        H[3, position[3]] = 1

        z = np.array(meas_value)
        R = np.array(meas_variance)

        y = z.reshape(-1, 1) - H.dot(self._x)
        # print("zshape", z.shape)
        # print("HX shape", H.dot(self._x).shape)
        y = y.reshape(-1, 1)
        S = H.dot(self._P).dot(H.T) + R
        # print(y)
        # print(R)
        # print(self._P)
        # print("h shape", H.shape)
        # print("P shape", self._P.shape)
        # print("y shape", y.shape)
        # print("s shape", S.shape)
        # print(S)
        # K = self._P.dot(H.T).dot(np.linalg.inv(S))
        # print("k is", K)

        K = np.array([[0.8, 0, 0, 0],
                      [0, 0.8, 0, 0],
                      [0.2, 0, 0, 0],
                      [0, 0.2, 0, 0],
                      [0, 0, 0.8, 0],
                      [0, 0, 0, 0.8],
                      [0, 0, 0.2, 0],
                      [0, 0, 0, 0.2]])
        new_x = self._x + K.dot(y)
        # print(K.dot(y))
        # print("ky shape", K.dot(y).shape)
        # print("self._x. shape", self._x.shape)
        # print("new_x. shape", new_x.shape)
        new_P = (np.eye(NUMVARS) - K.dot(H)).dot(self._P)
        # V = new_x[velocity, 0]
        # delta = np.abs(self._x - new_x)
        # deltaV = delta[velocity, 0]
        # Q = self.lam**2 * deltaV**2 / (1+self.gamma*V**2)
        # print(Q)
        # self._accel_variance[velocity, velocity] = Q
        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self):
        return self._x[position]

    @property
    def vel(self) -> float:
        return self._x[velocity]
