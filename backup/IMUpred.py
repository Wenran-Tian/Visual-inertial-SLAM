import os
import pickle
from DataPreprocess import *
import matplotlib.pyplot as plt
from scipy import linalg

class IMUpred:
    def __init__(self, gv_series, tau):
        '''
            gv_series: generalized velocity matrix with size 6*n
        '''
        self.gv_series = gv_series
        self.tau = tau
        self.length = gv_series.shape[1]
        # self.cal_T_series()

    def cal_T_series(self, noise = None):
        if noise is None:
            noise = [0, 0]
        self.T_series = np.zeros((4,4,self.length + 1))
        self.T_series[:,:,0] = np.eye(4)
        for i in range(self.length):
            # T_inst = linalg.expm(self.tau*(self.gv2hat(self.gv_series[:, i])))
            # self.T_series[:,:,i+1] = self.T_series[:,:,i]@T_inst

            self.T_series[:, :, i + 1] = self.predict_inst(self.T_series[:,:,i], self.gv_series[:, i], noise)

    def gv2hat(self, gv):
        res = np.zeros((4,4))
        res[0:3, 0:3] = self.skew(gv[3:6])
        res[0:3, 3] = gv[0:3]
        return res

    def predict_inst(self, Tf, gv, noise=None):
        """ predict the transformation instantaneously
        gv: 6*1 vector
        """
        if noise is None:
            noise = [0, 0]
        # print(np.random.normal(loc=0.0, scale=noise[0], size=(3, 1)).shape)
        gv[0:3] += np.random.normal(loc=0.0, scale=noise[0], size=(3))
        gv[3:6] += np.random.normal(loc=0.0, scale=noise[1], size=(3))
        T_inst = linalg.expm(self.tau * (self.gv2hat(gv)))
        return Tf@T_inst

    @staticmethod
    def skew(x):
        """x must be a 3*3 matrix"""
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])


if __name__=="__main__":

    file = open(IMU_DATA_PATH, 'rb')
    [linear_velocity, angular_velocity] = pickle.load(file)
    file.close()

    file = open(TIME_DATA_PATH, 'rb')
    time_series = pickle.load(file)
    file.close()

    time_series = time_series[0]
    time_series = time_series - time_series[0]
    tau = time_series[-1]/ time_series.shape[0]

    print(tau)
    gv = np.row_stack((linear_velocity, angular_velocity))
    IMU = IMUpred(gv, tau)
    # IMU.cal_T_series(noise=[0.01, 0.01])
    IMU.cal_T_series()
    T_series = IMU.T_series


    visualize_trajectory_2d(T_series, show_ori=True)
    # print(T_series.shape)
    # plt.plot(T_series[:,0,3], T_series[:,1,3])
    # # plt.plot(linear_velocity[0])
    # plt.show()


