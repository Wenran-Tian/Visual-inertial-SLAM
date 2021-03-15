import os
import pickle
from DataPreprocess import *
from IMUc import *
import matplotlib.pyplot as plt
from scipy import linalg
import time

class visualMapping:
    def __init__(self,M, Tio):
        self.M = M
        self.Toi = invTrans(Tio)

    def Observation(self, ms, Tt, noise = None):
        """
            ms : 4*n homogeneous coordinate
        """
        if noise is None:
            # noise = 2 is quite enough
            noise = 0
        z = np.zeros(ms.shape)
        v = np.round(np.random.normal(0, noise, ms.shape))
        z = self.M@pai(self.Toi@invTrans(Tt)@ms) + v
        z[z < 0] = 0
        return z


if __name__ == "__main__":
    # t, features, linear_velocity, angular_velocity, Kc, b, imu_T_cam = load_data("code/data/10.npz", load_features=True)
    file = open(IMU_DATA_PATH, 'rb')
    [linear_velocity, angular_velocity] = pickle.load(file)
    file.close()
    file = open(TIME_DATA_PATH, 'rb')
    time_series = pickle.load(file)
    file.close()
    time_series = time_series[0]
    time_series = time_series - time_series[0]
    tau = time_series[-1]/ time_series.shape[0]
    gv = np.row_stack((linear_velocity, angular_velocity))
    IMU = IMUc(gv, tau)
    IMU.cal_T_series(noise=[0.01, 0.01])
    IMU.cal_T_series()
    T_series = IMU.T_series

    file = open(FEATURE_TEST_DATA_PATH, 'rb')
    features = pickle.load(file)
    file.close()

    it_number = features.shape[2]-1
    # it_number = 100
    # LANDMARKs = np.zeros((4, M))  # landmarks in homo coordinate
    # SIGMA = (np.eye(3 * M)).astype(np.float32)*0.01


    for i in range(it_number):
        print("iteration: " + str(i) + "/" + str(it_number))
        feature = features[:, :, i]
        feature_tp1 = features[:, :, i+1]
        # extract the valid features in one observation
        # validation = feature[0] > 0  # all the indices, valid = True, invalid = False
        # print(feature[:, validation].shape)

        vallist = np.where(feature[0] > 0)
        vallist = vallist[0]

        vallist1 = np.where(feature_tp1[0] > 0)
        vallist1 = vallist1[0]
        val_t_list = []
        # update or initialize the landmark
        Trans = T_OI@invTrans(T_series[:,:,i+1])
        Trans1 = T_series[:, :, i+1]@T_IO

        for j in vallist:
            if LANDMARKs[3, j] == 0:
                LANDMARKs[:, j] = Trans1@stereo2Homo(K, B, feature[:, j])
            else:
                val_t_list.append(j)

        # continue

        if(len(val_t_list) == 0):
            continue
        # we will only deal with mu and sigma in a range
        il = min(val_t_list)  # smallest index
        ir = max(val_t_list)  # largest index
        range = ir - il + 1

        V = np.eye(4)  # standard deviation
        z_minus = np.zeros(4*range)
        H_tp1 = np.zeros((4*range,3*range))

        for j in val_t_list:
            j_in_H = j - il
            if j in vallist1:
                z_tilde = KM@pai(Trans@LANDMARKs[:, j])
                z_minus[4*(j-il): 4*(j-il)+4] = feature_tp1[:, j] - z_tilde
            H_tp1[4 * j_in_H: 4 * j_in_H + 4, 3 * j_in_H: 3 * j_in_H + 3] = KM @ dpai_dq(Trans @ LANDMARKs[:, j]) @ Trans @ P.T

        Sigma = SIGMA[3*il: 3*il+3*range, 3*il: 3*il+3*range]

        K_tp1 = Sigma@H_tp1.T@np.linalg.inv(H_tp1@Sigma@H_tp1.T + np.kron(np.eye(range), V))
        LANDMARKs[0:3, il: il+range] += (K_tp1@(z_minus)).reshape((range, 3)).T
        midvalue = K_tp1@H_tp1
        Sigma = (np.eye(Sigma.shape[0]) - midvalue)@Sigma

    plt.scatter(LANDMARKs[0, :], LANDMARKs[1, :], s=1)
    plt.show()

    # assert(1==2)