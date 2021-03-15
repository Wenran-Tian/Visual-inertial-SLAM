import numpy as np
from utils import *
import os
import pickle

IMU_DATA_PATH = "IMUdata.pkl"
TIME_DATA_PATH = "Timedata.pkl"
FEATURE_TEST_DATA_PATH = "Featuredata.pkl"
Numbers = 3026
M = 13289  # number of landmarks

B = 0.6
K = np.array([[552.554261,   0.,       682.049453],[  0.,   552.554261, 238.769549],[0. ,    0.,     1.    ]])
# orginal K
Ko = np.array([[552.554261,   0.,       682.049453],[  0.,   552.554261, 238.769549],[0. ,    0.,     1.    ]])
KM = np.zeros((4, 4))
KM[0:2, 0:3] = K[0:2, 0:3]
KM[2:4, 0:3] = K[0:2, 0:3]
KM[2,3] = -K[0,0]*B
# transformation from imu to optical
T_IO = np.array([[ 0.03717833, -0.09861822,  0.9944306,   1.5752681 ], [ 0.99926755, -0.00535534, -0.03789026,  0.00439141],
 [ 0.00906218,  0.99511094,  0.09834688, -0.65      ], [ 0.,          0. ,         0.,          1.,        ]])
T_OI = invTrans(T_IO)

P = np.eye(3, 4)

LANDMARKs = (np.zeros((4, M)))  # landmarks in homo coordinate .astype(np.float32)
SIGMA = np.eye(3*M, dtype = np.float16)  # initial sigma for landmarks

SIGMA_Pos = np.eye(6)*0.001  # initial sigma for position


if __name__ =="__main__":
    filename = "code/data/10.npz"
    t, features, linear_velocity, angular_velocity, Kc, b, imu_T_cam = load_data(filename, load_features=True)

    # features shape : (4, 13289, 3026)


    file = open(IMU_DATA_PATH, 'wb')
    pickle.dump( [linear_velocity, angular_velocity], file)
    file.close()

    file = open(TIME_DATA_PATH, 'wb')
    pickle.dump(t, file)
    file.close()

    file = open(FEATURE_TEST_DATA_PATH, 'wb')
    pickle.dump(features[:, :, 0:Numbers], file)
    file.close()

    K = Kc
    B = b
    T_ic = imu_T_cam
    for it in [K,B,T_ic]:
        print(it)

    # file = open("SIGMA", 'wb')
    # pickle.dump(SIGMA, file)
    # file.close()
