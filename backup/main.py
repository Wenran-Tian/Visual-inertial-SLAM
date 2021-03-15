import os
import pickle
from DataPreprocess import *
from IMUpred import *
import matplotlib.pyplot as plt
from scipy import linalg


""" part (a) """
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
IMU = IMUpred(gv, tau)
IMU.cal_T_series(noise=[0.001, 0.001])
T_series = IMU.T_series

# visualize_trajectory_2d(T_series, show_ori=True)

""" part (c) """

