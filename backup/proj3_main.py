import numpy as np
from utils import *


if __name__ == '__main__':

	# only choose ONE of the following data 
	    
	# data 1. this data has features, use this if you plan to skip the extra credit feature detection and tracking part 
	filename = "code/data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)

	# data 2. this data does NOT have features, you need to do feature detection and tracking but will receive extra credit 
	#filename = "./data/03.npz"
	#t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	
	# (a) IMU Localization via EKF Prediction
	print(linear_velocity.shape)
	print(angular_velocity.shape)

	# (b) Feature detection and matching
	# (c) Landmark Mapping via EKF Update

	# (d) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)

