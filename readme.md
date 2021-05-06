# Visual-inertial Simultaneous localization and mapping (SLAM) Based on Extended Kalman Filter

In this project, our main object is to build a visual-inertial simultaneous localization and mapping (SLAM) on a moving robot, and based on Extended Kalman Filter (EKF). There is a IMU on the robot to obtain the movement information, and a stereo camera to extract the feature and then obtain the localization information. And after implementing EKF on visual-inertial SLAM, we get a reasonable result.

## Getting Started

Above all else, we need to run DataProprocess.py firstly, this will extract the data from ,npz file 

"DataProprocess.py": extract all the data from .npz file and store them via .pkl file, initial all the global variable

"IMUc.py" : contains the IMU class, run it will execute the IMU-based Localization via EKF Prediction part

"VisualMapping.py",  run it will execute the Landmark Mapping via EKF Update part. This requires about 40min to run

"main.py", run it will execute the Visual-Inertial SLAM part. This requires about 50min to run

"pose.pkl/landmarks.pkl" The results from "main.py"

"code/utils":  contains many basic functions

