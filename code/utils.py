import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
import pickle

def load_data(file_name, load_features = False):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
        load_features: a boolean variable to indicate whether to load features 
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:

        t = data["time_stamps"] # time_stamps
        features = None 

        # only load features for 03.npz
        # 10.npz already contains feature tracks 
        if load_features:
            features = data["features"] # 4 x num_features : pixel coordinates of features
        
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam


def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of pose, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax
    

def pai(vec):
    """
    vec.shape = (4,)
    """
    return vec/vec[2]


def dpai_dq(vec):
    """
    vec.shape = (4,)
    """
    res = np.eye(4)
    col2 = -vec/vec[2]
    col2[2] = 0
    res[:,2] = col2
    return res/vec[2]


def stereo2Homo(K, b, vis):
    """
    vis : 4*n vector for ul vl and ur vr
    """
    res = np.ones(vis.shape)
    k11 = K[0, 0]
    k22 = K[1, 1]
    k13 = K[0, 2]
    k23 = K[1, 2]

    midval = vis[0] - vis[2] # ul-ur
    res[2] = k11*b/midval
    res[1] = res[2] * (vis[1]-k23)/k22
    res[0] = res[2] * (vis[0]-k13)/k11

    return res


def invTrans(T):
    """
     inverse only for transformation matrix
    """
    assert(T.shape == (4, 4))
    res = np.eye(4)
    res[0:3, 0:3] = T[0:3, 0:3].T
    res[0:3, 3] = - T[0:3, 0:3].T@T[0:3, 3]
    return res


def skew(x):
    """x must be a vector with length 3 """
    assert(x.shape[0] == 3)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def Odot(vec):
    """
     Odot operation, vec must be a vector with length=4
    """
    res = np.zeros((4, 6))
    res[0:3, 0:3] = np.eye(3)
    res[0:3, 3:6] = - skew(vec[0:3])
    return res

def fancyhat(x):
    """ x must be a vector with length 6 """
    res = np.kron(np.eye(2), skew(x[3:6]))
    res[0:3, 3:6] = skew(x[0:3])
    return res


def findUnion(list1, list2):
    return list(set(list1).intersection(set(list2)))


def saveData(data, name):
    file = open(name, 'wb')
    pickle.dump( data, file)
    file.close()


def loadData(name):
    file = open(name, 'wb')
    res = pickle.load(file)
    file.close()
    return res