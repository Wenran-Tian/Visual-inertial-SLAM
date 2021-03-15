import numpy as np
from scipy import linalg
from utils import *
from DataPreprocess import *
import time

time0 = time.time()

x = np.array([0, 0, 1]) * 0.104
x_hat = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
# print(linalg.expm(x_hat))
# print(1/1e135)
# v = np.random.normal(loc=0.0, scale=0.01, size=(3, 1))
s = np.ones((3,5))
s[0,0] = -1
validation = s[0, :] >= 0
s[:, validation] += 1
s = np.random.randn(4, 2)

# res = np.eye(4)
# # res[0:3, 0:3] =
# res[0:3, 3] = np.array([1,-2,3])
# # print(np.round(np.random.normal(0, 1, (4, 6))))
# s = np.array([1,2,3,3,3,45,2,123,24,523,52,6])
# u = np.where(s>10)
# u = u[0]
# H_tp1 = np.zeros((4*2,3*2))
# # list = [1,2,3,4,52]
# h = H_tp1[0:3, 0:3]
# h[0,0] = 123

file = open(FEATURE_TEST_DATA_PATH, 'rb')
features = pickle.load(file)
file.close()

# it_number = features.shape[2]
#
feature = features[ :, :,100 ]
valid = np.where(feature[1] > 0)
valid = valid[0]
# r = np.linspace(0, 11, 12)
# print(r.reshape(4, 3).T)
# plt.scatter(r,r, color = 'purple', s = 1)
# V = np.eye(4)*0.1  # standard deviation
# res = np.linspace(1, 6, 6)
# time.sleep(4)
# # plt.show()
# time1 = time.time()

# l1 = [1,2,3,4,5]
# l2 = [1,2,3,6,7,6]
#
# M1 = np.array([1,2,3,4]).reshape((2,2))
# M2 = np.array([2,1,3,4]).reshape((2,2))
# M3 = np.array([4,2,3,1]).reshape((2,2))
# print(M1@M2.T@M3)
print(valid)


# z_tilde = KM@pai(Trans@LANDMARKs[:, j])
# tans = np.eye(4)
# tans[0:3, 3] = np.array([2, 3, 4])
# tans[0:3, 0:3] = np.array([    [0.8529,   -0.5000 ,   0.1504],
#     [0.5133  ,  0.7500  , -0.4172],
#     [0.0958  ,  0.4330  ,  0.8963]])
#
#
# # landmark = np.array([80, 50, 50, 1])
# # stereo = KM@pai(tans@landmark)
# # print(stereo)
# # homo = invTrans(tans)@stereo2Homo(Ko, B, stereo)
# # print(homo)
# r = np.array([2, 3, 4])
# plt.plot(r,r, color = 'purple')
# plt.scatter(tans[0], tans[1])
# plt.show()