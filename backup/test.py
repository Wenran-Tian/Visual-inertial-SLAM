import numpy as np
from scipy import linalg
from utils import *
from DataPreprocess import *

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
# feature = features[ :, :,300 ]
# valid = np.where(feature[0] > 0)
# valid = valid[0]
# r = np.linspace(0, 11, 12)
# print(r.reshape(4, 3).T)
# plt.scatter(r,r, color = 'purple', s = 1)
V = np.eye(4)*0.1  # standard deviation
res = np.kron(np.eye(2), V)
print(res)
# plt.show()