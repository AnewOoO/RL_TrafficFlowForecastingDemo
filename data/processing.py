import time

import matplotlib.pyplot as plt
from numpy import load
import numpy as np
# data = load('pems04.npz')
# lst = data.files
# PeMS04_1 = np.array([[6.20e+01, 7.70e-03, 6.79e+01]])
# for key in range(1, 16992):
#     PeMS04_1 = np.insert(PeMS04_1,key,data['data'][key][0],0)
#     print(key)
# print(PeMS04_1)
# np.save('PeMS04_1',PeMS04_1)

# data = load('PeMS04_1.npy')
# PeMS04_1 = np.array([[6.20e+01,7.70e-03,6.79e+01,1.00e+00]])
# print("flow(0-100):1 flow(100-200):2 flow(200-300):3 flow(300-400):4 flow(400-626):52")
# for key in range(1,16992):
#     if data[key][0] < 100:
#         b = np.insert(data[key], 3, 0)
#         PeMS04_1 = np.insert(PeMS04_1,key,b,0)
#     elif data[key][0] < 200:
#         b = np.insert(data[key], 3, 1)
#         PeMS04_1 = np.insert(PeMS04_1,key,b,0)
#     elif data[key][0] < 300:
#         b = np.insert(data[key], 3, 2)
#         PeMS04_1 = np.insert(PeMS04_1, key, b, 0)
#     elif data[key][0] < 400:
#         b = np.insert(data[key], 3, 3)
#         PeMS04_1 = np.insert(PeMS04_1, key, b, 0)
#     else:
#         b = np.insert(data[key], 3, 4)
#         PeMS04_1 = np.insert(PeMS04_1, key, b, 0)
#     print(key)
# print(PeMS04_1)
# np.save('pems04_01',PeMS04_1)
# print("the max value of the whole flows", 626.0)
# print("the min value of the whole flows", 0.0)
#
# data = np.load('pems04_01.npy')
# for i in range(0, 16992):
#     print(data[i])

data = np.load('pems04_01.npy')
for idx in range(16992):
    print(data[idx])
