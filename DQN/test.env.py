import gym
import numpy as np

from DQNenv import ITSEnv

data = np.load('D:\RLITS\data\pems04_01.npy')
env = ITSEnv(data)
s, r, d, _ =env.step(1)
print(s)
print(r)
print(d)