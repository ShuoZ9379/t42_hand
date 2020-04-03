#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import random

with open('/media/pracsys/DATA/hand_images_data/rollout_blue_2.pkl', 'rb') as f:
    Timages, Simages, images, S, A, T = pickle.load(f)
Simages = np.array(Simages)

print len(images), len(Timages)

# plt.plot(Simages[:,0],Simages[:,1])
# plt.show()

i = 0
for I, t, s in zip(images, Timages, Simages):
    print t, s
    i += 1
    if i % 10:
        continue

    plt.imshow(I)
    plt.show()