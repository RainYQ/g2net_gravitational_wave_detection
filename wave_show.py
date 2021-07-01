import os
import numpy as np
import matplotlib.pyplot as plt

wave = np.load('./train/0/0/0/00000e74ad.npy')
print(wave)
plt.figure()
plt.plot(wave[0])
plt.figure()
plt.plot(wave[1])
plt.figure()
plt.plot(wave[2])
plt.show()
