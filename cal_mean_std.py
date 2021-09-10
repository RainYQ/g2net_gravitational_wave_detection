import numpy as np
from welford import Welford
import glob
from tqdm import tqdm

w1 = Welford()
w2 = Welford()
w3 = Welford()
train_files = glob.glob('./train/*/*/*/*.npy')

for sample in tqdm(train_files[:100]):
    data = np.load(sample)
    for i in range(3):
        w1.add(data[0, :])
        w2.add(data[1, :])
        w3.add(data[2, :])
print(w1.mean, w1.var_s, w1.var_p)
print(w2.mean, w2.var_s, w2.var_p)
print(w3.mean, w3.var_s, w3.var_p)
