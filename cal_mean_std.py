import numpy as np
from welford import Welford
import glob
from tqdm import tqdm
import joblib
import math

w1 = Welford()
w2 = Welford()
w3 = Welford()
train_files = glob.glob('./train/*/*/*/*.npy')


def cal_std_mean(train_files):
    for sample in tqdm(train_files):
        data = np.load(sample)
        for i in range(3):
            w1.add_all(np.expand_dims(data[0, :], axis=1))
            w2.add_all(np.expand_dims(data[1, :], axis=1))
            w3.add_all(np.expand_dims(data[2, :], axis=1))


_ = joblib.Parallel(n_jobs=16)(
    joblib.delayed(cal_std_mean)
    (train_files[n * 4096:min((n + 1) * 4096, len(train_files))],)
    for n in range(math.ceil(len(train_files) / 4096))
)
print(w1.mean, w1.var_s, w1.var_p)
print(w2.mean, w2.var_s, w2.var_p)
print(w3.mean, w3.var_s, w3.var_p)