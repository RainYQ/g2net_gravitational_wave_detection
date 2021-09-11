import numpy as np
import tensorflow_datasets
from welford import Welford
import glob
from tqdm import tqdm
import joblib
import math
import tensorflow as tf

use_welford = False
mode = 'test'
train_files = glob.glob('./{}/*/*/*/*.npy'.format(mode))

if use_welford:
    w1 = Welford()
    w2 = Welford()
    w3 = Welford()

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

else:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 1

    def _parse_function(sample):
        data = tf.io.read_file(sample)
        data = tf.strings.substr(data, 128, 98304)
        data = tf.reshape(tf.io.decode_raw(data, tf.float64), (3, 4096))
        return data

    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    train_ds = train_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)


    def tf_welford(ds):
        ds_numpy = tensorflow_datasets.as_numpy(ds)
        w_mean = np.zeros(3, dtype=np.float64)
        w_var = np.zeros(3, dtype=np.float64)
        sumsq = np.zeros(3, dtype=np.float64)
        cnt = 0.0
        for data in tqdm(ds_numpy):
            cnt += 1.0
            for j in range(3):
                x = data[0, j]
                delta = tf.math.reduce_mean(x - w_mean[j]).numpy()
                w_mean[j] += delta / cnt
                sumsq[j] += tf.math.reduce_sum(tf.math.multiply(x, x)).numpy()
                w_var[j] = (sumsq[j] / (cnt * 4096.)) - w_mean[j] * w_mean[j]
        return w_mean, w_var

    w1_mean, w1_var = tf_welford(train_ds)
    print(w1_mean, w1_var)
