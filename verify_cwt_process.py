import pickle
import random
import os
import math
import numpy as np
import tensorflow as tf
import scipy
from scipy import signal
from matplotlib import pyplot as plt
import tensorflow_addons as tfa

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CFG:
    # *******************************************************************************************
    # CWT Parameters
    sample_rate = 2048.0
    fmin = 20.0
    fmax = 512.0
    nv = 32
    whiten = False
    bandpass = True
    trainable = False
    ts = 0.1
    length = 4096
    tukey = tf.cast(scipy.signal.windows.get_window(('tukey', ts), length), tf.float32)
    use_tukey = True
    # *******************************************************************************************
    # tfrecords folder location
    tfrecords_fold_prefix = "./TFRecords/BandPass" if bandpass else "./TFRecords/No BandPass"
    # *******************************************************************************************
    # Dataset Parameters
    HEIGHT = 256
    WIDTH = 256
    SEED = 2022
    batch_size = 16
    check_aug = False
    # *******************************************************************************************
    # split folder location
    split_data_location = "./data_local"


AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_ROOT = os.path.join(CFG.tfrecords_fold_prefix, "train_tfrecords")
TEST_DATA_ROOT = os.path.join(CFG.tfrecords_fold_prefix, "test_tfrecords")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def cwt_pre(shape, nv=16, sr=2048.0, flow=20.0, fhigh=512.0, trainable=False):
    max_scale = shape[-1] // (np.sqrt(2) * 2)
    if max_scale <= 1:
        max_scale = shape[-1] // 2
    max_scale = np.floor(nv * np.log2(max_scale))
    scales = 2 * (2 ** (1 / nv)) ** np.arange(0, max_scale + 1)
    frequencies = sr * (6 / (2 * np.pi)) / scales
    frequencies = frequencies[frequencies >= flow]  # remove low frequencies
    scales = scales[0:len(frequencies)]
    frequencies = frequencies[frequencies <= fhigh]  # remove high frequencies
    scales = scales[len(scales) - len(frequencies):len(scales)]
    # wavft
    padvalue = shape[-1] // 2
    n = padvalue * 2 + shape[-1]
    omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
    omega = omega * (2 * np.pi) / n
    omega = np.concatenate((np.array([0]), omega, -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1]))
    _wft = np.zeros([scales.size, omega.size])
    for jj, scale in enumerate(scales):
        expnt = -(scale * omega - 6) ** 2 / 2 * (omega > 0)
        _wft[jj,] = 2 * np.exp(expnt) * (omega > 0)
    wft = tf.Variable(_wft, trainable=trainable)
    num_scales = scales.shape[-1]
    wft = tf.cast(wft, tf.complex64)
    return wft, padvalue, num_scales


wft, padvalue, num_scales = cwt_pre((3, CFG.length),
                                    nv=CFG.nv,
                                    sr=CFG.sample_rate,
                                    flow=CFG.fmin,
                                    fhigh=CFG.fmax,
                                    trainable=CFG.trainable)


@tf.function
# Change to function
def cwt(input):
    x = tf.concat([tf.reverse(input[:, :, 0:padvalue], axis=[2]), input,
                   tf.reverse(input[:, :, -padvalue:], axis=[2])], 2)
    x = tf.signal.fft(tf.cast(x, tf.complex64))
    cwtcfs = tf.signal.ifft(tf.expand_dims(x, 2) * wft)
    x = tf.math.log(tf.math.abs(cwtcfs[:, :, :, padvalue:padvalue + input.shape[-1]]))
    x = tf.transpose(x, (0, 2, 3, 1))
    return x


@tf.function
def _cwt(image, label, id):
    return MinMaxScaler(tf.image.resize(cwt(image), (CFG.HEIGHT, CFG.WIDTH)), 0.0, 1.0), label, id


@tf.function
def MinMaxScaler(data, lower, upper):
    lower = tf.cast(lower, tf.float32)
    upper = tf.cast(upper, tf.float32)
    min_val = tf.reshape(tf.reduce_min(data, axis=[1, 2]), [tf.shape(data)[0], 1, 1, tf.shape(data)[-1]])
    max_val = tf.reshape(tf.reduce_max(data, axis=[1, 2]), [tf.shape(data)[0], 1, 1, tf.shape(data)[-1]])
    std_data = tf.divide(tf.subtract(data, min_val), tf.subtract(max_val, min_val))
    return tf.add(tf.multiply(std_data, tf.subtract(upper, lower)), lower)


@tf.function
def whiten(signal):
    if CFG.use_tukey:
        window = CFG.tukey
    else:
        window = tf.signal.hann_window(signal.shape[-1], periodic=True)
    spec = tf.signal.fft(tf.cast(signal * window, tf.complex128))
    mag = tf.math.sqrt(tf.math.real(spec * tf.math.conj(spec)))
    return tf.cast(tf.math.real(tf.signal.ifft(spec / tf.cast(mag, tf.complex128))), tf.float32) * tf.math.sqrt(
        signal.shape[-1] / 2)


@tf.function
def tukey_window(data):
    window = CFG.tukey
    return data * window


seed_everything(CFG.SEED)
tfrecs = tf.io.gfile.glob(TRAIN_DATA_ROOT + "/*.tfrecords")
raw_image_dataset = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)

# Create a dictionary describing the features.
image_feature_description = {
    'id': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


@tf.function
def _parse_raw_function(sample):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(sample, image_feature_description)


@tf.function
def _decode_raw(sample):
    data = tf.io.decode_raw(sample['data'], tf.float32)
    data = tf.reshape(data, (3, 4096))
    if CFG.whiten:
        data = whiten(data)
    return data, tf.cast(sample['label'], tf.float32), sample['id']


@tf.function
def _aug(image, label, id):
    if CFG.check_aug:
        image = tf.image.per_image_standardization(image)
        image = tfa.image.random_cutout(image, [20, 20])
        image = tfa.image.random_cutout(image, [20, 20])
        image = tfa.image.random_cutout(image, [20, 20])
        image = tfa.image.random_cutout(image, [20, 20])
    return image, label, id


def create_idx_filter(indice):
    def _filt(i, sample):
        return tf.reduce_any(indice == i)

    return _filt


def _remove_idx(i, sample):
    return sample


indices = []
id = []
label = []
preprocess_dataset = (raw_image_dataset.map(_parse_raw_function, num_parallel_calls=AUTOTUNE)
                      .enumerate())
with open(os.path.join(CFG.split_data_location, "splits.data"), 'rb') as file:
    splits = pickle.load(file)
print("DataSet Split Successful.")

opt = tf.data.Options()
opt.experimental_deterministic = False


def create_train_dataset(train_idx):
    global preprocess_dataset
    parsed_train = (preprocess_dataset
                    .filter(create_idx_filter(train_idx))
                    .map(_remove_idx))
    dataset = (parsed_train
               .with_options(opt)
               .repeat()
               .map(_decode_raw, num_parallel_calls=AUTOTUNE)
               .batch(CFG.batch_size, num_parallel_calls=AUTOTUNE)
               .map(_cwt, num_parallel_calls=AUTOTUNE)
               .map(_aug, num_parallel_calls=AUTOTUNE)
               .unbatch())
    return dataset


idx_train_tf = tf.cast(tf.constant(splits[0][0]), tf.int64)
dataset = create_train_dataset(idx_train_tf)
for image, label, sample_id in dataset.take(2):
    plt.figure()
    plt.pcolormesh(np.flip(image.numpy()[:, :, 0], 0))
    plt.figure()
    plt.pcolormesh(np.flip(image.numpy()[:, :, 1], 0))
    plt.figure()
    plt.pcolormesh(np.flip(image.numpy()[:, :, 2], 0))
    print(label.numpy())
    print(sample_id.numpy())
plt.show()
