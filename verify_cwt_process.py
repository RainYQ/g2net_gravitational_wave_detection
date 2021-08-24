import pickle
import random
import os
import math
import numpy as np
import tensorflow as tf
import scipy
from scipy import signal
from scipy.signal import butter, sosfiltfilt
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CFG:
    sample_rate = 2048.0
    fmin = 20.0
    fmax = 512.0
    nv = 32
    whiten = True
    bandpass = True
    trainable = False
    ts = 0.1
    len = 4096
    tukey = tf.cast(scipy.signal.windows.get_window(('tukey', ts), len), tf.float32)
    use_tukey = True
    tfrecords_fold_prefix = "./"
    HEIGHT = 256
    WIDTH = 256
    SEED = 2022
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
    return wft, padvalue, num_scales


wft, padvalue, num_scales = cwt_pre((1, CFG.len),
                                    nv=CFG.nv,
                                    sr=CFG.sample_rate,
                                    flow=CFG.fmin,
                                    fhigh=CFG.fmax,
                                    trainable=CFG.trainable)


@tf.function
def kron(a, b):
    return tf.numpy_function(np.kron, [a, b], tf.complex64)


@tf.function
# Change to function
def cwt(input, flow=20.0, fhigh=512.0, batch_size=None):
    assert fhigh > flow, 'fhigh parameters must be > flow!'
    assert batch_size is not None, 'batch size must be set!'
    assert len(input.shape) == 2, 'Input dimension must be 2! Dimension is {}'.format(len(input.shape))
    # wft, padvalue, num_scales = cwt_pre(input.shape, nv, sr, flow, fhigh, trainable)
    max_loop = tf.shape(input)[0]

    @tf.function
    def sum_cwt(i, pre_data):
        next_data = tf.nn.embedding_lookup(input, i)
        x = tf.concat([tf.reverse(next_data[0:padvalue], axis=[0]), next_data,
                       tf.reverse(next_data[-padvalue:], axis=[0])], 0)
        f = tf.signal.fft(tf.cast(x, tf.complex64))
        cwtcfs = tf.signal.ifft(
            kron(tf.ones([num_scales, 1], dtype=tf.complex64), f) * tf.cast(wft, tf.complex64))
        logcwt = tf.math.log(tf.math.abs(cwtcfs[:, padvalue:padvalue + next_data.shape[-1]]))
        pre_data = tf.tensor_scatter_nd_add(pre_data, indices=[[i]], updates=[logcwt])
        i_next = i + 1
        return i_next, pre_data

    _, cwt = tf.while_loop(cond=lambda i, result: tf.less(i, max_loop),
                           body=sum_cwt,
                           loop_vars=(tf.constant(0, dtype=tf.int32),
                                      tf.zeros([batch_size, num_scales, input.shape[-1]],
                                               dtype=tf.float32)))
    return MinMaxScaler(tf.squeeze(tf.image.resize(tf.expand_dims(tf.squeeze(cwt), -1), (CFG.HEIGHT, CFG.WIDTH))), 0.0, 1.0)


@tf.function
def MinMaxScaler(data, lower, upper):
    assert upper > lower, 'fhigh parameters must be > flow!'
    lower = tf.cast(lower, tf.float32)
    upper = tf.cast(upper, tf.float32)
    min_val = tf.math.reduce_min(data)
    max_val = tf.math.reduce_max(data)
    std_data = tf.divide(tf.subtract(data, min_val), tf.subtract(max_val, min_val))
    return tf.add(tf.multiply(std_data, tf.subtract(upper, lower)), lower)


@tf.function
def whiten(signal):
    if CFG.use_tukey:
        window = CFG.tukey
    else:
        window = tf.signal.hann_window(signal.shape[1], periodic=True)
    spec = tf.signal.fft(tf.cast(signal[0, :] * window, tf.complex128))
    mag = tf.math.sqrt(tf.math.real(spec * tf.math.conj(spec)))
    channel_0 = tf.cast(tf.math.real(tf.signal.ifft(spec / tf.cast(mag, tf.complex128))), tf.float32) * tf.math.sqrt(
        signal.shape[1] / 2)
    spec = tf.signal.fft(tf.cast(signal[1, :] * window, tf.complex128))
    mag = tf.math.sqrt(tf.math.real(spec * tf.math.conj(spec)))
    channel_1 = tf.cast(tf.math.real(tf.signal.ifft(spec / tf.cast(mag, tf.complex128))), tf.float32) * tf.math.sqrt(
        signal.shape[1] / 2)
    spec = tf.signal.fft(tf.cast(signal[2, :] * window, tf.complex128))
    mag = tf.math.sqrt(tf.math.real(spec * tf.math.conj(spec)))
    channel_2 = tf.cast(tf.math.real(tf.signal.ifft(spec / tf.cast(mag, tf.complex128))), tf.float32) * tf.math.sqrt(
        signal.shape[1] / 2)
    return tf.stack([channel_0, channel_1, channel_2], axis=0)


def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data):
    filter_sos = butter_bandpass(CFG.fmin, CFG.fmax, CFG.sample_rate, order=8)
    y = sosfiltfilt(filter_sos, data, padlen=1024)
    return y


@tf.function
def tukey_window(data):
    window = CFG.tukey
    return data * window


# wrap numpy-based function for use with TF
@tf.function
def tf_bp_filter(input):
    input = tf.cast(input, tf.float64)
    y = tf.numpy_function(butter_bandpass_filter, [input], tf.float64)
    return y


seed_everything(CFG.SEED)
tfrecs = tf.io.gfile.glob(TRAIN_DATA_ROOT + "/*.tfrecords")
raw_image_dataset = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)

# Create a dictionary describing the features.
image_feature_description = {
    'id': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_raw_function(sample):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(sample, image_feature_description)


def _preprocess_image_function(single_photo):
    data = tf.io.decode_raw(single_photo['data'], tf.float32)
    data = tf.reshape(data, (3, 4096))
    if CFG.bandpass:
        tf_bp_filter(data)
    if CFG.whiten:
        data = whiten(data)
    channel_r = cwt(tf.expand_dims(data[0, :], axis=0), flow=CFG.fmin, fhigh=CFG.fmax, batch_size=1)
    channel_g = cwt(tf.expand_dims(data[1, :], axis=0), flow=CFG.fmin, fhigh=CFG.fmax, batch_size=1)
    channel_b = cwt(tf.expand_dims(data[2, :], axis=0), flow=CFG.fmin, fhigh=CFG.fmax, batch_size=1)
    image = tf.stack([channel_r, channel_g, channel_b], axis=-1)
    # image = tf.image.per_image_standardization(image)
    single_photo['data'] = image
    return single_photo['data'], tf.cast(single_photo['label'], tf.float32), single_photo['id']


def create_idx_filter(indice):
    def _filt(i, single_photo):
        return tf.reduce_any(indice == i)

    return _filt


def _remove_idx(i, single_photo):
    return single_photo


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
               .shuffle(10240)
               .with_options(opt)
               .repeat()
               .map(_preprocess_image_function, num_parallel_calls=AUTOTUNE))
    return dataset

idx_train_tf = tf.cast(tf.constant(splits[0][0]), tf.int64)
dataset = create_train_dataset(idx_train_tf)
for image, label, sample_id in dataset.take(1):
    plt.figure()
    plt.pcolormesh(np.flip(image.numpy()[:, :, 0], 0))
    plt.figure()
    plt.pcolormesh(np.flip(image.numpy()[:, :, 1], 0))
    plt.figure()
    plt.pcolormesh(np.flip(image.numpy()[:, :, 2], 0))
    print(label.numpy())
    print(sample_id.numpy())
plt.show()

