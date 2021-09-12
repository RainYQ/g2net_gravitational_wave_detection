import pickle
import random
import tensorflow_addons as tfa
from scipy import signal
import warnings
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from scipy.signal import get_window
import scipy
import os
from matplotlib import pyplot as plt

# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CFG:
    # *******************************************************************************************
    # CQT Parameters
    sample_rate = 2048.0
    fmin = 20.0
    fmax = 512.0
    hop_length = 16
    bins_per_octave = 24
    whiten = False
    whiten_use_tukey = True
    bandpass = True
    ts = 0.1
    length = 4096
    tukey = tf.cast(scipy.signal.windows.get_window(('tukey', ts), length), tf.float32)
    # *******************************************************************************************
    # tfrecords folder location
    tfrecords_fold_prefix = "./TFRecords/BandPass" if bandpass else "./TFRecords/No BandPass"
    # *******************************************************************************************
    # split folder location
    split_data_location = "./data_local"
    # *******************************************************************************************
    # Train Parameters
    SEED = 2020
    HEIGHT = 256
    WIDTH = 256
    batch_size = 16
    # *******************************************************************************************
    # Augmentation
    use_shuffle_channel = True
    Use_Gaussian_Noise = True
    mixup = False
    label_smooth = True
    ls = 0.99
    T_SHIFT = False
    S_SHIFT = False
    # *******************************************************************************************
    # Normalization Style
    signal_use_channel_std_mean = True
    mean = tf.reshape(tf.cast([5.36416325e-27, 1.21596245e-25, 2.37073866e-27], tf.float64), (3, 1))
    std = tf.reshape(tf.cast(np.sqrt(np.array([5.50707291e-41, 5.50458798e-41, 3.37861660e-42], dtype=np.float64)),
                             tf.float64), (3, 1))
    # 'channel' or 'global' or None
    image_norm_type = None
    # *******************************************************************************************
    # Set to True for first Run
    generate_split_data = False


AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_ROOT = os.path.join(CFG.tfrecords_fold_prefix, "train_tfrecords")
TEST_DATA_ROOT = os.path.join(CFG.tfrecords_fold_prefix, "test_tfrecords")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_cqt_kernels(
        q: float,
        fs: float,
        fmin: float,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        norm: float = 1,
        window: str = "hann",
        fmax: Optional[float] = None,
        topbin_check: bool = True
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    fft_len = 2 ** _nextpow2(np.ceil(q * fs / fmin))

    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    else:
        warnings.warn("If nmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check:
        raise ValueError(f"The top bin {np.max(freqs)} Hz has exceeded the Nyquist frequency, \
                           please reduce the `n_bins`")

    kernel = np.zeros((int(n_bins), int(fft_len)), dtype=np.complex64)

    length = np.ceil(q * fs / freqs)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = np.ceil(q * fs / freq)

        if l % 2 == 1:
            start = int(np.ceil(fft_len / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fft_len / 2.0 - l / 2.0))

        sig = get_window(window, int(l), fftbins=True) * np.exp(
            np.r_[-l // 2:l // 2] * 1j * 2 * np.pi * freq / fs) / l

        if norm:
            kernel[k, start:start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            kernel[k, start:start + int(l)] = sig
    return kernel, fft_len, length, freqs


def _nextpow2(a: float) -> int:
    return int(np.ceil(np.log2(a)))


def prepare_cqt_kernel(
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        bins_per_octave=12,
        norm=1,
        filter_scale=1,
        window="hann"
):
    q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
    print(q)
    return create_cqt_kernels(q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)


cqt_kernels, KERNEL_WIDTH, lengths, _ = prepare_cqt_kernel(
    sr=int(CFG.sample_rate),
    hop_length=CFG.hop_length,
    fmin=CFG.fmin,
    fmax=CFG.fmax,
    bins_per_octave=CFG.bins_per_octave)
LENGTHS = tf.constant(lengths, dtype=tf.float32)
CQT_KERNELS_REAL = tf.constant(np.swapaxes(cqt_kernels.real[:, np.newaxis, :], 0, 2))
CQT_KERNELS_IMAG = tf.constant(np.swapaxes(cqt_kernels.imag[:, np.newaxis, :], 0, 2))
PADDING = tf.constant([[0, 0],
                       [0, 0],
                       [KERNEL_WIDTH // 2, KERNEL_WIDTH // 2],
                       [0, 0]])


@tf.function
def cqt(wave, hop_length=16):
    x = tf.expand_dims(wave, 3)
    x = tf.pad(x, PADDING, "REFLECT")
    CQT_real = tf.nn.conv1d(x, CQT_KERNELS_REAL, stride=hop_length, padding="VALID")
    CQT_imag = -tf.nn.conv1d(x, CQT_KERNELS_IMAG, stride=hop_length, padding="VALID")
    CQT_real *= tf.math.sqrt(LENGTHS)
    CQT_imag *= tf.math.sqrt(LENGTHS)
    CQT = tf.math.sqrt(tf.pow(CQT_real, 2) + tf.pow(CQT_imag, 2))
    return tf.transpose(CQT, (0, 2, 3, 1))


@tf.function
def _cqt(image, label, id):
    return cqt(image, CFG.hop_length), label, id


@tf.function
def MinMaxScaler(data, lower, upper, mode):
    lower = tf.cast(lower, tf.float32)
    upper = tf.cast(upper, tf.float32)
    if mode == 'channel':
        min_val = tf.reshape(tf.reduce_min(data, axis=[1, 2]), [tf.shape(data)[0], 1, 1, tf.shape(data)[-1]])
        max_val = tf.reshape(tf.reduce_max(data, axis=[1, 2]), [tf.shape(data)[0], 1, 1, tf.shape(data)[-1]])
        std_data = tf.divide(tf.subtract(data, min_val), tf.subtract(max_val, min_val))
    elif mode == 'global':
        lower = tf.cast(lower, tf.float32)
        upper = tf.cast(upper, tf.float32)
        min_val = tf.reshape(tf.reduce_min(data, axis=[1, 2, 3]), [tf.shape(data)[0], 1, 1, 1])
        max_val = tf.reshape(tf.reduce_max(data, axis=[1, 2, 3]), [tf.shape(data)[0], 1, 1, 1])
        std_data = tf.divide(tf.subtract(data, min_val), tf.subtract(max_val, min_val))
    else:
        return data
    return tf.add(tf.multiply(std_data, tf.subtract(upper, lower)), lower)


@tf.function
def whiten(signal):
    if CFG.whiten_use_tukey:
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
    data = tf.io.decode_raw(sample['data'], tf.float64)
    data = tf.reshape(data, (3, 4096))
    if CFG.signal_use_channel_std_mean:
        data = (data - CFG.mean) / CFG.std
    else:
        data /= tf.reshape(tf.reduce_max(data, axis=1), (3, 1))
    data = tf.cast(data, tf.float32)
    if CFG.use_shuffle_channel:
        # Shuffle Channel
        indice = tf.range(len(data))
        indice = tf.random.shuffle(indice)
        data = tf.gather(data, indice, axis=0)
    if CFG.whiten:
        data = whiten(data)
    label = tf.cast(sample['label'], tf.float32)
    return data, label, sample['id']


@tf.function
def time_shift(img):
    T = CFG.WIDTH
    P = tf.random.uniform([], 0, 1)
    SHIFT = tf.cast(T * P, tf.int32)
    return tf.concat([img[-SHIFT:], img[:-SHIFT]], axis=0)


@tf.function
def spector_shift(img):
    T = CFG.HEIGHT
    P = tf.random.uniform([], 0, 1)
    SHIFT = tf.cast(T * P, tf.int32)
    return tf.concat([img[:, -SHIFT:], img[:, :-SHIFT]], axis=1)


@tf.function
def _aug(image, label, id):
    image = tf.image.resize(image, (CFG.HEIGHT, CFG.WIDTH))
    image = MinMaxScaler(image, 0.0, 1.0, CFG.image_norm_type)
    # image = tf.image.per_image_standardization(image)
    # if CFG.T_SHIFT:
    #     image = tf.map_fn(time_shift, image, dtype=tf.float32)
    # if CFG.S_SHIFT:
    #     image = tf.map_fn(spector_shift, image, dtype=tf.float32)
    # if CFG.Use_Gaussian_Noise:
    #     # 高斯噪声的标准差为 0.1
    #     gau = tf.keras.layers.GaussianNoise(0.1)
    #     # 以 50％ 的概率为图像添加高斯噪声
    #     image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    # image = tfa.image.random_cutout(image, [20, 20])
    # image = tfa.image.random_cutout(image, [20, 20])
    # image = tfa.image.random_cutout(image, [20, 20])
    # image = tfa.image.random_cutout(image, [20, 20])
    return image, label, id


def create_idx_filter(indice):
    @tf.function
    def _filt(i, sample):
        return tf.reduce_any(indice == i)

    return _filt


def _remove_idx(i, sample):
    return sample


preprocess_dataset = (raw_image_dataset.map(_parse_raw_function, num_parallel_calls=AUTOTUNE)
                      .enumerate())
with open(os.path.join(CFG.split_data_location, "splits.data"), 'rb') as file:
    splits = pickle.load(file)
print("DataSet Split Successful.")

opt = tf.data.Options()
opt.experimental_deterministic = False


def create_train_dataset(batchsize, train_idx):
    global preprocess_dataset
    parsed_train = (preprocess_dataset
                    .filter(create_idx_filter(train_idx))
                    .map(_remove_idx))

    dataset = (parsed_train
               .with_options(opt)
               .repeat()
               .map(_decode_raw, num_parallel_calls=AUTOTUNE)
               .batch(batchsize, num_parallel_calls=AUTOTUNE)
               .map(_cqt, num_parallel_calls=AUTOTUNE)
               .map(_aug, num_parallel_calls=AUTOTUNE)
               .unbatch()
               .prefetch(AUTOTUNE))
    return dataset


idx_train_tf = tf.cast(tf.constant(splits[0][0]), tf.int64)
dataset = create_train_dataset(CFG.batch_size, idx_train_tf)
for image, label, sample_id in dataset.take(2):
    plt.figure()
    plt.pcolormesh(image.numpy()[:, :, 0].T)
    plt.figure()
    plt.pcolormesh(image.numpy()[:, :, 1].T)
    plt.figure()
    plt.pcolormesh(image.numpy()[:, :, 2].T)
    print(label.numpy())
    print(sample_id.numpy())
plt.show()
