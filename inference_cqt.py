import random
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import efficientnet.tfkeras as efn
import math
import scipy
from scipy import signal
from scipy.signal import get_window
from GroupNormalization import GroupNormalization
import warnings
from typing import Optional, Tuple

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Use Local config
class CFG:
    # *******************************************************************************************
    # CQT Parameters
    sample_rate = 2048.0
    fmin = 20.0
    fmax = 512.0
    hop_length = 16
    bins_per_octave = 24
    whiten = False
    bandpass = True
    ts = 0.1
    length = 4096
    tukey = tf.cast(scipy.signal.windows.get_window(('tukey', ts), length), tf.float32)
    use_tukey = True
    # *******************************************************************************************
    # Test Parameters
    fold = [0]
    k_fold = len(fold)
    batch_size = 64
    HEIGHT = 256
    WIDTH = 256
    SEED = 1234
    use_tta = True
    TTA_STEP = 4
    from_local = True
    # *******************************************************************************************
    # OOF Inference Result Folder
    result_folder = "./"


AUTOTUNE = tf.data.experimental.AUTOTUNE
TEST_DATA_ROOT = "./TFRecords/BandPass/test_tfrecords" if CFG.bandpass else "./TFRecords/No BandPass/test_tfrecords"
test_img_lists = os.listdir(TEST_DATA_ROOT)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(CFG.SEED)
tfrecs = tf.io.gfile.glob(TEST_DATA_ROOT + "/*.tfrecords")
raw_image_dataset = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)

# Create a dictionary describing the features.
image_feature_description = {
    'id': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string),
}


def _parse_sample_function(sample):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(sample, image_feature_description)

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
def _cqt_test(image, id):
    return cqt(image), id



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


@tf.function
def _parse_raw_function(sample):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(sample, image_feature_description)


@tf.function
def _decode_raw_test(sample):
    data = tf.io.decode_raw(sample['data'], tf.float32)
    data = tf.reshape(data, (3, 4096))
    if CFG.use_tta:
        # Shuffle Channel
        indice = tf.range(len(data))
        indice = tf.random.shuffle(indice)
        data = tf.gather(data, indice, axis=0)
    if CFG.whiten:
        data = whiten(data)
    return data, sample['id']


@tf.function
def _aug_test(image, id):
    image = MinMaxScaler(tf.image.resize(image, (CFG.HEIGHT, CFG.WIDTH)), 0.0, 1.0)
    image = tf.image.per_image_standardization(image)
    return image, id


opt = tf.data.Options()
opt.experimental_deterministic = False


def create_test_dataset():
    global raw_image_dataset
    dataset = (raw_image_dataset
               .map(_parse_sample_function, num_parallel_calls=AUTOTUNE)
               .with_options(opt)
               .map(_decode_raw_test, num_parallel_calls=AUTOTUNE)
               .batch(CFG.batch_size, num_parallel_calls=AUTOTUNE)
               .map(_cqt_test, num_parallel_calls=AUTOTUNE)
               .map(_aug_test, num_parallel_calls=AUTOTUNE)
               .prefetch(AUTOTUNE))
    return dataset


def create_model():
    backbone = efn.EfficientNetB0(
        include_top=False,
        input_shape=(CFG.HEIGHT, CFG.WIDTH, 3),
        weights=None,
        pooling='avg'
    )
    if not CFG.from_local:
        model = tf.keras.Sequential([
            backbone,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    else:
        model = tf.keras.Sequential([
            backbone,
            GroupNormalization(group=32),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu'),
            GroupNormalization(group=32),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    return model


model = create_model()
tdataset = create_test_dataset()


def inference(count, path):
    global tdataset
    model.load_weights(path + "/model_best_%d.h5" % CFG.fold[count])
    rec_ids = []
    probs = []
    for data, name in tqdm(tdataset):
        pred = model.predict_on_batch(data)
        rec_ids.append(name.numpy())
        probs.append(pred.reshape(pred.shape[0]))
    crec_ids = np.concatenate(rec_ids)
    cprobs = np.concatenate(probs)
    sub_with_prob = pd.DataFrame({
        'id': list(map(lambda x: x.decode(), crec_ids.tolist())),
        'target': cprobs
    })
    sub_with_prob.describe()
    return sub_with_prob


# sub_with_prob = inference(0, "./model/0907-CWT-TPU/0.8657").set_index('id').reset_index()
# sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob_0.csv"), index=False)

# k-Fold TTA Sample
if CFG.use_tta:
    sub_with_prob = sum(
        map(
            lambda j:
            inference(math.floor(j / CFG.TTA_STEP), "./model").set_index('id')
            / (CFG.k_fold * CFG.TTA_STEP), range(CFG.k_fold * CFG.TTA_STEP)
        )
    ).reset_index()
    sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob.csv"), index=False)
else:
    sub_with_prob = sum(
        map(
            lambda j:
            inference(j, "./model").set_index('id') / CFG.k_fold, range(CFG.k_fold)
        )
    ).reset_index()
    sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob.csv"), index=False)
