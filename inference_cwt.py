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
from scipy.signal import butter, sosfiltfilt
from GroupNormalization import GroupNormalization

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Use Local config
class CFG:
    # *******************************************************************************************
    # CWT Parameters
    sample_rate = 2048.0
    fmin = 20.0
    fmax = 512.0
    nv = 32
    whiten = False
    bandpass = False
    trainable = False
    ts = 0.1
    length = 4096
    tukey = tf.cast(scipy.signal.windows.get_window(('tukey', ts), length), tf.float32)
    use_tukey = True
    # *******************************************************************************************
    # Test Parameters
    fold = [0]
    k_fold = len(fold)
    # if batch_size sets to 64, CWT will OOM
    batch_size = 32
    HEIGHT = 256
    WIDTH = 256
    SEED = 2022
    use_tta = False
    TTA_STEP = 4
    from_local = True
    # *******************************************************************************************
    # OOF Inference Result Folder
    result_folder = "./"


AUTOTUNE = tf.data.experimental.AUTOTUNE
TEST_DATA_ROOT = "./test_tfrecords"
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
def _cwt_test(image, id):
    return MinMaxScaler(tf.image.resize(cwt(image), (CFG.HEIGHT, CFG.WIDTH)), 0.0, 1.0), id


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


@tf.function
def _parse_raw_function(sample):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(sample, image_feature_description)


@tf.function
def _decode_raw_test(sample):
    data = tf.io.decode_raw(sample['data'], tf.float32)
    data = tf.reshape(data, (3, 4096))
    if CFG.bandpass:
        tf_bp_filter(data)
    if CFG.whiten:
        data = whiten(data)
    return data, sample['id']


@tf.function
def _aug_test(image, id):
    image = tf.image.per_image_standardization(image)
    if CFG.use_tta:
        image = tf.image.random_contrast(image, lower=1.0, upper=1.3)
        image = tf.cond(tf.random.uniform([]) < 0.5,
                        lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
                        lambda: tf.image.random_hue(image, max_delta=0.3))
        image = tf.image.random_brightness(image, 0.3)
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
               .map(_cwt_test, num_parallel_calls=AUTOTUNE)
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


# sub_with_prob = inference(0, "./model/0827-CWT/0.9143").set_index('id').reset_index()
# sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob_0.csv"), index=False)

# k-Fold TTA Sample
if CFG.use_tta:
    sub_with_prob = sum(
        map(
            lambda j:
            inference(math.floor(j / CFG.TTA_STEP), "./model/0827-CWT/0.9143").set_index('id')
            / (CFG.k_fold * CFG.TTA_STEP), range(CFG.k_fold * CFG.TTA_STEP)
        )
    ).reset_index()
    sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob.csv"), index=False)
else:
    sub_with_prob = sum(
        map(
            lambda j:
            inference(j, "./model/0827-CWT/0.9143").set_index('id') / CFG.k_fold, range(CFG.k_fold)
        )
    ).reset_index()
    sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob.csv"), index=False)
