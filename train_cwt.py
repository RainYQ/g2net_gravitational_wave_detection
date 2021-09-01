import pickle
import random
import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import efficientnet.tfkeras as efn
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K
from GroupNormalization import GroupNormalization
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import scipy
from scipy import signal

# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

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
    # split folder location
    split_data_location = "./data_local"
    # *******************************************************************************************
    # Train Parameters
    SEED = 1234
    HEIGHT = 256
    WIDTH = 256
    batch_size = 16
    epoch = 20
    iteration_per_epoch = 28000
    learning_rate = 1e-4
    k_fold = 5
    # *******************************************************************************************
    # Augmentation
    mixup = True
    # *******************************************************************************************
    # Tensorboard
    tensorboard = False
    # *******************************************************************************************
    # Set to True for first Run
    generate_split_data = True
    # *******************************************************************************************
    # Test Code
    use_small_dataset = False


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
def _cwt(image, label):
    return MinMaxScaler(tf.image.resize(cwt(image), (CFG.HEIGHT, CFG.WIDTH)), 0.0, 1.0), label


@tf.function
def _cwt_val_extra(image, id):
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


@tf.function
def tukey_window(data):
    window = CFG.tukey
    return data * window


seed_everything(CFG.SEED)
if CFG.use_small_dataset:
    tfrecs = tf.io.gfile.glob(TRAIN_DATA_ROOT + "/*.tfrecords")[0]
else:
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
    return data, tf.cast(sample['label'], tf.float32)


@tf.function
def _decode_raw_val_extra(sample):
    data = tf.io.decode_raw(sample['data'], tf.float32)
    data = tf.reshape(data, (3, 4096))
    if CFG.whiten:
        data = whiten(data)
    return data, sample['id']


@tf.function
def _aug(image, label):
    image = tf.image.per_image_standardization(image)
    # 高斯噪声的标准差为 0.3
    gau = tf.keras.layers.GaussianNoise(0.2)
    # 以 50％ 的概率为图像添加高斯噪声
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    # image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    # image = tf.cond(tf.random.uniform([]) < 0.5,
    #                 lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
    #                 lambda: tf.image.random_hue(image, max_delta=0.3))
    # # brightness随机调整
    # image = tf.image.random_brightness(image, 0.3)
    image = tfa.image.random_cutout(image, [20, 20])
    image = tfa.image.random_cutout(image, [20, 20])
    image = tfa.image.random_cutout(image, [20, 20])
    image = tfa.image.random_cutout(image, [20, 20])
    return image, label


@tf.function
def _aug_val(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


@tf.function
def _aug_val_extra(image, id):
    image = tf.image.per_image_standardization(image)
    return image, id


def create_idx_filter(indice):
    def _filt(i, sample):
        return tf.reduce_any(indice == i)

    return _filt


def _remove_idx(i, sample):
    return sample


@tf.function
def _mixup(data, targ):
    indice = tf.range(len(data))
    indice = tf.random.shuffle(indice)
    sinp = tf.gather(data, indice, axis=0)
    starg = tf.gather(targ, indice, axis=0)
    alpha = 0.2
    t = tfp.distributions.Beta(alpha, alpha).sample([len(data)])
    tx = tf.reshape(t, [-1, 1, 1, 1])
    ty = t
    x = data * tx + sinp * (1 - tx)
    y = targ * ty + starg * (1 - ty)
    return x, y


indices = []
id = []
label = []
preprocess_dataset = (raw_image_dataset.map(_parse_raw_function, num_parallel_calls=AUTOTUNE)
                      .enumerate())
if CFG.generate_split_data:
    for i, sample in tqdm(preprocess_dataset):
        indices.append(i.numpy())
        label.append(sample['label'].numpy())
        id.append(sample['id'].numpy().decode())

    table = pd.DataFrame({'indices': indices, 'id': id, 'label': label})
    skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
    X = np.array(table.index)
    Y = np.array(list(table.label.values), dtype=np.uint8)
    splits = list(skf.split(X, Y))
    with open(os.path.join(CFG.split_data_location, "splits.data"), 'wb') as file:
        pickle.dump(splits, file)
    print("origin: ", np.sum(np.array(list(table["label"].values), dtype=np.uint8), axis=0))
    for j in range(5):
        print("Train Fold", j, ":", np.sum(np.array(list(table["label"][splits[j][0]].values), dtype=np.uint8), axis=0))
        print("Val Fold", j, ":", np.sum(np.array(list(table["label"][splits[j][1]].values), dtype=np.uint8), axis=0))
    for j in range(5):
        with open(os.path.join(CFG.split_data_location, "k-fold_" + str(j) + ".txt"), 'w') as writer:
            writer.write("Train:\n")
            indic_str = "\n".join([str(l) for l in list(splits[j][0])])
            writer.write(indic_str)
            writer.write("\n")
            writer.write("Val:\n")
            indic_str = "\n".join([str(l) for l in list(splits[j][1])])
            writer.write(indic_str)
        writer.close()
else:
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
               # .cache()
               # .shuffle(len(train_idx))
               .shuffle(10240)
               .with_options(opt)
               .repeat()
               .map(_decode_raw, num_parallel_calls=AUTOTUNE)
               .batch(batchsize, num_parallel_calls=AUTOTUNE)
               .map(_cwt, num_parallel_calls=AUTOTUNE)
               .map(_aug, num_parallel_calls=AUTOTUNE))
    if CFG.mixup:
        dataset = (dataset.map(_mixup, num_parallel_calls=AUTOTUNE)
                   .prefetch(AUTOTUNE))
    else:
        dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def create_val_dataset(batchsize, val_idx):
    global preprocess_dataset
    parsed_val = (preprocess_dataset
                  .filter(create_idx_filter(val_idx))
                  .map(_remove_idx))
    dataset = (parsed_val
               # .cache()
               .map(_decode_raw, num_parallel_calls=AUTOTUNE)
               .batch(batchsize, num_parallel_calls=AUTOTUNE)
               .map(_cwt, num_parallel_calls=AUTOTUNE)
               .map(_aug_val, num_parallel_calls=AUTOTUNE))
    return dataset


def create_val_extra_dataset(batchsize, val_idx):
    global preprocess_dataset
    parsed_val = (preprocess_dataset
                  .filter(create_idx_filter(val_idx))
                  .map(_remove_idx))
    dataset = (parsed_val
               # .cache()
               .map(_decode_raw_val_extra, num_parallel_calls=AUTOTUNE)
               .batch(batchsize, num_parallel_calls=AUTOTUNE)
               .map(_cwt_val_extra, num_parallel_calls=AUTOTUNE)
               .map(_aug_val_extra, num_parallel_calls=AUTOTUNE))
    return dataset


def create_model():
    backbone = efn.EfficientNetB0(
        include_top=False,
        input_shape=(CFG.HEIGHT, CFG.WIDTH, 3),
        weights='noisy-student',
        pooling='avg'
    )

    model = tf.keras.Sequential([
        backbone,
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu'),
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    # optimizer = tf.keras.optimizers.Adam(CFG.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=CFG.learning_rate,
                                             total_steps=CFG.epoch * CFG.iteration_per_epoch,
                                             warmup_proportion=0.1, min_lr=1e-6)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', num_thresholds=498)])
    return model


def plot_history(history, name):
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("loss")
    plt.subplot(1, 4, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("metric")
    plt.subplot(1, 4, 3)
    plt.plot(history.history["auc"])
    plt.plot(history.history["val_auc"])
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title("metric")
    plt.savefig(name)


model = create_model()


# Run Inference On Val Dataset.
# Save as "./submission_with_prob_val_i.csv"
def inference(count, path):
    idx_val_tf = tf.cast(tf.constant(splits[count][1]), tf.int64)
    v_test_dataset = create_val_extra_dataset(CFG.batch_size * 2, idx_val_tf)
    model.load_weights(path + "/model_best_%d.h5" % count)
    rec_ids = []
    probs = []
    for data, name in tqdm(v_test_dataset):
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
    sub_with_prob.to_csv(path + "/submission_with_prob_" + str(count) + ".csv", index=False)
    return sub_with_prob


def train(splits, split_id):
    print("batchsize", CFG.batch_size)
    model = create_model()
    idx_train_tf = tf.cast(tf.constant(splits[split_id][0]), tf.int64)
    idx_val_tf = tf.cast(tf.constant(splits[split_id][1]), tf.int64)
    # 生成训练集和验证集
    dataset = create_train_dataset(CFG.batch_size, idx_train_tf)
    vdataset = create_val_dataset(CFG.batch_size, idx_val_tf)
    if CFG.tensorboard:
        log_dir = "logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=2)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./model/model_best_%d.h5' % split_id,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True),
            tensorboard_callback
        ]
    else:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./model/model_best_%d.h5' % split_id,
                save_weights_only=True,
                monitor='val_auc',
                mode='max',
                save_best_only=True)
        ]
    history = model.fit(dataset,
                        batch_size=CFG.batch_size,
                        steps_per_epoch=CFG.iteration_per_epoch,
                        epochs=CFG.epoch,
                        validation_data=vdataset,
                        callbacks=callbacks
                        )

    plot_history(history, 'history_%d.png' % split_id)
    with open("history_%d.data" % split_id, 'wb') as file:
        pickle.dump(history.history, file)

for i in range(CFG.k_fold):
    train(splits, i)
    inference(i, "./model")
