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


class CFG:
    tfrecords_fold_prefix = "./"
    batch_size = 16
    epoch = 80
    iteration_per_epoch = 28000
    learning_rate = 1e-4
    k_fold = 5
    HEIGHT = 256
    WIDTH = 256
    RAW_HEIGHT = 512
    RAW_WIDTH = 512
    SEED = 2022
    TRAIN_DATA_SIZE = 560000
    TEST_DATA_SIZE = 226000
    TTA_STEP = 16
    mixup = True
    tensorboard = False


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_ROOT = os.path.join(CFG.tfrecords_fold_prefix, "train_tfrecords")
TEST_DATA_ROOT = os.path.join(CFG.tfrecords_fold_prefix, "test_tfrecords")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(CFG.SEED)
tfrecs = tf.io.gfile.glob(TRAIN_DATA_ROOT + "/*.tfrecords")
raw_image_dataset = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)

# Create a dictionary describing the features.
image_feature_description = {
    'id': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_image_function(single_photo):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(single_photo, image_feature_description)


def _preprocess_image_function(single_photo):
    image = tf.image.decode_jpeg(single_photo['data'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if CFG.RAW_WIDTH != CFG.WIDTH or CFG.RAW_HEIGHT != CFG.HEIGHT:
        image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    image = tf.image.random_jpeg_quality(image, 90, 100)
    # 高斯噪声的标准差为 0.3
    gau = tf.keras.layers.GaussianNoise(0.3)
    # 以 50％ 的概率为图像添加高斯噪声
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.cond(tf.random.uniform([]) < 0.5,
                    lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
                    lambda: tf.image.random_hue(image, max_delta=0.3))
    # brightness随机调整
    image = tf.image.random_brightness(image, 0.3)
    single_photo['data'] = image
    return single_photo['data'], tf.cast(single_photo['label'], tf.float32)


def _preprocess_image_val_function(single_photo):
    image = tf.image.decode_jpeg(single_photo['data'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if CFG.RAW_WIDTH != CFG.WIDTH or CFG.RAW_HEIGHT != CFG.HEIGHT:
        image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    single_photo['data'] = image
    return single_photo['data'], tf.cast(single_photo['label'], tf.float32)


def _preprocess_image_val_extra_function(single_photo):
    image = tf.image.decode_jpeg(single_photo['data'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if CFG.RAW_WIDTH != CFG.WIDTH or CFG.RAW_HEIGHT != CFG.HEIGHT:
        image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    single_photo['data'] = image
    return single_photo['data'], single_photo['id']


def create_idx_filter(indice):
    def _filt(i, single_photo):
        return tf.reduce_any(indice == i)

    return _filt


def _remove_idx(i, single_photo):
    return single_photo


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
preprocess_dataset = (raw_image_dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
                      .enumerate())
for i, sample in tqdm(preprocess_dataset):
    indices.append(i.numpy())
    label.append(sample['label'].numpy())
    id.append(sample['id'].numpy().decode())

table = pd.DataFrame({'indices': indices, 'id': id, 'label': label})
skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
X = np.array(table.index)
Y = np.array(list(table.label.values), dtype=np.uint8).reshape(CFG.TRAIN_DATA_SIZE)
splits = list(skf.split(X, Y))
with open("splits.data", 'wb') as file:
    pickle.dump(splits, file)
# with open("splits.data", 'rb') as file:
#     splits = pickle.load(file)
print("DataSet Split Successful.")
print("origin: ", np.sum(np.array(list(table["label"].values), dtype=np.uint8), axis=0))
for j in range(5):
    print("Train Fold", j, ":", np.sum(np.array(list(table["label"][splits[j][0]].values), dtype=np.uint8), axis=0))
    print("Val Fold", j, ":", np.sum(np.array(list(table["label"][splits[j][1]].values), dtype=np.uint8), axis=0))
for j in range(5):
    with open("./k-fold_" + str(j) + ".txt", 'w') as writer:
        writer.write("Train:\n")
        indic_str = "\n".join([str(l) for l in list(splits[j][0])])
        writer.write(indic_str)
        writer.write("\n")
        writer.write("Val:\n")
        indic_str = "\n".join([str(l) for l in list(splits[j][1])])
        writer.write(indic_str)
    writer.close()

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
               .map(_preprocess_image_function, num_parallel_calls=AUTOTUNE)
               .batch(batchsize, num_parallel_calls=AUTOTUNE))
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
               .map(_preprocess_image_val_function, num_parallel_calls=AUTOTUNE)
               .batch(batchsize * 2, num_parallel_calls=AUTOTUNE))
    return dataset


def create_val_extra_dataset(batchsize, val_idx):
    global preprocess_dataset
    parsed_val = (preprocess_dataset
                  .filter(create_idx_filter(val_idx))
                  .map(_remove_idx))
    dataset = (parsed_val
               # .cache()
               .map(_preprocess_image_val_extra_function, num_parallel_calls=AUTOTUNE)
               .batch(batchsize * 2, num_parallel_calls=AUTOTUNE))
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
    optimizer = tfa.optimizers.RectifiedAdam(lr=CFG.learning_rate, total_steps=CFG.epoch * CFG.iteration_per_epoch,
                                             warmup_proportion=0.1, min_lr=1e-5)
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
        pred = model.predict_on_batch(tf.reshape(data, [-1, CFG.HEIGHT, CFG.WIDTH, 3]))
        rec_id_stack = tf.reshape(name, [-1, 1])
        for rec in name.numpy():
            assert len(np.unique(rec)) == 1
        rec_ids.append(rec_id_stack.numpy()[:, 0])
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
                monitor='val_loss',
                mode='min',
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


for i in range(CFG.k_fold):
    train(splits, i)
    inference(i, "./model")
