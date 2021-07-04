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


class CFG:
    batch_size = 16
    epoch = 50
    iteration_per_epoch = 1024
    learning_rate = 1e-6
    k_fold = 5
    HEIGHT = 224
    WIDTH = 224
    RAW_HEIGHT = 27
    RAW_WIDTH = 128
    SEED = 2022


# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_ROOT = "./train_tfrecords"
TEST_DATA_ROOT = "./test_tfrecords"
train_img_lists = os.listdir(TRAIN_DATA_ROOT)
test_img_lists = os.listdir(TEST_DATA_ROOT)


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
    image = tf.io.decode_raw(sample['data'], tf.float32)
    image = tf.reshape(image, [CFG.RAW_HEIGHT, CFG.RAW_WIDTH])
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    image = (image - tf.reduce_min(image)) / (
            tf.reduce_max(image) - tf.reduce_min(image)) * 255.0
    image = tf.image.grayscale_to_rgb(image)
    # image = tf.image.random_jpeg_quality(image, 80, 100)
    # 高斯噪声的标准差为 0.3
    gau = tf.keras.layers.GaussianNoise(0.3)
    # # 以 50％ 的概率为图像添加高斯噪声
    # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    # image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    # image = tf.cond(tf.random.uniform([]) < 0.5,
    #                 lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
    #                 lambda: tf.image.random_hue(image, max_delta=0.3))
    # # brightness随机调整
    # image = tf.image.random_brightness(image, 0.3)
    single_photo['data'] = image
    return single_photo


def _preprocess_image_val_function(single_photo):
    image = tf.io.decode_raw(sample['data'], tf.float32)
    image = tf.reshape(image, [CFG.RAW_HEIGHT, CFG.RAW_WIDTH])
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    image = (image - tf.reduce_min(image)) / (
            tf.reduce_max(image) - tf.reduce_min(image)) * 255.0
    image = tf.image.grayscale_to_rgb(image)
    single_photo['data'] = image
    return single_photo


def create_idx_filter(indice):
    def _filt(i, single_photo):
        return tf.reduce_any(indice == i)

    return _filt


def _remove_idx(i, single_photo):
    return single_photo


def _create_annot(single_photo):
    return single_photo['data'], tf.one_hot(single_photo['label'], 1)


def _create_annot_val(single_photo):
    return single_photo['data'], tf.one_hot(single_photo['label'], 1)


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
Y = np.array(list(table.label.values), dtype=np.uint8).reshape(560000)
splits = list(skf.split(X, Y))
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


def create_train_dataset(batchsize, train_idx):
    global preprocess_dataset
    parsed_train = (preprocess_dataset
                    .filter(create_idx_filter(train_idx))
                    .map(_remove_idx))
    dataset = (parsed_train
               .cache()
               # .shuffle(len(train_idx))
               .shuffle(10240)
               .repeat()
               .map(_preprocess_image_function, num_parallel_calls=AUTOTUNE)
               .map(_create_annot, num_parallel_calls=AUTOTUNE)
               .batch(batchsize, num_parallel_calls=AUTOTUNE)
               .prefetch(AUTOTUNE))
    return dataset


def create_val_dataset(batchsize, val_idx):
    global preprocess_dataset
    parsed_val = (preprocess_dataset
                  .filter(create_idx_filter(val_idx))
                  .map(_remove_idx))
    dataset = (parsed_val
               .map(_preprocess_image_val_function, num_parallel_calls=AUTOTUNE)
               .map(_create_annot, num_parallel_calls=AUTOTUNE)
               .batch(batchsize, num_parallel_calls=AUTOTUNE)
               .cache())
    return dataset


class CosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            global_steps,
            learning_rate_max,
            learning_rate_min,
            cycle):
        super().__init__()
        self.global_steps = tf.cast(global_steps, dtype=tf.float32)
        self.learning_rate_max = tf.cast(learning_rate_max, dtype=tf.float32)
        self.learning_rate_min = tf.cast(learning_rate_min, dtype=tf.float32)
        self.cycle = tf.cast(cycle, dtype=tf.int32)
        self.learning_rate = tf.Variable(0., tf.float32)

    def __call__(self, step):
        step_epoch = tf.cast(step, tf.float32) / tf.cast(CFG.iteration_per_epoch, tf.float32)
        step_epoch = tf.cast(step_epoch, tf.int32)
        learning_rate = self.learning_rate_min + 0.5 * (self.learning_rate_max - self.learning_rate_min) * \
                        (1 + tf.math.cos(tf.constant(math.pi, tf.float32) *
                                         (tf.cast(step_epoch % self.cycle, tf.float32) / tf.cast(self.cycle,
                                                                                                 tf.float32))))
        self.learning_rate.assign(learning_rate)
        return learning_rate

    def get_config(self):
        return {
            "global_steps": self.global_steps,
            "learning_rate_max": self.learning_rate_max,
            "learning_rate_min": self.learning_rate_min,
            "cycle": self.cycle
        }

    def return_lr(self):
        return self.learning_rate


class ShowLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.return_lr()
        print("lr:", lr.numpy())


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
    optimizer = tf.keras.optimizers.Adam(CFG.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
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
    vdataset = create_val_dataset(CFG.batch_size, idx_val_tf)
    model.load_weights(path + "/model_best_%d.h5" % count)
    rec_ids = []
    probs = []
    for data, name in tqdm(vdataset):
        pred = model.predict_on_batch(tf.reshape(data, [-1, CFG.HEIGHT, CFG.WIDTH, 3]))
        rec_id_stack = tf.reshape(name, [-1, 1])
        for rec in name.numpy():
            assert len(np.unique(rec)) == 1
        rec_ids.append(rec_id_stack.numpy()[:, 0])
        probs.append(pred)
    crec_ids = np.concatenate(rec_ids)
    cprobs = np.concatenate(probs)
    sub_with_prob = pd.DataFrame({
        'id': list(map(lambda x: x.decode(), crec_ids.tolist())),
        'target': cprobs
    })
    sub_with_prob.describe()
    sub_with_prob.to_csv("submission_with_prob_" + str(count) + ".csv", index=False)
    return sub_with_prob


def train(splits, split_id):
    print("batchsize", CFG.batch_size)
    model = create_model()
    idx_train_tf = tf.cast(tf.constant(splits[split_id][0]), tf.int64)
    idx_val_tf = tf.cast(tf.constant(splits[split_id][1]), tf.int64)
    # 生成训练集和验证集
    dataset = create_train_dataset(CFG.batch_size, idx_train_tf)
    vdataset = create_val_dataset(CFG.batch_size, idx_val_tf)
    # log_dir = "logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=2)
    history = model.fit(dataset,
                        batch_size=CFG.batch_size,
                        steps_per_epoch=CFG.iteration_per_epoch,
                        epochs=CFG.epoch,
                        validation_data=vdataset,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath='./model/model_best_%d.h5' % split_id,
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True)
                        ])
    plot_history(history, 'history_%d.png' % split_id)


for i in range(CFG.k_fold):
    train(splits, i)
    inference(i, "./model")
