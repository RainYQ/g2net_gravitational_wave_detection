import random
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import efficientnet.tfkeras as efn
import math


# Use TPU config
class CFG:
    batch_size = 128
    epoch = 20
    iteration_per_epoch = 875
    learning_rate = 1e-3
    k_fold = 5
    HEIGHT = 512
    WIDTH = 512
    RAW_HEIGHT = 512
    RAW_WIDTH = 512
    SEED = 2022
    TRAIN_DATA_SIZE = 560000
    TEST_DATA_SIZE = 226000
    TTA_STEP = 16
    result_folder = "./"


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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


def _parse_image_function(single_photo):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(single_photo, image_feature_description)


def _preprocess_image_test_function(single_photo):
    image = tf.image.decode_png(single_photo['data'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if CFG.RAW_WIDTH != CFG.WIDTH or CFG.RAW_HEIGHT != CFG.HEIGHT:
        image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    image = tf.image.random_contrast(image, lower=1.0, upper=1.3)
    image = tf.cond(tf.random.uniform([]) < 0.5,
                    lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
                    lambda: tf.image.random_hue(image, max_delta=0.3))
    # brightness随机调整
    image = tf.image.random_brightness(image, 0.3)
    single_photo['data'] = image
    return single_photo['data'], single_photo['id']


opt = tf.data.Options()
opt.experimental_deterministic = False


def create_test_dataset():
    global raw_image_dataset
    dataset = (raw_image_dataset
               .map(_parse_image_function, num_parallel_calls=AUTOTUNE)
               .cache()
               .with_options(opt)
               .map(_preprocess_image_test_function, num_parallel_calls=AUTOTUNE)
               .batch(CFG.batch_size, num_parallel_calls=AUTOTUNE)
               .prefetch(AUTOTUNE))
    return dataset


def create_model():
    backbone = efn.EfficientNetB0(
        include_top=False,
        input_shape=(CFG.HEIGHT, CFG.WIDTH, 3),
        weights=None,
        pooling='avg'
    )

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    return model


model = create_model()
tdataset = create_test_dataset()


def inference(count, path):
    global tdataset
    model.load_weights(path + "/model_best_%d.h5" % count)
    rec_ids = []
    probs = []
    for data, name in tqdm(tdataset):
        pred = model.predict_on_batch(tf.reshape(data, [-1, CFG.HEIGHT, CFG.WIDTH, 3]))
        rec_id_stack = tf.reshape(name, [-1, 1])
        for rec in name.numpy():
            assert len(np.unique(rec)) == 1
        rec_ids.append(rec_id_stack.numpy()[:, 0])
        probs.append(pred.reshape(pred.shape[0]) / (CFG.TTA_STEP * CFG.k_fold))
    crec_ids = np.concatenate(rec_ids)
    cprobs = np.concatenate(probs)
    sub_with_prob = pd.DataFrame({
        'id': list(map(lambda x: x.decode(), crec_ids.tolist())),
        'target': cprobs
    })
    sub_with_prob.describe()
    return sub_with_prob


# 5-Fold TTA Sample
sub_with_prob = sum(
    map(
        lambda j:
        inference(math.floor(j / CFG.TTA_STEP), "./model").set_index('id'), range(CFG.k_fold * CFG.TTA_STEP)
    )
).reset_index()
sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob.csv"), index=False)


# Single Fold TTA Sample
# sub_with_prob = sum(
#     map(
#         lambda j:
#         inference(3, "./model").set_index('id'), range(CFG.TTA_STEP)
#     )
# ).reset_index()
# sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob_3.csv"), index=False)
#
# sub_with_prob = sum(
#     map(
#         lambda j:
#         inference(4, "./model").set_index('id'), range(CFG.TTA_STEP)
#     )
# ).reset_index()
# sub_with_prob.to_csv(os.path.join(CFG.result_folder, "submission_with_prob_4.csv"), index=False)
