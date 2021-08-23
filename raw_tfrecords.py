import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import math
import joblib

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CFG:
    Show = False
    NUMBER_IN_TFRECORD = 4096
    wave_data_prefix = "F:/"
    tfrecords_prefix = "./"


train = pd.read_csv('training_labels.csv')
test = pd.read_csv('sample_submission.csv')
if CFG.Show:
    print("label type:", len(set(train["target"])))
    data_count = train["target"].value_counts()
    plt.bar([i for i in range(len(set(train["target"])))], data_count)
    plt.xlabel('label type')
    plt.ylabel('count')
    plt.show()


def get_file_path(image_id, mode):
    return os.path.join(CFG.wave_data_prefix,
                        "{}/{}/{}/{}/{}.npy".format(mode, image_id[0], image_id[1], image_id[2], image_id))


def create_dataset(data, i, mode):
    with tf.io.TFRecordWriter(
            os.path.join(
                CFG.tfrecords_prefix + mode + '_tfrecords/' + mode + '_' + str(int(i)) + '.tfrecords')) as writer:
        for id, label in tqdm(zip(data["id"], data["target"])):
            raw = np.load(get_file_path(id, mode)).astype(np.float32).tobytes()
            if mode == "train":
                features = tf.train.Features(feature={
                    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                })
            elif mode == "test":
                features = tf.train.Features(feature={
                    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw]))
                })
            else:
                print("Error mode.")
                sys.exit(1)
            exam = tf.train.Example(features=features)
            writer.write(exam.SerializeToString())


if __name__ == "__main__":
    if not os.path.exists(os.path.join(CFG.tfrecords_prefix, "train_tfrecords")):
        os.mkdir(os.path.join(CFG.tfrecords_prefix, "train_tfrecords"))
    if not os.path.exists(os.path.join(CFG.tfrecords_prefix, "test_tfrecords")):
        os.mkdir(os.path.join(CFG.tfrecords_prefix, "test_tfrecords"))

    _ = joblib.Parallel(n_jobs=16)(
        joblib.delayed(create_dataset)
        (train[n * CFG.NUMBER_IN_TFRECORD:min((n + 1) * CFG.NUMBER_IN_TFRECORD, train.shape[0])], n, "train")
        for n in range(math.ceil(train.shape[0] / CFG.NUMBER_IN_TFRECORD))
    )
    _ = joblib.Parallel(n_jobs=16)(
        joblib.delayed(create_dataset)
        (test[n * CFG.NUMBER_IN_TFRECORD:min((n + 1) * CFG.NUMBER_IN_TFRECORD, test.shape[0])], n, "test")
        for n in range(math.ceil(test.shape[0] / CFG.NUMBER_IN_TFRECORD))
    )
