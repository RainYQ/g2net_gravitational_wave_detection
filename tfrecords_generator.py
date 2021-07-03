import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import math
import joblib

Show = False
NUMBER_IN_TFRECORD = 5120
TRAIN_DATA_ROOT = "./train_melspec"
TEST_DATA_ROOT = "./test_melspec"
train = pd.read_csv('training_labels.csv')
test = pd.read_csv('sample_submission.csv')
if Show:
    print("label type:", len(set(train["target"])))
    data_count = train["target"].value_counts()
    plt.bar([i for i in range(len(set(train["target"])))], data_count)
    plt.xlabel('label type')
    plt.ylabel('count')
    plt.show()


def pares_image_train(id):
    raw = np.load(os.path.join(TRAIN_DATA_ROOT, id) + ".npy").tobytes()
    # Performance Bottleneck Point
    label = train.loc[train["id"] == id]["target"].values[0]
    return raw, label


def pares_image_test(id):
    raw = np.load(os.path.join(TEST_DATA_ROOT, id) + ".npy").tobytes()
    return raw


def create_dataset(data, i, mode):
    with tf.io.TFRecordWriter('./' + mode + '_tfrecords/' + mode + '_' + str(int(i)) + '.tfrecords') as writer:
        for id in tqdm(data["id"]):
            if mode == "train":
                raw, label= pares_image_train(id)
                features = tf.train.Features(feature={
                    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                })
            elif mode == "test":
                raw = pares_image_test(id)
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
    # Enable Multi-Thread
    # for n in range(math.ceil(train.shape[0] / NUMBER_IN_TFRECORD)):
    #     t1 = threading.Thread(target=create_dataset, args=(
    #         train[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, train.shape[0])], n))
    #     t1.start()
    # for n in range(math.ceil(test.shape[0] / NUMBER_IN_TFRECORD)):
    #     t1 = threading.Thread(target=create_dataset, args=(
    #         test[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, test.shape[0])], n))
    #     t1.start()
    # Better Performence
    _ = joblib.Parallel(n_jobs=16)(
        joblib.delayed(create_dataset)
        (train[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, train.shape[0])], n, "train")
        for n in range(math.ceil(train.shape[0] / NUMBER_IN_TFRECORD))
    )
    _ = joblib.Parallel(n_jobs=16)(
        joblib.delayed(create_dataset)
        (test[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, test.shape[0])], n, "test")
        for n in range(math.ceil(test.shape[0] / NUMBER_IN_TFRECORD))
    )
