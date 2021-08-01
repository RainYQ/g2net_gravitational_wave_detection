import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import math
import joblib

Show = False
NUMBER_IN_TFRECORD = 512
TRAIN_DATA_ROOT = "./train_cqt_power"
TEST_DATA_ROOT = "./test_cqt_power"
train = pd.read_csv('training_labels.csv')
test = pd.read_csv('sample_submission.csv')
if Show:
    print("label type:", len(set(train["target"])))
    data_count = train["target"].value_counts()
    plt.bar([i for i in range(len(set(train["target"])))], data_count)
    plt.xlabel('label type')
    plt.ylabel('count')
    plt.show()


def create_dataset(data, i, mode):
    with tf.io.TFRecordWriter('F:/' + mode + '_tfrecords/' + mode + '_' + str(int(i)) + '.tfrecords') as writer:
        for id, label in tqdm(zip(data["id"], data["target"])):
            if mode == "train":
                raw = np.load(os.path.join(TRAIN_DATA_ROOT, id) + ".npy").tobytes()
                features = tf.train.Features(feature={
                    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                })
            elif mode == "test":
                raw = np.load(os.path.join(TEST_DATA_ROOT, id) + ".npy").tobytes()
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
    if not os.path.exists("./train_tfrecords"):
        os.mkdir("./train_tfrecords")
    if not os.path.exists("./test_tfrecords"):
        os.mkdir("./test_tfrecords")
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
    # _ = joblib.Parallel(n_jobs=16)(
    #     joblib.delayed(create_dataset)
    #     (test[n * NUMBER_IN_TFRECORD:min((n + 1) * NUMBER_IN_TFRECORD, test.shape[0])], n, "test")
    #     for n in range(math.ceil(test.shape[0] / NUMBER_IN_TFRECORD))
    # )
