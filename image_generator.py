import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import joblib
import cv2

TRAIN_DATA_ROOT = "./train_melspec"
TEST_DATA_ROOT = "./test_melspec"
train = pd.read_csv('training_labels.csv')
test = pd.read_csv('sample_submission.csv')
THREAD_NUMBER = 512


def image_generator(data, mode):
    if mode == "train":
        ROOT = TRAIN_DATA_ROOT
    else:
        ROOT = TEST_DATA_ROOT
    for id, label in tqdm(zip(data["id"], data["target"])):
        data = np.load(os.path.join(ROOT, id) + ".npy")
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
        data = data.astype(np.uint8)
        data = cv2.resize(data, (512, 512))
        data = cv2.applyColorMap(data, cv2.COLORMAP_JET)
        if mode == "train":
            cv2.imwrite(os.path.join("./train_images", str(int(label)), id + ".png"), data)
        else:
            cv2.imwrite(os.path.join("./test_images", id + ".png"), data)


if __name__ == "__main__":
    if os.path.exists("./train_images")
        os.mkdir("./train_images")
    if os.path.exists("./test_images")
        os.mkdir("./test_images")
    image_generator(train[0:THREAD_NUMBER], "train")
    _ = joblib.Parallel(n_jobs=16)(
        joblib.delayed(image_generator)
        (train[n * THREAD_NUMBER:min((n + 1) * THREAD_NUMBER, train.shape[0])], "train")
        for n in range(math.ceil(train.shape[0] / THREAD_NUMBER))
    )
    _ = joblib.Parallel(n_jobs=16)(
        joblib.delayed(image_generator)
        (test[n * THREAD_NUMBER:min((n + 1) * THREAD_NUMBER, test.shape[0])], "test")
        for n in range(math.ceil(test.shape[0] / THREAD_NUMBER))
    )
