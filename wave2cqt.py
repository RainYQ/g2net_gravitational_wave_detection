import numpy as np
import pandas as pd
import os
import joblib
from tqdm.auto import tqdm
from gwpy.timeseries import TimeSeries
import cv2


def get_file_path(image_id, mode):
    return os.path.join(CFG.wave_data_prefix,
                        "{}/{}/{}/{}/{}.npy".format(mode, image_id[0], image_id[1], image_id[2], image_id))


class CFG:
    sample_rate = 2048
    f_min = 20
    f_max = 512
    out_dir_train = "train_cqt_power"
    out_dir_test = "test_cqt_power"
    wave_data_prefix = "/mnt/f"
    output_prefix = "./"
    whiten = True
    bandpass = True
    clip = False
    resize = True
    width = 512
    height = 512


train = pd.read_csv('training_labels.csv')
test = pd.read_csv('sample_submission.csv')
train['file_path'] = train['id'].apply(get_file_path, args=("train",))
test['file_path'] = test['id'].apply(get_file_path, args=("test",))


def _wav_to_cqt_gwpy(data):
    powers = []
    for i in range(data.shape[0]):
        ts = TimeSeries(data[i], sample_rate=CFG.sample_rate)
        if CFG.whiten:
            ts = ts.whiten(window=("tukey", 0.15))
        if CFG.bandpass:
            ts = ts.bandpass(20, 512, 0.1)
        cqt = ts.q_transform(qrange=(10, 16), frange=(CFG.f_min, CFG.f_max), logf=True, whiten=False)
        power = cqt.__array__()
        power = np.transpose(power)
        if CFG.resize:
            # width - x - col
            # height - y - row
            power = cv2.resize(power, (CFG.width, CFG.height))
        if CFG.clip:
            power = np.clip(power, 0, 15)
            power = power * 255.0 / 15.0
        else:
            power = (power - np.min(power)) / (np.max(power) - np.min(power)) * 255.0
        power = power.astype(np.uint8)
        powers.append(power)
    return np.array(powers).astype(np.uint8)


def save_data_train(id):
    waves = np.load(get_file_path(id, "train")).astype(np.float64)
    np.save(os.path.join(CFG.output_prefix, CFG.out_dir_train, id + ".npy"), _wav_to_cqt_gwpy(waves))


def save_data_test(id):
    waves = np.load(get_file_path(id, "test")).astype(np.float64)
    np.save(os.path.join(CFG.output_prefix, CFG.out_dir_test, id + ".npy"), _wav_to_cqt_gwpy(waves))


if __name__ == '__main__':
    if not os.path.exists(os.path.join(CFG.output_prefix, CFG.out_dir_train)):
        os.mkdir(os.path.join(CFG.output_prefix, CFG.out_dir_train))
    if not os.path.exists(os.path.join(CFG.output_prefix, CFG.out_dir_test)):
        os.mkdir(os.path.join(CFG.output_prefix, CFG.out_dir_test))

    # for id in tqdm(train['id'][:10].values):
    #     save_data_train(id)

    _ = joblib.Parallel(n_jobs=16)(
        joblib.delayed(save_data_train)(id) for id in tqdm(train['id'].values)
    )
    _ = joblib.Parallel(n_jobs=16)(
        joblib.delayed(save_data_test)(id) for id in tqdm(test['id'].values)
    )
