import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa.display
import os
import joblib
from tqdm.auto import tqdm
import threading
import math


class CFG:
    # Mel-Spectrogram parameters
    sample_rate = 2048
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    mel_power = 2
    f_min = 20
    f_max = 1024
    Pad_End = False
    # Out file parameters
    out_dir_train = "train_melspec"
    out_dir_test = "test_melspec"
    use_tf = False


# Get the Full filepath for every .npy
def get_file_path(image_id, mode):
    return "{}/{}/{}/{}/{}.npy".format(mode, image_id[0], image_id[1], image_id[2], image_id)


def preprocess(waves):
    max_data = np.max(waves, axis=1)
    waves[0] = waves[0] / max_data[0]
    waves[1] = waves[1] / max_data[1]
    waves[2] = waves[2] / max_data[2]
    return waves


train = pd.read_csv('training_labels.csv')
test = pd.read_csv('sample_submission.csv')
train['file_path'] = train['id'].apply(get_file_path, args=("train",))
test['file_path'] = test['id'].apply(get_file_path, args=("test",))
waves = np.load(train.loc[0, 'file_path']).astype(np.float32)  # (3, 4096)
waves = preprocess(waves)


# Use librosa
def _wav_to_spec_librosa(data):
    melspecs = []
    for j in range(3):
        melspec = librosa.feature.melspectrogram(data[j],
                                                 sr=CFG.sample_rate, n_mels=CFG.n_mels,
                                                 fmin=CFG.f_min, fmax=CFG.f_max, n_fft=CFG.n_fft,
                                                 hop_length=CFG.hop_length)
        melspec = librosa.power_to_db(melspec)
        melspec = melspec.transpose((1, 0))
        melspecs.append(melspec)
    return np.vstack(melspecs)


# plt.imshow(_wav_to_spec_librosa(waves))
# plt.show()


# Use tensorflow
def _wav_to_spec(data):
    log_mel_spectrograms = []
    for i in range(3):
        stfts = tf.signal.stft(data[i], frame_length=CFG.n_fft, frame_step=CFG.hop_length, fft_length=CFG.n_fft,
                               pad_end=CFG.Pad_End)
        spectrogram = tf.abs(stfts) ** CFG.mel_power
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=CFG.n_mels,
            num_spectrogram_bins=stfts.shape[-1],
            sample_rate=CFG.sample_rate,
            lower_edge_hertz=CFG.f_min,
            upper_edge_hertz=CFG.f_max)
        mel_spectrogram = tf.tensordot(
            spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms.append(tf.math.log(mel_spectrogram + 1e-6).numpy())
    return np.vstack(log_mel_spectrograms)


# plt.imshow(_wav_to_spec(waves))
# plt.show()


def save_data_train(id):
    waves = np.load(get_file_path(id, "train")).astype(np.float32)
    waves = preprocess(waves)
    if not CFG.use_tf:
        np.save(os.path.join(CFG.out_dir_train, id + ".npy"), _wav_to_spec_librosa(waves))
    else:
        np.save(os.path.join(CFG.out_dir_train, id + ".npy"), _wav_to_spec(waves))


def save_data_test(id):
    waves = np.load(get_file_path(id, "test")).astype(np.float32)
    waves = preprocess(waves)
    if not CFG.use_tf:
        np.save(os.path.join(CFG.out_dir_test, id + ".npy"), _wav_to_spec_librosa(waves))
    else:
        np.save(os.path.join(CFG.out_dir_test, id + ".npy"), _wav_to_spec(waves))


def tf_train_thread(data):
    for id in tqdm(data['id'].values):
        save_data_train(id)


def tf_test_thread(data):
    for id in tqdm(data['id'].values):
        save_data_test(id)


if __name__ == '__main__':
    if os.path.exists("./train_melspec")
        os.mkdir("./train_melspec")
    if os.path.exists("./test_melspec")
        os.mkdir("./test_melspec")
    if not CFG.use_tf:
        _ = joblib.Parallel(n_jobs=16)(
            joblib.delayed(save_data_train)(id) for id in tqdm(train['id'].values)
        )
        _ = joblib.Parallel(n_jobs=16)(
            joblib.delayed(save_data_test)(id) for id in tqdm(test['id'].values)
        )
    else:
        number = math.ceil(train.shape[0] / 16)
        for n in range(16):
            t1 = threading.Thread(target=tf_train_thread, args=(
                train[n * number:min((n + 1) * number, train.shape[0])], ))
            t1.start()
        number = math.ceil(test.shape[0] / 16)
        for n in range(16):
            t1 = threading.Thread(target=tf_test_thread, args=(
                test[n * number:min((n + 1) * number, test.shape[0])], ))
            t1.start()
