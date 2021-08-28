"""
From https://www.kaggle.com/mistag/wavelet1d-custom-keras-wavelet-transform-layer/comments
"""

import scipy
from scipy import signal
from matplotlib import pyplot as plt
import math
from tensorflow import keras
import tensorflow as tf
import numpy as np
import tensorflow.experimental.numpy as tnp
import torch
from torch.fft import fft, ifft
from scipy.signal import butter, sosfiltfilt
import time
import os


class CFG:
    # *******************************************************************************************
    # CWT Parameters
    sample_rate = 2048.0
    fmin = 20.0
    fmax = 512.0
    nv = 32
    whiten = True
    bandpass = True
    trainable = False
    ts = 0.1
    len = 4096
    tukey = tf.cast(scipy.signal.windows.get_window(('tukey', ts), len), tf.float32)
    use_tukey = True
    # *******************************************************************************************
    # Sample Parameters
    wave_data_prefix = "F:/"
    sample_id_group = ['01d162f247', '333a674c18']
    mode = 'train'
    # *******************************************************************************************
    # Resize Parameters
    HEIGHT = 256
    WIDTH = 256
    # *******************************************************************************************
    # Show Parameters
    show_error_plot = False


def get_file_path(image_id, mode):
    return os.path.join(CFG.wave_data_prefix,
                        "{}/{}/{}/{}/{}.npy".format(mode, image_id[0], image_id[1], image_id[2], image_id))


# calculate CWT of input signal
class Wavelet1D(keras.layers.Layer):
    def __init__(self, nv=12, sr=1., flow=0., fhigh=0.5, batch_size=None, trainable=False):
        super(Wavelet1D, self).__init__()
        assert fhigh > flow, 'fhigh parameters must be > flow!'
        assert batch_size is not None, 'batch size must be set!'

        self.batch_size = batch_size
        self.nv = nv  # number of voices
        self.sr = sr  # sample rate (Hz)
        self.flow = flow  # lowest frequency of interest (Hz)
        self.fhigh = fhigh  # highest frequency of interest (Hz)
        self.trainable = trainable  # True to train the wavelet filter bank

    def build(self, input_shape):
        assert len(input_shape) == 2, 'Input dimension must be 2! Dimension is {}'.format(len(input_shape))

        max_scale = input_shape[-1] // (np.sqrt(2) * 2)
        if max_scale <= 1:
            max_scale = input_shape[-1] // 2
        max_scale = np.floor(self.nv * np.log2(max_scale))
        scales = 2 * (2 ** (1 / self.nv)) ** np.arange(0, max_scale + 1)
        frequencies = self.sr * (6 / (2 * np.pi)) / scales
        frequencies = frequencies[frequencies >= self.flow]  # remove low frequencies
        scales = scales[0:len(frequencies)]
        frequencies = frequencies[frequencies <= self.fhigh]  # remove high frequencies
        scales = scales[len(scales) - len(frequencies):len(scales)]
        # wavft
        padvalue = input_shape[-1] // 2
        n = padvalue * 2 + input_shape[-1]
        omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
        omega = omega * (2 * np.pi) / n
        omega = np.concatenate((np.array([0]), omega, -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1]))
        _wft = np.zeros([scales.size, omega.size])
        for jj, scale in enumerate(scales):
            expnt = -(scale * omega - 6) ** 2 / 2 * (omega > 0)
            _wft[jj,] = 2 * np.exp(expnt) * (omega > 0)
        # parameters we want to use during call():
        self.wft = tf.Variable(_wft, trainable=self.trainable)  # yes, the wavelets can be trainable if desired
        self.padvalue = padvalue
        self.num_scales = scales.shape[-1]

    def call(self, inputs):
        max_loop = tf.shape(inputs)[0]

        def sum_cwt(i, pre_data):
            next_data = tf.nn.embedding_lookup(inputs, i)
            x = tf.concat([tf.reverse(next_data[0:self.padvalue], axis=[0]), next_data,
                           tf.reverse(next_data[-self.padvalue:], axis=[0])], 0)
            f = tf.signal.fft(tf.cast(x, tf.complex64))
            cwtcfs = tf.signal.ifft(
                tnp.kron(tf.ones([self.num_scales, 1], dtype=tf.complex64), f) * tf.cast(self.wft, tf.complex64))
            logcwt = tf.math.log(tf.math.abs(cwtcfs[:, self.padvalue:self.padvalue + next_data.shape[-1]]))
            pre_data = tf.tensor_scatter_nd_add(pre_data, indices=[[i]], updates=[logcwt])
            i_next = i + 1
            return i_next, pre_data

        _, cwt = tf.while_loop(cond=lambda i, result: tf.less(i, max_loop),
                               body=sum_cwt,
                               loop_vars=(tf.constant(0, dtype=tf.int32),
                                          tf.zeros([self.batch_size, self.num_scales, inputs.shape[-1]],
                                                   dtype=tf.float32)))
        return cwt


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
    # parameters we want to use during call():
    wft = tf.Variable(_wft, trainable=trainable)  # yes, the wavelets can be trainable if desired
    num_scales = scales.shape[-1]
    wft = tf.cast(wft, tf.complex64)
    return wft, padvalue, num_scales


wft, padvalue, num_scales = cwt_pre((1, CFG.len),
                                    nv=CFG.nv,
                                    sr=CFG.sample_rate,
                                    flow=CFG.fmin,
                                    fhigh=CFG.fmax,
                                    trainable=CFG.trainable)


# Change to function
def cwt(input, flow=20.0, fhigh=512.0, batch_size=None):
    assert fhigh > flow, 'fhigh parameters must be > flow!'
    assert batch_size is not None, 'batch size must be set!'
    assert len(input.shape) == 3, 'Input dimension must be 3! Dimension is {}'.format(len(input.shape))

    x = tf.concat([tf.reverse(input[:, :, 0:padvalue], axis=[2]), input,
                   tf.reverse(input[:, :, -padvalue:], axis=[2])], 2)
    x = tf.signal.fft(tf.cast(x, tf.complex64))
    cwtcfs = tf.signal.ifft(tf.expand_dims(x, 2) * wft)
    x = tf.math.log(tf.math.abs(cwtcfs[:, :, :, padvalue:padvalue + input.shape[-1]]))
    x = tf.transpose(x, (0, 2, 3, 1))
    return x


def MinMaxScaler(data, lower, upper):
    assert upper > lower, 'upper parameters must be > lower!'
    lower = tf.cast(lower, tf.float32)
    upper = tf.cast(upper, tf.float32)
    min_val = tf.reshape(tf.reduce_min(data, axis=[1, 2]), [tf.shape(data)[0], 1, 1, tf.shape(data)[-1]])
    max_val = tf.reshape(tf.reduce_max(data, axis=[1, 2]), [tf.shape(data)[0], 1, 1, tf.shape(data)[-1]])
    std_data = tf.divide(tf.subtract(data, min_val), tf.subtract(max_val, min_val))
    return tf.add(tf.multiply(std_data, tf.subtract(upper, lower)), lower)


def whiten_torch(signal):
    hann = torch.hann_window(signal.shape[-1], periodic=True, dtype=float)
    spec = fft(torch.from_numpy(signal.copy()).float() * hann)
    mag = torch.sqrt(torch.real(spec * torch.conj(spec)))
    return torch.real(ifft(spec / mag)).numpy() * np.sqrt(signal.shape[-1] / 2)


def whiten(signal):
    if CFG.use_tukey:
        window = CFG.tukey
    else:
        window = tf.signal.hann_window(signal.shape[-1], periodic=True)
    spec = tf.signal.fft(tf.cast(signal * window, tf.complex128))
    mag = tf.math.sqrt(tf.math.real(spec * tf.math.conj(spec)))
    return tf.cast(tf.math.real(tf.signal.ifft(spec / tf.cast(mag, tf.complex128))), tf.float32) * tf.math.sqrt(
        signal.shape[-1] / 2)


def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data):
    filter_sos = butter_bandpass(20., 512., 2048, order=8)
    y = sosfiltfilt(filter_sos, data, padlen=1024)
    return y


def tukey_window(data):
    window = CFG.tukey
    return data * window


d_raw = np.load(get_file_path(CFG.sample_id_group[0], CFG.mode)).astype(np.float64).astype(np.float64)
# Min Max Scaler -1 1
d = (d_raw - np.min(d_raw)) / (np.max(d_raw) - np.min(d_raw))
d = (d - 0.5) * 2
d = d.astype(np.float32)
plt.figure()
# bandpass filter
if CFG.bandpass:
    d = butter_bandpass_filter(d)

if CFG.show_error_plot:
    d_torch = whiten_torch(d)
    d_tf = whiten(d)
    for i in range(d_torch.shape[0]):
        plt.figure()
        plt.plot(d_torch[i, :], label='torch')
        plt.plot(d_tf.numpy()[i, :], label='tensorflow')
        plt.plot(d_torch[i, :] - d_tf.numpy()[i, :], label='difference')
        plt.legend()
        plt.title('Channel ' + str(i))
    plt.show()

if CFG.whiten:
    d = whiten(d)

d = tf.cast(d, tf.float32)

start = time.time()
y = Wavelet1D(CFG.nv, sr=CFG.sample_rate, flow=CFG.fmin, fhigh=CFG.fmax, batch_size=d.shape[0],
              trainable=CFG.trainable)(d)
end = time.time()
print('Time cost:', end - start)
y = tf.transpose(y, (1, 2, 0))
y = tf.image.resize(y, (CFG.HEIGHT, CFG.WIDTH))
y = tf.expand_dims(y, 0)
y = MinMaxScaler(y, 0, 1)
# flip for pcolormesh
y = np.flip(y.numpy()[0, :, :, 0], 0)
data_origin = y
plt.pcolormesh(y)
plt.show()

data = []
for sample_id in CFG.sample_id_group:
    d_raw = np.load(get_file_path(sample_id, CFG.mode)).astype(np.float64).astype(np.float64)
    # Min Max Scaler -1 1
    d = (d_raw - np.min(d_raw)) / (np.max(d_raw) - np.min(d_raw))
    d = (d - 0.5) * 2
    d = d.astype(np.float32)
    plt.figure()
    # bandpass filter
    if CFG.bandpass:
        d = butter_bandpass_filter(d)

    if CFG.show_error_plot:
        d_torch = whiten_torch(d)
        d_tf = whiten(d)
        for i in range(d_torch.shape[0]):
            plt.figure()
            plt.plot(d_torch[i, :], label='torch')
            plt.plot(d_tf.numpy()[i, :], label='tensorflow')
            plt.plot(d_torch[i, :] - d_tf.numpy()[i, :], label='difference')
            plt.legend()
            plt.title('Channel ' + str(i))
        plt.show()

    if CFG.whiten:
        d = whiten(d)

    d = tf.cast(d, tf.float32)
    data.append(d)

d = tf.stack(data, axis=0)
plt.figure()
start = time.time()
y = cwt(d, flow=CFG.fmin, fhigh=CFG.fmax, batch_size=3)
end = time.time()
print('Time cost:', end - start)
y = tf.image.resize(y, (CFG.HEIGHT, CFG.WIDTH))
y = MinMaxScaler(y, 0, 1)
# flip for pcolormesh
image = np.flip(y.numpy()[0, :, :, 0], 0)
plt.pcolormesh(image)
plt.show()
data_optimize = image
print(np.max(data_origin - data_optimize))
image = np.flip(y.numpy()[1, :, :, 0], 0)
plt.pcolormesh(image)
plt.show()
