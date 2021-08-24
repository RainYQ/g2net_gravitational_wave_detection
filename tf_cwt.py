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


class CFG:
    origin_data_prefix = "F:"
    sample_id = '0021f9dd71'
    part = 'test'
    sample_rate = 2048.0
    channel = 0
    fmin = 20.0
    fmax = 512.0
    nv = 16
    whiten = True
    bandpass = True
    trainable = False


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
    padvalue = padvalue
    num_scales = scales.shape[-1]
    return wft, padvalue, num_scales


# Change to function
def cwt(input, nv=16, sr=2048.0, flow=20.0, fhigh=512.0, batch_size=None, trainable=False):
    assert fhigh > flow, 'fhigh parameters must be > flow!'
    assert batch_size is not None, 'batch size must be set!'
    assert len(input.shape) == 2, 'Input dimension must be 2! Dimension is {}'.format(len(input.shape))
    wft, padvalue, num_scales = cwt_pre(input.shape, nv, sr, flow, fhigh, trainable)
    max_loop = tf.shape(input)[0]

    def sum_cwt(i, pre_data):
        next_data = tf.nn.embedding_lookup(input, i)
        x = tf.concat([tf.reverse(next_data[0:padvalue], axis=[0]), next_data,
                       tf.reverse(next_data[-padvalue:], axis=[0])], 0)
        f = tf.signal.fft(tf.cast(x, tf.complex64))
        cwtcfs = tf.signal.ifft(
            tnp.kron(tf.ones([num_scales, 1], dtype=tf.complex64), f) * tf.cast(wft, tf.complex64))
        logcwt = tf.math.log(tf.math.abs(cwtcfs[:, padvalue:padvalue + next_data.shape[-1]]))
        pre_data = tf.tensor_scatter_nd_add(pre_data, indices=[[i]], updates=[logcwt])
        i_next = i + 1
        return i_next, pre_data

    _, cwt = tf.while_loop(cond=lambda i, result: tf.less(i, max_loop),
                           body=sum_cwt,
                           loop_vars=(tf.constant(0, dtype=tf.int32),
                                      tf.zeros([batch_size, num_scales, input.shape[-1]],
                                               dtype=tf.float32)))
    return cwt


def MinMaxScaler(data, lower, upper):
    assert upper > lower, 'fhigh parameters must be > flow!'
    lower = tf.cast(lower, tf.float32)
    upper = tf.cast(upper, tf.float32)
    min_val = tf.math.reduce_min(data)
    max_val = tf.math.reduce_max(data)
    std_data = tf.divide(tf.subtract(data, min_val), tf.subtract(max_val, min_val))
    return tf.add(tf.multiply(std_data, tf.subtract(upper, lower)), lower)


def whiten_torch(signal):
    hann = torch.hann_window(len(signal), periodic=True, dtype=float)
    spec = fft(torch.from_numpy(signal.copy()).float() * hann)
    mag = torch.sqrt(torch.real(spec * torch.conj(spec)))
    cmx = spec / mag
    real = ifft(cmx)
    return torch.real(ifft(spec / mag)).numpy() * np.sqrt(len(signal) / 2)


def whiten(signal):
    signal = tf.cast(signal, tf.float32)
    hann = tf.signal.hann_window(signal.shape[0], periodic=True)
    spec = tf.signal.fft(tf.cast(signal * hann, tf.complex128))
    mag = tf.math.sqrt(tf.math.real(spec * tf.math.conj(spec)))
    return tf.cast(tf.math.real(tf.signal.ifft(spec / tf.cast(mag, tf.complex128))), tf.float32) * tf.math.sqrt(
        signal.shape[0] / 2)


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


def tukey_window(data, ts, len):
    window = scipy.signal.windows.get_window(('tukey', ts), len)
    return data * window


d_raw = np.load('F:/test/0/0/2/0021f9dd71.npy').astype(np.float64)[0]
# Min Max Scaler -1 1
d = (d_raw - np.min(d_raw)) / (np.max(d_raw) - np.min(d_raw))
d = (d - 0.5) * 2
plt.figure()
# tukey
# d = tukey_window(d, 0.1, 4096)
# bandpass filter
if CFG.bandpass:
    d = butter_bandpass_filter(d)

d_torch = whiten_torch(d)
d_tf = whiten(d)
plt.figure()
plt.plot(d_torch, label='torch')
plt.plot(d_tf.numpy(), label='tensorflow')
plt.plot(d_torch - d_tf.numpy(), label='difference')
plt.legend()
plt.show()

if CFG.whiten:
    d = whiten(d)

d = tf.cast(d, tf.float32)

start = time.time()
Wavelet1D(CFG.nv, sr=CFG.sample_rate, flow=CFG.fmin, fhigh=CFG.fmax, batch_size=1, trainable=CFG.trainable)(tf.expand_dims(d, axis=0))
end = time.time()
print('Time cost:', end - start)

plt.figure()
start = time.time()
y = cwt(tf.expand_dims(d, axis=0), nv=CFG.nv, sr=CFG.sample_rate, flow=CFG.fmin,
        fhigh=CFG.fmax, batch_size=1, trainable=CFG.trainable)
end = time.time()
print('Time cost:', end - start)
y = MinMaxScaler(y, 0, 1)
# Remove batch size dim
y = tf.squeeze(y)
# flip for pcolormesh
y = np.flip(y.numpy(), 0)
plt.pcolormesh(y)
plt.show()
