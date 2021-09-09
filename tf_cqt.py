import warnings
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from scipy.signal import get_window, butter, sosfiltfilt
from scipy import signal
import scipy
import os
from matplotlib import pyplot as plt
import time


class CFG:
    # *******************************************************************************************
    # CQT Parameters
    sample_rate = 2048.0
    fmin = 20.0
    fmax = 512.0
    hop_length = 16
    bins_per_octave = 24
    whiten = False
    bandpass = True
    bandpass_with_tukey = True
    ts = 0.1
    length = 4096
    tukey = tf.cast(scipy.signal.windows.get_window(('tukey', ts), length), tf.float32)
    use_tukey = True
    # *******************************************************************************************
    # Sample Parameters
    wave_data_prefix = "./"
    sample_id_group = ['0021f9dd71', '544b2aeb60']
    mode = 'test'
    # *******************************************************************************************
    # Resize Parameters
    HEIGHT = 256
    WIDTH = 256


def get_file_path(image_id, mode):
    return os.path.join(CFG.wave_data_prefix,
                        "{}/{}/{}/{}/{}.npy".format(mode, image_id[0], image_id[1], image_id[2], image_id))


def create_cqt_kernels(
        q: float,
        fs: float,
        fmin: float,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        norm: float = 1,
        window: str = "hann",
        fmax: Optional[float] = None,
        topbin_check: bool = True
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    fft_len = 2 ** _nextpow2(np.ceil(q * fs / fmin))

    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    else:
        warnings.warn("If nmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check:
        raise ValueError(f"The top bin {np.max(freqs)} Hz has exceeded the Nyquist frequency, \
                           please reduce the `n_bins`")

    kernel = np.zeros((int(n_bins), int(fft_len)), dtype=np.complex64)

    length = np.ceil(q * fs / freqs)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = np.ceil(q * fs / freq)

        if l % 2 == 1:
            start = int(np.ceil(fft_len / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fft_len / 2.0 - l / 2.0))

        sig = get_window(window, int(l), fftbins=True) * np.exp(
            np.r_[-l // 2:l // 2] * 1j * 2 * np.pi * freq / fs) / l

        if norm:
            kernel[k, start:start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            kernel[k, start:start + int(l)] = sig
    return kernel, fft_len, length, freqs


def _nextpow2(a: float) -> int:
    return int(np.ceil(np.log2(a)))


def prepare_cqt_kernel(
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        bins_per_octave=12,
        norm=1,
        filter_scale=1,
        window="hann"
):
    q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
    print(q)
    return create_cqt_kernels(q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)


cqt_kernels, KERNEL_WIDTH, lengths, _ = prepare_cqt_kernel(
    sr=int(CFG.sample_rate),
    hop_length=CFG.hop_length,
    fmin=CFG.fmin,
    fmax=CFG.fmax,
    bins_per_octave=CFG.bins_per_octave)
LENGTHS = tf.constant(lengths, dtype=tf.float32)
CQT_KERNELS_REAL = tf.constant(np.swapaxes(cqt_kernels.real[:, np.newaxis, :], 0, 2))
CQT_KERNELS_IMAG = tf.constant(np.swapaxes(cqt_kernels.imag[:, np.newaxis, :], 0, 2))
PADDING = tf.constant([[0, 0],
                       [KERNEL_WIDTH // 2, KERNEL_WIDTH // 2],
                       [0, 0]])


def create_cqt_image(wave, hop_length=16):
    CQTs = []
    for i in range(3):
        x = wave[i]
        x = tf.expand_dims(tf.expand_dims(x, 0), 2)
        x = tf.pad(x, PADDING, "REFLECT")

        CQT_real = tf.nn.conv1d(x, CQT_KERNELS_REAL, stride=hop_length, padding="VALID")
        CQT_imag = -tf.nn.conv1d(x, CQT_KERNELS_IMAG, stride=hop_length, padding="VALID")
        CQT_real *= tf.math.sqrt(LENGTHS)
        CQT_imag *= tf.math.sqrt(LENGTHS)

        CQT = tf.math.sqrt(tf.pow(CQT_real, 2) + tf.pow(CQT_imag, 2))
        CQTs.append(CQT[0])
    return tf.stack(CQTs, axis=2)


def create_cqt_image_optimized(wave, hop_length=16):
    x = tf.expand_dims(wave, 2)
    x = tf.pad(x, PADDING, "REFLECT")
    CQT_real = tf.nn.conv1d(x, CQT_KERNELS_REAL, stride=hop_length, padding="VALID")
    CQT_imag = -tf.nn.conv1d(x, CQT_KERNELS_IMAG, stride=hop_length, padding="VALID")
    CQT_real *= tf.math.sqrt(LENGTHS)
    CQT_imag *= tf.math.sqrt(LENGTHS)
    CQT = tf.math.sqrt(tf.pow(CQT_real, 2) + tf.pow(CQT_imag, 2))
    return tf.transpose(CQT, (1, 2, 0))


def create_cqt_image_batch(wave, hop_length=16):
    x = tf.expand_dims(wave, 3)
    x = tf.pad(x, PADDING, "REFLECT")
    CQT_real = tf.nn.conv1d(x, CQT_KERNELS_REAL, stride=hop_length, padding="VALID")
    CQT_imag = -tf.nn.conv1d(x, CQT_KERNELS_IMAG, stride=hop_length, padding="VALID")
    CQT_real *= tf.math.sqrt(LENGTHS)
    CQT_imag *= tf.math.sqrt(LENGTHS)
    CQT = tf.math.sqrt(tf.pow(CQT_real, 2) + tf.pow(CQT_imag, 2))
    return tf.transpose(CQT, (0, 2, 3, 1))


def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data):
    filter_sos = butter_bandpass(20., 500., 2048, order=5)
    y = sosfiltfilt(filter_sos, data, padlen=1024)
    return y


def tukey_window(data):
    window = CFG.tukey
    return data * window


d_raw = np.load(get_file_path(CFG.sample_id_group[0], CFG.mode)).astype(np.float64).astype(np.float64)
# Min Max Scaler -1 1
d = (d_raw - np.min(d_raw)) / (np.max(d_raw) - np.min(d_raw))
d = (d - 0.5) * 2
# bandpass filter
if CFG.bandpass:
    if CFG.bandpass_with_tukey:
        d = d * signal.tukey(4096, 0.2)
    d = butter_bandpass_filter(d)
d = d.astype(np.float32)
plt.figure()
start = time.time()
image = create_cqt_image(d)
end = time.time()
print('Time cost:', end - start)
image = image.numpy()[:, :, 0].T
origin_data = image.copy()
plt.pcolormesh(image)
plt.show()

plt.figure()
start = time.time()
image = create_cqt_image_optimized(d)
end = time.time()
print('Time cost:', end - start)
image = image.numpy()[:, :, 0].T
plt.pcolormesh(image)
plt.show()
print(np.max(origin_data - image))

PADDING = tf.constant([[0, 0],
                       [0, 0],
                       [KERNEL_WIDTH // 2, KERNEL_WIDTH // 2],
                       [0, 0]])
data = []
for sample_id in CFG.sample_id_group:
    d_raw = np.load(get_file_path(sample_id, CFG.mode)).astype(np.float64).astype(np.float64)
    # Min Max Scaler -1 1
    d = (d_raw - np.min(d_raw)) / (np.max(d_raw) - np.min(d_raw))
    d = (d - 0.5) * 2

    plt.figure()
    # bandpass filter
    if CFG.bandpass:
        if CFG.bandpass_with_tukey:
            d = d * signal.tukey(4096, 0.2)
        d = butter_bandpass_filter(d)
    d = d.astype(np.float32)
    d = tf.cast(d, tf.float32)
    data.append(d)

d = tf.stack(data, axis=0)
plt.figure()
start = time.time()
image = create_cqt_image_batch(d)
end = time.time()
print('Time cost:', end - start)
image = image.numpy()[0, :, :, 0].T
origin_data = image.copy()
plt.pcolormesh(image)
plt.show()
