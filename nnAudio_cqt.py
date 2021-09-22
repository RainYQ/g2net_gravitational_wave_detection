import numpy as np
import warnings
from typing import Optional, Tuple
from nnAudio.Spectrogram import CQT1992v2
import scipy.signal
from scipy.signal import get_window, butter, sosfiltfilt
import torch.nn.functional as F
import scipy
import torch
import os
import time
from matplotlib import pyplot as plt


def get_file_path(image_id, mode):
    return os.path.join(CFG.wave_data_prefix,
                        "{}/{}/{}/{}/{}.npy".format(mode, image_id[0], image_id[1], image_id[2], image_id))


class CFG:
    # *******************************************************************************************
    # CQT Parameters
    sample_rate = 2048.0
    fmin = 20.0
    fmax = 512.0
    hop_length = 16
    bins_per_octave = 24
    whiten = False
    whiten_use_tukey = True
    bandpass = True
    bandpass_with_tukey = True
    ts = 0.1
    length = 4096
    tukey = torch.from_numpy(scipy.signal.windows.get_window(('tukey', ts), length)).float()
    # *******************************************************************************************
    # Sample Parameters
    wave_data_prefix = "./"
    sample_id = '0021f9dd71'
    mode = 'test'
    # *******************************************************************************************
    # Resize Parameters
    HEIGHT = 512
    WIDTH = 512


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
LENGTHS = torch.tensor(lengths, dtype=torch.float32)
CQT_KERNELS_REAL = torch.tensor(cqt_kernels.real, dtype=torch.float32).unsqueeze(1)
CQT_KERNELS_IMAG = torch.tensor(cqt_kernels.imag, dtype=torch.float32).unsqueeze(1)
padding = torch.nn.ReflectionPad1d(KERNEL_WIDTH // 2)


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


def cqt(wave, hop_length=16):
    x = padding(wave.unsqueeze(1))
    CQT_real = F.conv1d(x, CQT_KERNELS_REAL, stride=CFG.hop_length)
    CQT_imag = -F.conv1d(x, CQT_KERNELS_IMAG, stride=CFG.hop_length)
    CQT_real *= torch.sqrt(LENGTHS.view(-1, 1))
    CQT_imag *= torch.sqrt(LENGTHS.view(-1, 1))
    CQT = torch.sqrt(torch.pow(CQT_real, 2) + torch.pow(CQT_imag, 2))
    return CQT


d_raw = np.load(get_file_path(CFG.sample_id, CFG.mode)).astype(np.float64)
# Min Max Scaler -1 1
d = (d_raw - np.min(d_raw)) / (np.max(d_raw) - np.min(d_raw))
d = (d - 0.5) * 2

plt.figure()
# bandpass filter
if CFG.bandpass:
    if CFG.bandpass_with_tukey:
        d = d * scipy.signal.tukey(4096, 0.2)
    d = butter_bandpass_filter(d)
d = d.astype(np.float32)
d = torch.tensor(d, dtype=torch.float32)

cqt_kernel_nnAudio = CQT1992v2(sr=int(CFG.sample_rate),
                       fmin=CFG.fmin,
                       fmax=CFG.fmax,
                       hop_length=CFG.hop_length,
                       bins_per_octave=CFG.bins_per_octave)

origin_data = cqt_kernel_nnAudio(d).numpy()
for i in range(3):
    plt.figure()
    image = origin_data[i, :, :]
    plt.pcolormesh(image)
start = time.time()
torch_data = cqt(d).numpy()
end = time.time()
print('Time cost:', end - start)
for i in range(3):
    plt.figure()
    image = torch_data[i, :, :]
    plt.pcolormesh(image)
plt.show()
print(np.max(origin_data - torch_data))
