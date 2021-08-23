"""
    Use WSL
"""
import numpy as np
import os
import pycbc
from pycbc.psd import welch, interpolate
from gwpy.timeseries import TimeSeries
import pylab
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl
from scipy import signal
from scipy.cluster.vq import whiten


class CFG:
    USE_LIGO = False
    sample_id = '0021f9dd71'
    part = 'test'
    channel = 0
    # Use for GWpy
    whiten = True
    # Use for All
    clip = False
    fig_save_path = "./cqt_analysis_sample"
    origin_data_prefix = "/mnt/f"


def get_file_path(image_id):
    return os.path.join(CFG.origin_data_prefix,
                        CFG.part + "/{}/{}/{}/{}.npy".format(image_id[0], image_id[1], image_id[2], image_id))


def std_cqt(ts, name):
    plt.figure()
    time, freq, power = ts.qtransform(15.0 / 2048.0, logfsteps=1000, qrange=(10, 10), frange=(20, 512))
    pylab.title(CFG.sample_id)
    # np.clip is equal to vmin vmax
    # power = np.clip(power, 0, 15)
    # pylab.pcolormesh(time, freq, power)
    if CFG.clip:
        pylab.pcolormesh(time, freq, power, vmax=15, vmin=0)
    else:
        pylab.pcolormesh(time, freq, power)
    pylab.yscale('log')
    pylab.savefig(os.path.join(CFG.fig_save_path, name + ".png"), dpi=300)
    return power


def std_gwpy_cqt(ts, name):
    cqt = ts.q_transform(qrange=(10, 16), frange=(20, 512), logf=True, whiten=False)
    power = cqt.__array__()
    power = np.transpose(power)
    time = cqt.xindex.__array__()
    freq = cqt.yindex.__array__()
    if CFG.clip:
        plt.pcolormesh(time, freq, power, vmax=15, vmin=0)
    else:
        plt.pcolormesh(time, freq, power)
    plt.yscale('log')
    plt.title(CFG.sample_id)
    plt.savefig(os.path.join(CFG.fig_save_path, "cqt_gwpy_whiten.png"), dpi=300)


if CFG.USE_LIGO:
    fileName = './LIGO_data/L-L1_GWOSC_O3a_4KHZ_R1-1238163456-4096.hdf5'
    strain, time, channel_dict = rl.loaddata(fileName)
    ts = time[1] - time[0]  # -- Time between samples
    fs = int(1.0 / ts)  # -- Sampling frequency
    segList = rl.dq_channel_to_seglist(channel_dict['DEFAULT'], fs)
    length = 16  # seconds
    strain_seg = strain[segList[0]][0:(length * fs)]
    time_seg = time[segList[0]][0:(length * fs)]
    plt.figure(0)
    plt.plot(time_seg - time_seg[0], strain_seg)
    plt.xlabel('Time since GPS ' + str(time_seg[0]))
    plt.ylabel('Strain')
    window = np.blackman(strain_seg.size)
    windowed_strain = strain_seg * window
    freq_domain = np.fft.rfft(windowed_strain) / fs
    freq = np.fft.rfftfreq(len(windowed_strain)) * fs
    plt.figure(1)
    plt.loglog(freq, abs(freq_domain))
    plt.axis([10, fs / 2.0, 1e-28, 1e-19])
    plt.grid('on')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Strain / Hz')
    Pxx, freqs = mlab.psd(strain_seg, Fs=fs, NFFT=fs)
    plt.figure(2)
    plt.loglog(freqs, Pxx)
    plt.axis([10, 2000, 1e-51, 1e-38])
    plt.grid('on')
    plt.ylabel('PSD')
    plt.xlabel('Freq (Hz)')
    NFFT = 1024
    window = np.blackman(NFFT)
    plt.figure(3)
    plt.title('Spectrograms')
    spec_power, freqs, bins, im = plt.specgram(strain_seg, NFFT=NFFT, Fs=fs,
                                               window=window)
    med_power = np.zeros(freqs.shape)
    norm_spec_power = np.zeros(spec_power.shape)
    index = 0
    for row in spec_power:
        med_power[index] = np.median(row)
        norm_spec_power[index] = row / med_power[index]
        index += 1

    plt.figure(4)
    plt.title('Normalized Spectrograms')
    plt.pcolormesh(bins, freqs, np.log10(norm_spec_power))
    plt.show()
else:
    train = pd.read_csv('training_labels.csv')
    if CFG.part == "train":
        label = train.loc[train["id"] == CFG.sample_id]["target"].values[0]
    else:
        label = None
    print(CFG.sample_id, ": ", label)
    fileName = get_file_path(CFG.sample_id)
    strain_seg = np.load(fileName).astype(np.float64)[CFG.channel]
    fs = 2048

    plt.figure()
    plt.plot(strain_seg)
    plt.xlabel('Time Unknown')
    plt.ylabel('Strain')

    plt.figure()
    window = np.blackman(strain_seg.size)
    windowed_strain = strain_seg * window
    freq_domain = np.fft.rfft(windowed_strain) / fs
    freq = np.fft.rfftfreq(len(windowed_strain)) * fs
    plt.loglog(freq, abs(freq_domain))
    plt.axis([10, fs / 2.0, 1e-26, 1e-20])
    plt.grid('on')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Strain / Hz')

    plt.figure()
    Pxx, freqs = mlab.psd(strain_seg, Fs=fs, NFFT=fs)
    plt.loglog(freqs, Pxx)
    plt.axis([10, 2000, 1e-49, 1e-40])
    plt.grid('on')
    plt.ylabel('PSD')
    plt.xlabel('Freq (Hz)')

    plt.figure()
    ts = pycbc.types.TimeSeries(strain_seg, epoch=0, delta_t=1.0 / 2048)
    std_cqt(ts, "cqt_pycbc_no_highpass")

    plt.figure()
    # whitened by tsd
    ts = whiten(ts)
    ts = pycbc.types.TimeSeries(ts, epoch=0, delta_t=1.0 / 2048)
    psd_origin = ts.psd(2)
    # max freq: 2048Hz
    # cut-off freq: 20Hz
    b, a = signal.butter(8, 20.0 / 1024.0, 'highpass')
    ts = signal.filtfilt(b, a, ts)
    ts = pycbc.types.TimeSeries(ts, epoch=0, delta_t=1.0 / 2048)
    psd_highpass = ts.psd(2)
    std_cqt(ts, "cqt_pycbc_no_whiten")

    plt.figure()
    # highpass
    ts = signal.filtfilt(b, a, strain_seg)
    ts = pycbc.types.TimeSeries(ts, epoch=0, delta_t=1.0 / 2048)
    # whitened by psd
    psd = interpolate(welch(ts), 1.0 / ts.duration)
    ts = (ts.to_frequencyseries() / psd ** 0.5).to_timeseries()
    psd_whitened = ts.psd(2)
    pylab.loglog(psd_origin.sample_frequencies, psd_origin, label='whitened by std')
    pylab.loglog(psd_highpass.sample_frequencies, psd_highpass, label='whitened by std & highpass')
    pylab.loglog(psd_whitened.sample_frequencies, psd_whitened, label='whitened by psd')
    pylab.legend()
    pylab.ylabel('$Strain^2 / Hz$')
    pylab.xlabel('Frequency (Hz)')
    pylab.grid()
    pylab.xlim(10, 1024)
    pylab.savefig(os.path.join(CFG.fig_save_path, "psd.png"), dpi=300)
    power = std_cqt(ts, "cqt_pycbc_whiten")
    np.save(os.path.join(CFG.fig_save_path, "power.npy"), power)

    # GWpy
    # whiten
    plt.figure()
    ts = TimeSeries(strain_seg, sample_rate=2048)
    ts = ts.whiten(window=("tukey", 0.15))
    std_gwpy_cqt(ts, "cqt_gwpy_whiten")

    # GWpy
    # no whiten
    ts = TimeSeries(strain_seg, sample_rate=2048)
    ts = ts.bandpass(20, 512, 0.05)
    std_gwpy_cqt(ts, "cqt_gwpy_no_whiten")

    # nnAudio
    from nnAudio.Spectrogram import CQT
    import torch

    transform = CQT(sr=2048,  # sample rate
                    fmin=20,  # min freq
                    fmax=1024,  # max freq
                    hop_length=64,  # hop length
                    pad_mode='reflect')
    strain_seg = ((strain_seg - np.min(strain_seg)) / (np.max(strain_seg) - np.min(strain_seg)) - 0.5) * 2.0
    power = transform(torch.from_numpy(strain_seg).float())
    power = power.numpy()[0, :, :]
    power = ((power - np.min(power)) / (np.max(power) - np.min(power)) * 255.0).astype(np.uint8)
    plt.figure()
    plt.pcolormesh(power)
    plt.title(CFG.sample_id)
    plt.savefig(os.path.join(CFG.fig_save_path, "cqt_nnaudio.png"), dpi=300)
