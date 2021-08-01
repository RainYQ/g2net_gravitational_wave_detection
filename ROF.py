import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py

import readligo as rl


# Handford
fn_H1 = './LIGO_data/H-H1_LOSC_4_V1-1126259446-32.hdf5'
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
# Livingston
fn_L1 = './LIGO_data/L-L1_LOSC_4_V1-1126259446-32.hdf5'
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')

fs = 4096
time = time_H1
dt = time[1] - time[0]

# Template
NRtime, NR_H1 = np.genfromtxt('./LIGO_data/GW150914_4_NR_waveform.txt').transpose()


def iir_bandstops(fstops, fs, order=4):
    """ellip notch filter
    fstops is a list of entries of the form [frequency (Hz), df, df2]                           
    where df is the pass width and df2 is the stop width (narrower                              
    than the pass width). Use caution if passing more than one freq at a time,                  
    because the filter response might behave in ways you don't expect.
    """
    nyq = 0.5 * fs

    # Zeros zd, poles pd, and gain kd for the digital filter
    zd = np.array([])
    pd = np.array([])
    kd = 1

    # Notches
    for fstopData in fstops:
        fstop = fstopData[0]
        df = fstopData[1]
        df2 = fstopData[2]
        low = (fstop - df) / nyq
        high = (fstop + df) / nyq
        low2 = (fstop - df2) / nyq
        high2 = (fstop + df2) / nyq
        z, p, k = iirdesign([low, high], [low2, high2], gpass=1, gstop=6,
                            ftype='ellip', output='zpk')
        zd = np.append(zd, z)
        pd = np.append(pd, p)

    # Set gain to one at 100 Hz...better not notch there                                        
    bPrelim, aPrelim = zpk2tf(zd, pd, 1)
    outFreq, outg0 = freqz(bPrelim, aPrelim, 100 / nyq)

    # Return the numerator and denominator of the digital filter                                
    b, a = zpk2tf(zd, pd, k)
    return b, a


def get_filter_coefs(fs):
    # assemble the filter b,a coefficients:
    coefs = []

    # bandpass filter parameters
    lowcut = 43
    highcut = 2047
    order = 4

    # bandpass filter coefficients 
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    bb, ab = butter(order, [low, high], btype='band')
    coefs.append((bb, ab))

    # Frequencies of notches at known instrumental spectral line frequencies.
    # You can see these lines in the ASD above, so it is straightforward to make this list.
    notchesAbsolute = np.array(
        [14.0, 34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00,
         120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])

    # notch filter coefficients:
    for notchf in notchesAbsolute:
        bn, an = iir_bandstops(np.array([[notchf, 1, 0.1]]), fs, order=4)
        coefs.append((bn, an))

    # Manually do a wider notch filter around 510 Hz etc.          
    bn, an = iir_bandstops(np.array([[510, 200, 20]]), fs, order=4)
    coefs.append((bn, an))

    # also notch out the forest of lines around 331.5 Hz
    bn, an = iir_bandstops(np.array([[331.5, 10, 1]]), fs, order=4)
    coefs.append((bn, an))

    return coefs


# Find the coefficients
coefs = get_filter_coefs(fs)


def filter_data(data_in, coefs):
    data = data_in.copy()
    for coef in coefs:
        b, a = coef
        # filtfilt applies a linear filter twice, once forward and once backwards.
        # The combined filter has linear phase.
        data = filtfilt(b, a, data)
    return data

strain_H1_filt = filter_data(strain_H1, coefs)
strain_L1_filt = filter_data(strain_L1, coefs)
# It is necessary to make a shift in L1
strain_L1_shift = -np.roll(strain_L1_filt, int(0.008 * fs))

tevent = 1126259462.422
plt.figure()
plt.plot(time - tevent, strain_H1_filt, 'r', label='H1 strain')
plt.plot(time - tevent, strain_L1_shift, 'g', label='L1 strain')
plt.xlabel('time (s) since ')
plt.ylabel('strain')
plt.legend(loc='lower left')
plt.title('Bandpassing+notching filtering')
plt.show()

strain_H1_TV = gs1(strain_H1_filt, 1, 1, 0.2, 0.01)
strain_L1_TV = gs1(strain_L1_filt, 1, 1, 0.2, 0.01)
strain_L1_shift = -np.roll(strain_L1_TV, int(0.008 * fs))

# The same steps with the template
strain_NR_filt = filter_data(NR_H1, coefs)
strain_NR_TV = gs1(strain_NR_filt, 1, 1, 0.2, 0.01)

plt.figure()
plt.plot(time - tevent, strain_H1_TV, 'r', label='H1 strain')
plt.plot(time - tevent, strain_L1_shift, 'g', label='L1 strain')
plt.plot(NRtime + 0.002, strain_NR_TV, 'k', label='NR strain')
plt.xlim([-0.15, 0.05])
plt.ylim([-1e-21, 1e-21])
plt.xlabel('time (s) since ')
plt.ylabel('strain')
plt.legend(loc='lower left')
plt.title('ROF filter')
plt.show()

# Fourier transform
NFFT = 1 * fs
fmin = 10
fmax = 2000
Pxx_H1, freqs = mlab.psd(strain_H1, Fs=fs, NFFT=NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs=fs, NFFT=NFFT)
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)
bb, ab = butter(4, [20. * 2. / fs, 300. * 2. / fs], btype='band')

# Filter application
strain_H1_TV1 = filtfilt(bb, ab, strain_H1_TV)
strain_L1_TV1 = filtfilt(bb, ab, strain_L1_TV)
strain_L1_shift = -np.roll(strain_L1_TV1, int(0.008 * fs))
strain_NR_TV1 = filtfilt(bb, ab, strain_NR_TV)

# Plot
tevent = 1126259462.422
plt.figure()
plt.plot(time - tevent, strain_H1_TV1, 'r', label='H1 strain')
plt.plot(time - tevent, strain_L1_shift, 'g', label='L1 strain')
plt.plot(NRtime + 0.002, strain_NR_TV1, 'k', label='NR strain')
plt.xlim([-0.15, 0.05])
plt.ylim([-1e-21, 1e-21])
plt.xlabel('time (s) since ')
plt.ylabel('strain')
plt.legend(loc='lower left')
plt.title('Total Variation denoising')
plt.show()

tevent = 1126259462.422
deltat = 10  # seconds around the event
indxt = np.where((time_H1 >= tevent - deltat) & (time_H1 < tevent + deltat))
# pick a shorter FTT time interval, like 1/8 of a second:
NFFT = fs // 8  # fs= 4096 s
NOVL = NFFT * 15 // 16
# and choose a window that minimizes "spectral leakage" 
window = np.blackman(NFFT)

# Color
spec_cmap = 'inferno'

# H1 Spectrogram
plt.figure()
spec_H1, freqs, bins, im = plt.specgram(strain_H1_TV1[indxt], NFFT=NFFT, Fs=fs, window=window,
                                        noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat, deltat], vmin=-550, vmax=-440)
plt.xlabel('time (s) since ' + str(tevent))
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.axis([-0.5, 0.5, 0, 500])
plt.title('aLIGO H1 strain data near GW150914')

# L1 Spectrogram
plt.figure()
spec_H1, freqs, bins, im = plt.specgram(strain_L1_TV1[indxt], NFFT=NFFT, Fs=fs, window=window,
                                        noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat, deltat], vmin=-550, vmax=-440)
plt.xlabel('time (s) since ' + str(tevent))
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.axis([-0.5, 0.5, 0, 500])
plt.title('aLIGO L1 strain data near GW150914')
plt.show()

#  Time-domain filtering - Bandpassing+notching
coefs = get_filter_coefs(fs)

iterations = 10  # Number of iterations

# H1
strain_H1_filt = filter_data(strain_H1, coefs)
strain_H1_TV = gs1(strain_H1_filt, 1, 1, 2, 0.01)
i = 0
while i < iterations:
    strain_H1_TV = gs1(strain_H1_TV, 1, 1, 2, 0.01)
    i += 1

# L1
strain_L1_filt = filter_data(strain_L1, coefs)
strain_L1_TV = gs1(strain_L1_filt, 1, 1, 2, 0.01)
i = 0
while i < iterations:
    strain_L1_TV = gs1(strain_L1_TV, 1, 1, 2, 0.01)
    i += 1
strain_L1_shift = -np.roll(strain_L1_TV, int(0.008 * fs))

# Template
strain_NR_filt = filter_data(NR_H1, coefs)
strain_NR_TV = gs1(strain_NR_filt, 1, 1, 0.2, 0.01)
strain_NR_TV1 = filtfilt(bb, ab, strain_NR_TV)

# Plot
tevent = 1126259462.422
plt.figure()
plt.plot(time - tevent, strain_H1_TV, 'r', label='H1 strain')
plt.plot(time - tevent, strain_L1_shift, 'g', label='L1 strain')
plt.plot(NRtime + 0.002, strain_NR_TV, 'k', label='NR strain')
plt.xlim([-0.15, 0.05])
plt.ylim([-1e-21, 1e-21])
plt.xlabel('time (s) since ')
plt.ylabel('strain')
plt.legend(loc='lower left')
plt.title('Total Variation denoising with iterations')
plt.show()

#  Time-domain filtering - Bandpassing+notching
coefs = get_filter_coefs(fs)
strain_H1_filt = filter_data(strain_H1, coefs)
strain_H1_TV1 = gs1(strain_H1_filt, 1, 1, 2, 0.01)
strain_H1_TV2 = gs1(strain_H1_TV1, 1, 1, 2, 0.01)
strain_H1_TV3 = gs1(strain_H1_TV2, 1, 1, 2, 0.01)
strain_H1_TV4 = gs1(strain_H1_TV3, 1, 1, 2, 0.01)
strain_H1_TV5 = gs1(strain_H1_TV4, 1, 1, 2, 0.01)
strain_H1_TV6 = gs1(strain_H1_TV5, 1, 1, 2, 0.01)
strain_H1_TV7 = gs1(strain_H1_TV6, 1, 1, 2, 0.01)
strain_H1_TV8 = gs1(strain_H1_TV7, 1, 1, 2, 0.01)
strain_H1_TV9 = gs1(strain_H1_TV8, 1, 1, 2, 0.01)
strain_H1_TV10 = gs1(strain_H1_TV9, 1, 1, 2, 0.01)

# Plot
tevent = 1126259462.422
plt.figure()
plt.plot(time - tevent, strain_H1_TV1, label='H1 strain', color='#d4daff')
plt.plot(time - tevent, strain_H1_TV2, label='H1 strain', color='#bac4ff')
plt.plot(time - tevent, strain_H1_TV3, label='H1 strain', color='#a1aeff')
plt.plot(time - tevent, strain_H1_TV4, label='H1 strain', color='#8798ff')
plt.plot(time - tevent, strain_H1_TV5, label='H1 strain', color='#6e82ff')
plt.plot(time - tevent, strain_H1_TV6, label='H1 strain', color='#546dff')
plt.plot(time - tevent, strain_H1_TV7, label='H1 strain', color='#3b57ff')
plt.plot(time - tevent, strain_H1_TV8, label='H1 strain', color='#2141ff')
plt.plot(time - tevent, strain_H1_TV9, label='H1 strain', color='#082bff')
plt.plot(time - tevent, strain_H1_TV10, '#2c328c', label='H1 strain')
plt.xlim([-0.15, 0.05])
plt.ylim([-1e-21, 1e-21])
plt.xlabel('time (s) since ')
plt.ylabel('strain')
plt.title('ROF iterations')
plt.show()

# -- SET ME   Tutorial should work with most binary black hole events
# -- Default is no event selection; you MUST select one to proceed.
eventname = ''
# eventname = 'GW150914' 
eventname = 'GW151226'
# eventname = 'LVT151012'

# want plots?
make_plots = 1
plottype = "png"
# plottype = "pdf"
# -- SET ME   Tutorial should work with most binary black hole events
# -- Default is no event selection; you MUST select one to proceed.
eventname = ''
# eventname = 'GW150914' 
eventname = 'GW151226'
# eventname = 'LVT151012'

# want plots?
make_plots = 1
plottype = "png"
# plottype = "pdf"
# Standard python numerical analysis imports:
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json

# the IPython magic below must be commented out in the .py file, since it doesn't work there.
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# LIGO-specific readligo.py 
import readligo as rl

# you might get a matplotlib warning here; you can ignore it.
# Read the event properties from a local json file
fnjson = "BBH_events_v2.json"
try:
    events = json.load(open(fnjson, "r"))
except IOError:
    print("Cannot find resource file " + fnjson)
    print("You can download it from https://losc.ligo.org/s/events/" + fnjson)
    print("Quitting.")
    quit()

# did the user select the eventname ?
try:
    events[eventname]
except:
    print('You must select an eventname that is in ' + fnjson + '! Quitting.')
    quit()
# Extract the parameters for the desired event:
event = events[eventname]
fn_H1 = event['fn_H1']  # File name for H1 data
fn_L1 = event['fn_L1']  # File name for L1 data
fn_template = event['fn_template']  # File name for template waveform
fs = event['fs']  # Set sampling rate
tevent = event['tevent']  # Set approximate event GPS time
fband = event['fband']  # frequency band for bandpassing signal
# ----------------------------------------------------------------
# Load LIGO data from a single file.
# FIRST, define the filenames fn_H1 and fn_L1, above.
# ----------------------------------------------------------------
try:
    # read in data from H1 and L1, if available:
    strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
    strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')
except:
    print("Cannot find data files!")
    print("You can download them from https://losc.ligo.org/s/events/" + eventname)
    print("Quitting.")
    quit()

# both H1 and L1 will have the same time vector, so:
time = time_H1
# the time sample interval (uniformly sampled!)
dt = time[1] - time[0]

make_psds = 1
if make_psds:
    # number of sample for the fast fourier transform:
    NFFT = 4 * fs
    Pxx_H1, freqs = mlab.psd(strain_H1, Fs=fs, NFFT=NFFT)
    Pxx_L1, freqs = mlab.psd(strain_L1, Fs=fs, NFFT=NFFT)

    # We will use interpolations of the ASDs computed above for whitening:
    psd_H1 = interp1d(freqs, Pxx_H1)
    psd_L1 = interp1d(freqs, Pxx_L1)

    # Here is an approximate, smoothed PSD for H1 during O1, with no lines. We'll use it later.    
    Pxx = (1.e-22 * (18. / (0.1 + freqs)) ** 2) ** 2 + 0.7e-23 ** 2 + ((freqs / 2000.) * 4.e-23) ** 2
    psd_smooth = interp1d(freqs, Pxx)

# read in the template (plus and cross) and parameters for the theoretical waveform
try:
    f_template = h5py.File(fn_template, "r")
except:
    print("Cannot find template file!")
    print("You can download it from https://losc.ligo.org/s/events/" + eventname + '/' + fn_template)
    print("Quitting.")
    quit()
# extract metadata from the template file:
template_p, template_c = f_template["template"][...]
t_m1 = f_template["/meta"].attrs['m1']
t_m2 = f_template["/meta"].attrs['m2']
t_a1 = f_template["/meta"].attrs['a1']
t_a2 = f_template["/meta"].attrs['a2']
t_approx = f_template["/meta"].attrs['approx']
f_template.close()
# the template extends to roughly 16s, zero-padded to the 32s data length. The merger will be roughly 16s in.
template_offset = 16.

# -- To calculate the PSD of the data, choose an overlap and a window (common to all detectors)
#   that minimizes "spectral leakage" https://en.wikipedia.org/wiki/Spectral_leakage
NFFT = 4 * fs
psd_window = np.blackman(NFFT)
# and a 50% overlap:
NOVL = NFFT / 2

# define the complex template, common to both detectors:
template = (template_p + template_c * 1.j)
# We will record the time where the data match the END of the template.
etime = time + template_offset
# the length and sampling rate of the template MUST match that of the data.
datafreq = np.fft.fftfreq(template.size) * fs
df = np.abs(datafreq[1] - datafreq[0])

# to remove effects at the beginning and end of the data stretch, window the data
# https://en.wikipedia.org/wiki/Window_function#Tukey_window
try:
    dwindow = signal.tukey(template.size, alpha=1. / 8)  # Tukey window preferred, but requires recent scipy version
except:
    dwindow = signal.blackman(template.size)  # Blackman window OK if Tukey is not available

# prepare the template fft.
template_fft = np.fft.fft(template * dwindow) / fs

# loop over the detectors
dets = ['H1', 'L1']
for det in dets:

    if det is 'L1':
        data = strain_L1.copy()
    else:
        data = strain_H1.copy()

    # -- Calculate the PSD of the data.  Also use an overlap, and window:
    data_psd, freqs = mlab.psd(data, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data * dwindow) / fs

    # -- Interpolate to get the PSD values at the needed frequencies
    power_vec = np.interp(np.abs(datafreq), freqs, data_psd)

    # -- Calculate the matched filter output in the time domain:
    # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
    # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
    # so the result will be plotted as a function of time off-set between the template and the data:
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2 * np.fft.ifft(optimal) * fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1 * (template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time / sigma

    # shift the SNR vector by the template length so that the peak is at the END of the template
    peaksample = int(data.size / 2)  # location of peak in the template
    SNR_complex = np.roll(SNR_complex, peaksample)
    SNR = abs(SNR_complex)

    # find the time and SNR value at maximum:
    indmax = np.argmax(SNR)
    timemax = time[indmax]
    SNRmax = SNR[indmax]

    # Calculate the "effective distance" (see FINDCHIRP paper for definition)
    # d_eff = (8. / SNRmax)*D_thresh
    d_eff = sigma / SNRmax
    # -- Calculate optimal horizon distnace
    horizon = sigma / 8

    # Extract time offset and phase at peak
    phase = np.angle(SNR_complex[indmax])
    offset = (indmax - peaksample)
    # print offset

    # apply time offset, phase, and d_eff to template, for plotting
    template_ = (template_p + template_c * 1.j)
    template_phaseshifted = np.real(template_ * np.exp(1j * phase))
    template_match = np.roll(template_phaseshifted, offset) / d_eff

from skimage.measure import compare_ssim as ssim

def extract_patches_1d(A, patch_size, max_patches=None, random_state=0, step=1):
    m, n = np.atleast_2d(A).shape
    if max_patches is None:
        numOfcols = (n - patch_size) // step + 1
        col = 0
        if m == 1:
            D = np.zeros([patch_size, numOfcols])
            for i in range(0, numOfcols * step, step):
                D[:, col] = A[i: i + patch_size]
                col += 1
        else:
            D = np.zeros([patch_size, numOfcols * n])
            for j in range(n):
                for i in range(numOfcols):
                    D[:, col] = A[i:i + patch_size, j]
                    col += 1
    else:
        D = np.zeros([patch_size, max_patches])
        np.random.seed(random_state)
        for k in range(max_patches):
            i = np.random.randint(0, m - 2)
            j = np.random.randint(0, n)
            D[:, k] = A[i:i + patch_size, j]

    return D


def reconstruct_from_patches_1d(patches, signal_len):
    m, n = patches.shape
    step = (signal_len - m) // (n - 1)
    y = np.zeros(signal_len)
    aux = np.zeros(signal_len)
    for i in range(n):
        y[i * step:i * step + m] += patches[:, i]
        aux[i * step:i * step + m] += np.ones(m)
    return y / aux


coefs = get_filter_coefs(fs)
# H1
strain_H1_filt = filter_data(strain_H1, coefs)
# L1
strain_L1_filt = filter_data(strain_L1, coefs)

# H1
strain_H1_cut = extract_patches_1d(strain_H1_filt, 16384, None, 0, 8192)
f, c = np.atleast_2d(strain_H1_cut).shape  # This value will be useful
# L1
strain_L1_cut = extract_patches_1d(strain_L1_filt, 16384, None, 0, 8192)
# Template
template_cut = extract_patches_1d(template_match, 16384, None, 0, 8192)


iterations = 10  # Number of ROF iterations
mu = np.arange(1, 11, 1)  # Array of possibles mu
test = []  # Array of SSIM values
# H1
for x in mu:
    s = np.zeros(f)  # Auxiliary variable
    u = np.zeros(f)
    j = 0
    while j < iterations:
        s = gs1(strain_H1_cut[:, 7], 1, 1, x, 0.01)
        u = gs1(template_cut[:, 7], 1, 1, x, 0.01)
        j += 1
    test.append(ssim(u, s, data_range=s.max() - s.min()))
# Once applied all the possibles µ, we will use the best one
j = 0
while j < iterations:
    strain_H1_cut[:, 7] = gs1(strain_H1_cut[:, 7], 1, 1, mu[test.index(max(test))], 0.01)
    template_cut[:, 7] = gs1(template_cut[:, 7], 1, 1, mu[test.index(max(test))], 0.01)
    j += 1
strain_H1_TV = reconstruct_from_patches_1d(strain_H1_cut, len(strain_H1))  # Reconstruct the strain
strain_H1_TV_filt = filtfilt(bb, ab, strain_H1_TV)  # Extra bandpassing

# L1
for x in mu:
    s = np.zeros(f)  # Auxiliar
    u = np.zeros(f)
    j = 0
    while j < iterations:
        s = gs1(strain_L1_cut[:, 7], 1, 1, x, 0.01)
        u = gs1(template_cut[:, 7], 1, 1, x, 0.01)
        j += 1
    test.append(ssim(u, s, data_range=s.max() - s.min()))
# Once applied all the possibles µ, we will use the best one
j = 0
while j < iterations:
    strain_L1_cut[:, 7] = gs1(strain_L1_cut[:, 7], 1, 1, mu[test.index(max(test))], 0.01)
    template_cut[:, 7] = gs1(template_cut[:, 7], 1, 1, mu[test.index(max(test))], 0.01)
    j += 1
strain_L1_TV = reconstruct_from_patches_1d(strain_L1_cut, len(strain_L1))  # Reconstruct the strain
strain_L1_TV_filt = filtfilt(bb, ab, strain_L1_TV)  # Extra bandpassing

# H1
strain_H1_TV_filt = strain_H1_TV_filt / strain_H1_TV_filt.max()
# L1
strain_L1_TV_filt = strain_L1_TV_filt / strain_L1_TV_filt.max()
# Template
template_TV = reconstruct_from_patches_1d(template_cut, len(strain_H1))
template_TV = template_TV / template_TV.max()
# Plot
plt.figure()
plt.plot(time - tevent, strain_H1_TV_filt, 'r', label='H1 strain')
plt.plot(time - tevent, strain_L1_TV_filt, 'g', label='L1 strain')
plt.plot(time - tevent, template_TV, 'k', label='Template')
plt.xlabel('Time (s)')
plt.ylabel('strain')
plt.legend(loc='upper left')
plt.title('Filtered data ')
plt.show()

