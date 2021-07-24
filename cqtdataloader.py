import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(0.0, 2.0, 1000)
freqs = np.linspace(20.0, 512.0, 274)


def std_cqt_show(times, freqs, powers, use_vmax_vmin):
    plt.figure()
    if(use_vmax_vmin):
        plt.pcolormesh(freqs, times, powers, vmax=15, vmin=0)
    else:
        plt.pcolormesh(freqs, times, powers)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.show()


powers = np.load("power.npy").astype(np.float64)
std_cqt_show(times, freqs, powers, True)
# Normalization
powers = (powers - np.min(powers)) / (np.max(powers) - np.min(powers)) * 255.0
powers = powers.astype(np.uint8)
std_cqt_show(times, freqs, powers, False)

