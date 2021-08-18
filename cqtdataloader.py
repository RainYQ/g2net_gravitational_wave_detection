import numpy as np
import matplotlib.pyplot as plt
import os

# GWpy
times = np.linspace(0.0, 2.0, 512)
freqs = np.linspace(20.0, 512.0, 512)


class CFG:
    clip = False
    out_dir_train = "train_cqt_power"
    out_dir_test = "test_cqt_power"
    cqt_power_prefix = "./"
    sample_id = 'eaaf7f02e1'


def std_cqt_show(times, freqs, powers):
    plt.figure()
    if CFG.clip:
        plt.pcolormesh(freqs, times, powers, vmax=15, vmin=0)
    else:
        plt.pcolormesh(freqs, times, powers)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.show()


powers = np.load(os.path.join(CFG.cqt_power_prefix, CFG.out_dir_train, CFG.sample_id + ".npy"))
std_cqt_show(times, freqs, powers[0])
std_cqt_show(times, freqs, powers[1])
std_cqt_show(times, freqs, powers[2])
plt.figure()
powers = np.moveaxis(powers, 0, -1)
powers = np.flip(powers, 0)
plt.imshow(powers)
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.show()
