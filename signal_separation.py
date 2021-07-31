import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from gwpy.timeseries import TimeSeries
from sklearn.decomposition import FastICA, PCA


def get_file_path(image_id):
    return "train/{}/{}/{}/{}.npy".format(image_id[0], image_id[1], image_id[2], image_id)


train = pd.read_csv('training_labels.csv')
samples = list(train[train['target'] == 0][:1000]['id'])
X = []
for id in samples:
    x = np.load(get_file_path(id)).astype(np.float64)[0]
    X.append(x)

X = np.transpose(np.array(X))
X /= X.std(axis=0)
ica = FastICA(n_components=5, max_iter=10000)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
print(X - np.dot(S_, A_.T))
# assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

pca = PCA(n_components=5)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
plt.figure()

models = [X, S_, H]
names = ['Source signals',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.savefig("noise_wave.png", dpi=300)

for i in range(5):
    plt.figure()
    ts = TimeSeries(np.transpose(S_)[i], sample_rate=2048)
    ts = ts.whiten(window=("tukey", 0.15))
    cqt = ts.q_transform(qrange=(10, 10), frange=(20, 512), logf=True, whiten=False)
    # Use the same plot pipeline
    power = cqt.__array__()
    power = np.transpose(power)
    time = cqt.xindex.__array__()
    freq = cqt.yindex.__array__()
    plt.pcolormesh(time, freq, power, vmax=15, vmin=0)
    plt.yscale('log')
    plt.savefig("noise_ica_" + str(i) + ".png", dpi=300)

for i in range(5):
    plt.figure()
    ts = TimeSeries(np.transpose(H)[i], sample_rate=2048)
    ts = ts.whiten(window=("tukey", 0.15))
    cqt = ts.q_transform(qrange=(10, 10), frange=(20, 512), logf=True, whiten=False)
    power = cqt.__array__()
    power = np.transpose(power)
    time = cqt.xindex.__array__()
    freq = cqt.yindex.__array__()
    plt.pcolormesh(time, freq, power, vmax=15, vmin=0)
    plt.yscale('log')
    plt.savefig("noise_pca_" + str(i) + ".png", dpi=300)
