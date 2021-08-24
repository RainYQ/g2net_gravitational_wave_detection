import numpy as np
import os
from matplotlib import pyplot as plt


class CFG:
    wave_data_prefix = "F:/"
    sample_id = '00000e74ad'
    mode = 'train'


def get_file_path(image_id, mode):
    return os.path.join(CFG.wave_data_prefix,
                        "{}/{}/{}/{}/{}.npy".format(mode, image_id[0], image_id[1], image_id[2], image_id))


data = np.load(get_file_path(CFG.sample_id, CFG.mode)).astype(np.float64)
data_scale = np.zeros_like(data)
for i in range(data.shape[0]):
    # Min Max Scaler -1 1
    data_scale[i, :] = (data[i, :] - np.min(data[i, :])) / (np.max(data[i, :]) - np.min(data[i, :]))
    data_scale[i, :] = (data_scale[i, :] - 0.5) * 2.0
    # Save Space
data_float32 = data_scale.astype(np.float32)
data_float32 = data_float32.astype(np.float64)

data_process = np.zeros_like(data_float32)
for i in range(data_scale.shape[0]):
    data_process[i, :] = data_float32[i, :] * 0.5 + 0.5
    data_process[i, :] = data_process[i, :] * (np.max(data[i, :]) - np.min(data[i, :])) + np.min(data[i, :])

plt.figure()
for i in range(data_scale.shape[0]):
    plt.plot(data_scale[i, :])
for i in range(data_scale.shape[0]):
    plt.plot((data_scale - data_float32)[i, :])
plt.show()
print(np.max((data_scale - data_float32) / data_scale))

plt.figure()
for i in range(data.shape[0]):
    plt.plot(data[i, :])
for i in range(data_process.shape[0]):
    plt.plot((data - data_process)[i, :])
plt.show()
error = data - data_process
print(np.max((data - data_process) / data))

