# g2net_gravitational_wave_detection
## Prepare Step
1. Use [GitHub Desktop | Simple collaboration from your desktop](https://desktop.github.com/)
2. Clone https://github.com/RainYQ/g2net_gravitational_wave_detection.git
3. Download datasets from [G2Net Gravitational Wave Detection | Kaggle](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data)
4. Unzip g2net-gravitational-wave-detection.zip
5. **In Pycharm, Exclude train & test File Folder (!Warning)**
6. Use 7-zip to unzip tfrecords.00*
## STEP1: Transform to Spectrogram
* Mel-Spectrogram Parameters
    * sample_rate = 2048 Hz (fixed)
    * n_mels = 128
    * n_fft = 2048
    * hop_length = 512
    * mel_power = 2 (fixed)
    * f_min = 20 Hz
    * f_max = 1024 Hz
## STEP2: Make TFRecords
* train <br/>
```python
features = tf.train.Features(feature={
                'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            })
```
* test <br/>
```python
features = tf.train.Features(feature={
                'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw]))
            })
```
## STEP3: Make tf.datasets