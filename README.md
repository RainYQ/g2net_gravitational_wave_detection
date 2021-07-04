# g2net_gravitational_wave_detection
## Prepare Step
1. Use [GitHub Desktop | Simple collaboration from your desktop](https://desktop.github.com/)
2. Clone https://github.com/RainYQ/g2net_gravitational_wave_detection.git
3. Download datasets from [G2Net Gravitational Wave Detection | Kaggle](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data)
4. Unzip g2net-gravitational-wave-detection.zip
5. **In PyCharm, Exclude train & test File Folder (!Warning)**
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
* parse samples from tfrecords
```python
def _parse_image_function(single_photo):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(single_photo, image_feature_description)
```
* decode image data
* resize
* image standardization
* gray → rgb
* gaussian noise (closed)
* random contrast (closed)
* random hue (closed)
* random brightness (closed)
```python
def _preprocess_image_function(single_photo):
    image = tf.io.decode_raw(sample['data'], tf.float32)
    image = tf.reshape(image, [CFG.RAW_HEIGHT, CFG.RAW_WIDTH])
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    image = (image - tf.reduce_min(image)) / (
            tf.reduce_max(image) - tf.reduce_min(image)) * 255.0
    image = tf.image.grayscale_to_rgb(image)
    # image = tf.image.random_jpeg_quality(image, 80, 100)
    # 高斯噪声的标准差为 0.3
    gau = tf.keras.layers.GaussianNoise(0.3)
    # # 以 50％ 的概率为图像添加高斯噪声
    # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    # image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    # image = tf.cond(tf.random.uniform([]) < 0.5,
    #                 lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
    #                 lambda: tf.image.random_hue(image, max_delta=0.3))
    # # brightness随机调整
    # image = tf.image.random_brightness(image, 0.3)
    single_photo['data'] = image
    return single_photo
```
* get one hot label
* unbatch image data & label
```python
def _create_annot(single_photo):
    return single_photo['data'], tf.one_hot(single_photo['label'], 1)
```
## STEP4: Train
* 5-Fold
* Train Dataset
  * 448000 images
  * batch size: 512
  * steps per epoch: 448000 / 512 = 875
  * repeat √
  * shuffle √
* Val Dataset
  * 112000 images
  * batch size: 512
  * steps per epoch: 112000 / 512 = 219
  * repeat ×
  * shuffle ×
* learning_rate: 1e-3 for batch_size = 512
* loss: tf.keras.losses.BinaryCrossentropy()
* metrics: tf.keras.metrics.AUC(name='auc', num_thresholds=498)
* optimizer: tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)