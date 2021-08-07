**# g2net_gravitational_wave_detection
## Prepare Step
1. Use [GitHub Desktop | Simple collaboration from your desktop](https://desktop.github.com/)
2. Clone https://github.com/RainYQ/g2net_gravitational_wave_detection.git
3. Download datasets from [G2Net Gravitational Wave Detection | Kaggle](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data)
4. Unzip g2net-gravitational-wave-detection.zip
5. **In PyCharm, Exclude train & test File Folder (!Warning)**
6. ~~Use 7-zip to unzip tfrecords.00*~~
## ~~STEP1: Transform to Spectrogram~~
* ~~Mel-Spectrogram Parameters~~
    * ~~sample_rate = 2048 Hz (fixed)~~
    * ~~n_mels = 128~~
    * ~~n_fft = 2048~~
    * ~~hop_length = 512 (try 128/64)~~
    * ~~mel_power = 2 (fixed)~~
    * ~~f_min = 20 Hz~~
    * ~~f_max = 1024 Hz~~
## STEP1: Constant Q Transform
* CQT Transform code <br/>
  ```python
  ts = TimeSeries(strain_seg, sample_rate=2048)
  ts = ts.whiten(window=("tukey", 0.15))
  cqt = ts.q_transform(qrange=(10, 10), frange=(20, 512), logf=True, whiten=False)
  # Use the same plot pipeline
  power = cqt.__array__()
  power = np.transpose(power)
  time = cqt.xindex.__array__()
  freq = cqt.yindex.__array__()
  plt.pcolormesh(time, freq, power, vmax=15, vmin=0)
  plt.yscale('log')
  ```
* Use ```cv2.resize``` resize to 512x512
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
***
* parse samples from tfrecords
  ```python
  def _parse_image_function(single_photo):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.io.parse_single_example(single_photo, image_feature_description)
  ```
***
* decode image data
* convert image datatype to ```tf.float32```
* resize
* image standardization
* random jpeg quality
* gaussian noise
* random contrast or random hue
* random brightness
***
* get one hot label
* unbatch image data & label
***
  ```python
  def _preprocess_image_function(single_photo):
    image = tf.image.decode_png(single_photo['data'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if CFG.RAW_WIDTH != CFG.WIDTH or CFG.RAW_HEIGHT != CFG.HEIGHT:
        image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    image = tf.image.random_jpeg_quality(image, 80, 100)
    # 高斯噪声的标准差为 0.3
    gau = tf.keras.layers.GaussianNoise(0.3)
    # 以 50％ 的概率为图像添加高斯噪声
    image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image), lambda: image)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.cond(tf.random.uniform([]) < 0.5,
                    lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
                    lambda: tf.image.random_hue(image, max_delta=0.3))
    # brightness随机调整
    image = tf.image.random_brightness(image, 0.3)
    single_photo['data'] = image
    return single_photo['data'], tf.cast(single_photo['label'], tf.float32)
  ```
***
## STEP4: Train
* On RTX2060, batch_size in train: 16
* On RTX2080Ti, batch_size in train: 32/64(oom)
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
* learning_rate: 
  * 1e-3 for batch_size = 512
  * 1e-4 for batch_size = 16/32/64
* loss
  ```python
  tf.keras.losses.BinaryCrossentropy()
  ```
* metrics
  ```python
  tf.keras.metrics.AUC(name='auc', num_thresholds=498)
  ```
* optimizer
  ```python
  tfa.optimizers.RectifiedAdam(lr=CFG.learning_rate, 
                               total_steps=CFG.epoch * CFG.iteration_per_epoch, 
                               warmup_proportion=0.1, min_lr=1e-5)
  ```
## STEP5: Predict
* Single Fold: LB 0.838
* Single Fold TTA_STEP = 16: LB 0.838
* Ensemble 2 Fold TTA_STEP = 16: LB 0.840
* CV ~= LB
* On RTX2060, batch_size in predict: 64/128(oom)
* On RTX2060, 9.5 min / fold (TTA OFF) 12.5 min / fold (TTA ON)
* TTA
  * No Gaussian Noise
  * random_contrast set (1.0, 1.3)
  * No random_jpeg_quality
  ```python
  def _preprocess_image_test_function(single_photo):
    image = tf.image.decode_png(single_photo['data'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if CFG.RAW_WIDTH != CFG.WIDTH or CFG.RAW_HEIGHT != CFG.HEIGHT:
        image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
    image = tf.image.per_image_standardization(image)
    image = tf.image.random_contrast(image, lower=1.0, upper=1.3)
    image = tf.cond(tf.random.uniform([]) < 0.5,
                    lambda: tf.image.random_saturation(image, lower=0.7, upper=1.3),
                    lambda: tf.image.random_hue(image, max_delta=0.3))
    # brightness随机调整
    image = tf.image.random_brightness(image, 0.3)
    single_photo['data'] = image
    return single_photo['data'], single_photo['id']
  ```

## STEP6: TODO
~~Sample~~ means finished <br/>
**Sample** means important <br/>
* ~~Add mixup~~
* **Test CosineAnnealing learning rate strategy**
* Add label smooth
* Add Cutout
* Add Cutmix
* Add Image Random Resize
* ~~**Add image Augmentation : random_brightness ...**~~
* ~~Use ROC_STAR_Loss https://github.com/iridiumblue/roc-star~~ 没啥好效果
* ~~**Add TTA** we can use large TTA_STEP~~ 提升非常小
* Test ResNet RegNet(PyTorch)
* ~~**Use Constant-Q Transform**~~
* **Wave Denoise**
* **Test ROF filter**