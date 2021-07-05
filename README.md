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
    * hop_length = 512 (try 128/64)
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
***
* parse samples from tfrecords
  ```python
  def _parse_image_function(single_photo):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.io.parse_single_example(single_photo, image_feature_description)
  ```
***
* decode image data
* resize
* image standardization
* gray → rgb
* gaussian noise (closed)
* random contrast (closed)
* random hue (closed)
* random brightness (closed)
***
* get one hot label
* unbatch image data & label
***
  ```python
  def _preprocess_image_function(single_photo):
      image = tf.io.decode_raw(sample['data'], tf.float32)
      image = tf.reshape(image, [CFG.RAW_HEIGHT, CFG.RAW_WIDTH])
      image = tf.expand_dims(image, axis=-1)
      image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH])
      image = tf.image.per_image_standardization(image)
      image = (image - tf.reduce_min(image)) / (
              tf.reduce_max(image) - tf.reduce_min(image))
      image = tf.image.grayscale_to_rgb(image)
      single_photo['data'] = image
      return single_photo['data'], tf.cast(single_photo['label'], tf.float32)
  ```
***
## STEP4: Train
* On RTX2060, batch_size in train: 16
* On RTX2080Ti, batch_size in train: 32/64(oof)
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
                               warmup_proportion=0.3, min_lr=1e-6)
  ```
## STEP5: Predict
* Single Fold: LB 0.826
* On RTX2060, batch_size in predict: 64/128(oof)
* On RTX2060, 9.5 min / fold

## STEP6: TODO
* Test performance for Mel-Spec transformer based on tf
* **Test hop_length = 128 / 64**
* Add mixup
* **Test CosineAnnealing learning rate strategy**
* Add label smooth
* **Add image Augmentation : random_brightness ...**
* **Add sign augmentation : white noise ...**
* **Use ROC_STAR_Loss https://github.com/iridiumblue/roc-star**
* **Add TTA** we can use large TTA_STEP
* Test ResNet RegNet(PyTorch)