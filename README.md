# g2net_gravitational_wave_detection
## Prepare Step
1. Use [GitHub Desktop | Simple collaboration from your desktop](https://desktop.github.com/)
2. Clone https://github.com/RainYQ/g2net_gravitational_wave_detection.git
3. Download datasets from [G2Net Gravitational Wave Detection | Kaggle](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data)
4. Unzip g2net-gravitational-wave-detection.zip
5. **In PyCharm, Exclude train & test File Folder (!Warning)**
## STEP1: Constant Wave Transform
* kaggle dataset
  * Raw Data (Apply MinMaxScaler)
    * https://www.kaggle.com/rainyq/g2netrawdata-train
    * https://www.kaggle.com/rainyq/g2netrawdata-test
    * https://www.kaggle.com/rainyq/split-data
  * CQT Dataset (Apply Bandpass filter & whiten & MinMaxScaler & Resize to 512 x 512)
    * https://www.kaggle.com/rainyq/constant-q-transform-dataset-train
    * https://www.kaggle.com/rainyq/constant-q-transform-dataset-test
  * CQT Dataset (Apply Highpass filter & VminVmaxTransform & Resize to 512 x 512)
    * https://www.kaggle.com/rainyq/cqt-dataset-encode-jpeg-no-whiten
    * https://www.kaggle.com/rainyq/cqt-dataset-encode-png-test-no-whiten
  * CQT Dataset (Apply Highpass filter & whiten & VminVmaxTransform & Resize to 512 x 512)
    * https://www.kaggle.com/rainyq/cqt-dataset-encode-jpeg
    * https://www.kaggle.com/rainyq/cqt-dataset-encode-jpeg-test
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
* decode Raw signal data
* bandpass filter & whiten **TPU cannot apply bandpass filter**
* batch
* CWT & MinMaxScaler & Resize
* Image Augmentation
* **TPU Can only calculate CWT at batch_size <= 128**

## STEP4: Train
* On RTX2060, batch_size: 16, time: ~20500 s/epoch
* On RTX2080Ti, batch_size: 32/64(oom), time: Unknown
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
* CV ~= LB + 0.01
* On RTX2060, batch_size: 64/128(oom)
* On RTX2060, 147 min/fold (TTA OFF), Unknown min/fold (TTA ON)
* On TPU, CWT batch_size: 128, batch_size: 1024
* On TPU, 33 min/fold (TTA OFF), Unknown min/fold (TTA ON)
* TTA
  * No Gaussian Noise
  * random_contrast set (1.0, 1.3)
  * No random_jpeg_quality

## STEP6: TODO
~~Sample~~ means finished <br/>
**Sample** means important <br/>
* ~~Add mixup~~
* **Test CosineAnnealing learning rate strategy**
* Add label smooth
* ~~Add Cutout~~
* Add Cutmix
* ~~**Add image Augmentation : random_brightness ...**~~
* ~~Use ROC_STAR_Loss https://github.com/iridiumblue/roc-star~~ 没啥好效果
* ~~**Add TTA** we can use large TTA_STEP~~ 提升非常小
* Test ResNet RegNet(PyTorch)
* ~~**Use Constant-Q Transform**~~
* **Wave Denoise**
* **Test ROF filter**