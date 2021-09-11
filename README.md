# g2net_gravitational_wave_detection
## Prepare Step
1. Use [GitHub Desktop | Simple collaboration from your desktop](https://desktop.github.com/)
2. Clone https://github.com/RainYQ/g2net_gravitational_wave_detection.git
3. Download datasets from [G2Net Gravitational Wave Detection | Kaggle](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data)
4. Unzip g2net-gravitational-wave-detection.zip
5. **In PyCharm, Exclude train & test File Folder (!Warning)**
## kaggle dataset
* split data for CPU GPU TPU
  * https://www.kaggle.com/rainyq/split-data
* **(New) Raw Data (Apply Bandpass filter)**
  * https://www.kaggle.com/rainyq/raw-datafloat64bandpass-train
  * https://www.kaggle.com/rainyq/raw-datafloat64bandpass-test
* **(New) Raw Data**
  * https://www.kaggle.com/rainyq/raw-datafloat64no-bandpass-train
  * https://www.kaggle.com/rainyq/raw-datafloat64no-bandpass-test
* Raw Data (Apply MinMaxScaler)
  * https://www.kaggle.com/rainyq/g2netrawdata-train
  * https://www.kaggle.com/rainyq/g2netrawdata-test
* Raw Data (Apply MinMaxScaler & Bandpass filter)
  * https://www.kaggle.com/rainyq/g2netrawdatabandpass-train
  * https://www.kaggle.com/rainyq/g2netrawdatabandpass-test
* CQT Dataset (Apply Bandpass filter & whiten & MinMaxScaler & Resize to 512 x 512)
  * https://www.kaggle.com/rainyq/constant-q-transform-dataset-train
  * https://www.kaggle.com/rainyq/constant-q-transform-dataset-test
* CQT Dataset (Apply Highpass filter & VminVmaxTransform & Resize to 512 x 512)
  * https://www.kaggle.com/rainyq/cqt-dataset-encode-jpeg-no-whiten
  * https://www.kaggle.com/rainyq/cqt-dataset-encode-png-test-no-whiten
* CQT Dataset (Apply Highpass filter & whiten & VminVmaxTransform & Resize to 512 x 512)
  * https://www.kaggle.com/rainyq/cqt-dataset-encode-jpeg
  * https://www.kaggle.com/rainyq/cqt-dataset-encode-jpeg-test
## STEP1: Continuous Wavelet Transform or Constant Q Transform
* Train data mean & var
  * channel 1
    * mean: 5.36416325e-27
    * var: 5.50707291e-41
  * channel 2
    * mean: 1.21596245e-25
    * var: 5.50458798e-41
  * channel3
    * mean: 2.37073866e-27
    * var: 3.37861660e-42
* Test data mean & var
  * channel 1
    * mean: 2.26719448e-25
    * var: 5.50354975e-41
  * channel 2
    * mean: -1.23312232e-25
    * var: 5.50793453e-41
  * channel3
    * mean: -5.39777633e-26
    * var: 3.38153083e-42
* Local Update to 09/11
* kaggle notebook
  * CWT Update to 09/08
    * https://www.kaggle.com/rainyq/train-g2net-cwt
    * https://www.kaggle.com/rainyq/inference-cwt
  * CQT Update to 09/11
    * https://www.kaggle.com/rainyq/train-g2net-cqt
    * https://www.kaggle.com/rainyq/inference-cqt
  * pre-processing CQT Update to 08/19
    * https://www.kaggle.com/rainyq/train-g2net
* Google Colab Notebook
  * CWT Update to 09/08
    * https://colab.research.google.com/drive/1iQ3ezj2dsZ79MVKq9Y_P4Ha9aod3uERZ?usp=sharing
  * CQT Update to 09/11
    * https://colab.research.google.com/drive/1-2IDAhoTasx7GnHndazHb-IlQq2_v-u5?usp=sharing
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
* whiten or no whiten
* batch
* CWT or CQT & Normalization & Resize
* Augmentation
  * Gaussian Noise
  * Cutout
  * Mixup
  * Shuffle Channel
* **TPU Can only calculate CWT at batch_size <= 64**

## STEP4: Train
* On RTX2060, batch_size: 16, time: ~9500 s/epoch
* On TPU, batch_size: 512, time: ~760s / first epoch, ~380s / other epochs
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
  * cache √
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
* LB ~= CV + 0.004
* On RTX2060, batch_size: 64
* On RTX2060, ~35 min/fold
* On TPU, batch_size: 1024
* On TPU, 24 min/fold

## STEP6: TODO
~~Sample~~ means finished <br/>
**Sample** means important <br/>
* ~~**Add mixup**~~
* ~~**Test CosineAnnealing learning rate strategy**~~
* ~~**Add label smooth**~~
* ~~**Add Cutout**~~ 没有观察到泛化能力提升
* Add Cutmix
* **Signal Augmentations**
  * ~~**Swap Channel**~~
  * ~~Time Shift~~
* **Image Augmentations**
  * ~~Gaussian Noise~~ std: 0.1
  * ~~random_brightness~~
  * ~~random_jpeg_quality~~ 不能一次作用于一个 batch
  * ~~random_contrast~~ 效果似乎变差了
  * ~~random_saturation~~ 效果似乎变差了
  * ~~random_hue~~ 效果似乎变差了
* ~~Use ROC_STAR_Loss https://github.com/iridiumblue/roc-star~~ 没啥好效果
* ~~**Add TTA** we can use large TTA_STEP~~ 提升非常小
* **Test ResNet RegNet(PyTorch)**
* ~~**Use Constant-Q Transform or Continuous Wavelet Transform**~~
* **Wave Denoise**
* Test ROF filter