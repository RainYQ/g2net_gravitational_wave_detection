# g2net_gravitational_wave_detection
# Results
## Part I CQT & CWT
| | CQT Fold 0 | CQT Fold 1 | CQT Fold 2 | CQT Fold 3 | CQT Fold 4 | CWT Fold 0 | CWT Fold 1 | CWT Fold 2 | CWT Fold 3 | CWT Fold 4 | CV / LB |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| CV | 0.8711 | 0.8730 | 0.8710 | 0.8713 | 0.8709 | 0.8722 | 0.8727 | 0.8720 | 0.8717 | 0.8716 |
| LB | 0.8736 | 0.8734 | 0.8733 | 0.8730 | 0.8726 | 0.8741 | 0.8741 | 0.8740 | 0.8735 | 0.8744 |
| CQT Ensemble | √ | √ | √ | √ | √ | | | | | | 0.87146 / - |
| CWT Ensemble | | | | | | √ | √ | √ | √ | √ | 0.87204 / - |
| Ensemble | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | 0.87175 / 0.8765 |

## Part II Combine CQT & CWT
| | Combine Fold 0 | Combine Fold 1 | Combine Fold 2 | Combine Fold 3 | Combine Fold 4 | CV / LB |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| CV | 0.8739 | 0.8746 | 0.8735 | 0.8739 | 0.8735 |
| LB | 0.8755 | 0.8755 | - | - | - |
| Combine Ensemble | √ | √ | √ | √ | √ | 0.87388 / 0.8764 |

## Prepare Step
1. Use [GitHub Desktop | Simple collaboration from your desktop](https://desktop.github.com/)
2. Clone https://github.com/RainYQ/g2net_gravitational_wave_detection.git
3. Download datasets from [G2Net Gravitational Wave Detection | Kaggle](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data)
4. Unzip g2net-gravitational-wave-detection.zip
5. **In PyCharm, Exclude train & test File Folder (!Warning)**
## kaggle dataset
* split data for CPU GPU TPU SEED 2020
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
* Local Update to 09/13
* kaggle notebook
  * CWT Update to 09/19
    * https://www.kaggle.com/rainyq/train-g2net-cwt
    * https://www.kaggle.com/rainyq/inference-cwt
  * CQT Update to 09/19
    * https://www.kaggle.com/rainyq/train-g2net-cqt
    * https://www.kaggle.com/rainyq/inference-cqt
  * pre-processing CQT Update to 08/19
    * https://www.kaggle.com/rainyq/train-g2net
* Google Colab Notebook
  * CWT Update to 09/19
    * https://colab.research.google.com/drive/1iQ3ezj2dsZ79MVKq9Y_P4Ha9aod3uERZ?usp=sharing
  * CQT Update to 09/19
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
* On RTX2060, EfficientNet B0, batch_size: 16, time: ~9500 s/epoch
* On TPU, EfficientNet B7, batch_size: 256, time: ~850s / first epoch, ~360s / other epochs
* 5-Fold
* Train Dataset
  * 448000 images
  * steps per epoch: 448000 / 256 // 4 = 437
  * repeat √
  * shuffle √
* Val Dataset
  * 112000 images
  * steps per epoch: 112000 / 256 = 438
  * repeat ×
  * shuffle ×
  * cache √
* learning_rate: 
  * 1e-3 for batch_size = 512/256/128
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
* LB ~= CV + 0.0015
* On RTX2060, batch_size: 64
* On RTX2060, ~35 min/fold
* On TPU, batch_size: 1024
* On TPU, CQT 10 min/fold CWT 12.5 min/fold

## STEP6: TODO
~~Sample~~ means finished <br/>
**Sample** means important <br/>
* ~~**Add mixup**~~
* ~~**Test CosineAnnealing learning rate strategy**~~
* ~~**Add label smooth**~~
* ~~**Add Cutout**~~
* Add Cutmix
* Compare different normalization styles
* **Use Combine Feature Extractor, train a better classifier**
  * Combine CQT CWT
  * Combine different parameters of CQT or CWT
* More Augmentations
* **~~Signal Augmentations~~**
  * ~~**Swap Channel**~~
  * ~~Time Shift~~
  * ~~Spector Shift~~
* **~~Image Augmentations~~**
  * ~~Gaussian Noise~~ std: 0.1
* ~~Use ROC_STAR_Loss https://github.com/iridiumblue/roc-star~~ 没啥好效果
* ~~**Add TTA** we can use large TTA_STEP~~ 提升非常小
* **Test ResNet RegNet(PyTorch)**
* ~~**Use Constant-Q Transform or Continuous Wavelet Transform**~~
* **Wave Denoise**
* Test ROF filter