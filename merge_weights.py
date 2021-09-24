import tensorflow as tf
import numpy
import efficientnet.tfkeras as efn


class CFG:
    HEIGHT = 512
    WIDTH = 512
    k_fold = 5


def create_model():
    backbone = efn.EfficientNetB5(
        include_top=False,
        input_shape=(CFG.HEIGHT, CFG.WIDTH, 3),
        weights=None,
        pooling='avg'
    )

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')])
    return model


def transform_model():
    cqt_feature_extractor = efn.EfficientNetB5(
        include_top=False,
        input_shape=(CFG.HEIGHT, CFG.WIDTH, 3),
        weights=None,
        pooling='avg'
    )
    cqt_feature_extractor._name = 'cqt_efficientnet-b5'
    cwt_feature_extractor = efn.EfficientNetB5(
        include_top=False,
        input_shape=(CFG.HEIGHT, CFG.WIDTH, 3),
        weights=None,
        pooling='avg'
    )
    cwt_feature_extractor._name = 'cwt_efficientnet-b5'
    cqt_input = tf.keras.Input(shape=(CFG.HEIGHT, CFG.WIDTH, 3), name='cqt')
    cwt_input = tf.keras.Input(shape=(CFG.HEIGHT, CFG.WIDTH, 3), name='cwt')
    cqt_feature = cqt_feature_extractor(cqt_input)
    cwt_feature = cwt_feature_extractor(cwt_input)
    feature = tf.keras.layers.Concatenate(axis=-1)([cqt_feature, cwt_feature])
    drop_out_0 = tf.keras.layers.Dropout(0.5)(feature)
    dense_0 = tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu')(
        drop_out_0)
    drop_out_1 = tf.keras.layers.Dropout(0.5)(dense_0)
    dense_1 = tf.keras.layers.Dense(64, kernel_initializer=tf.keras.initializers.he_normal(), activation='relu')(
        drop_out_1)
    drop_out_2 = tf.keras.layers.Dropout(0.5)(dense_1)
    out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.he_normal(), activation='sigmoid')(
        drop_out_2)
    model = tf.keras.Model(inputs=[cqt_input, cwt_input], outputs=out)
    return model


for i in range(CFG.k_fold):
    model = create_model()
    model.load_weights('./model/CQT_5_Fold/model_best_%d.h5' % i)
    layers = model.layers
    weights_cqt = layers[0].get_weights()
    model.load_weights('./model/CWT_5_Fold/model_best_%d.h5' % i)
    layers = model.layers
    weights_cwt = layers[0].get_weights()
    model = transform_model()
    layers = model.layers
    layers[2].set_weights(weights_cqt)
    layers[3].set_weights(weights_cwt)
    model.save('./model/merge_model/model_best_%d.h5' % i)
    model.save_weights('./model/merge_weights/model_best_%d.h5' % i)


model = transform_model()
model.summary()
