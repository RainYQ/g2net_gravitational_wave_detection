import tensorflow as tf
from tensorflow.keras import backend as K
import torch
import numpy as np


def roc_star_loss_tf(_y_true, y_pred, gamma, _epoch_true, epoch_pred):
    gamma = tf.cast(gamma, tf.float32)
    _y_true = tf.cast(_y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    _epoch_true = tf.cast(_epoch_true, tf.float32)
    epoch_pred = tf.cast(epoch_pred, tf.float32)
    y_true = (_y_true >= 0.50)
    epoch_true = (_epoch_true >= 0.50)
    if K.sum(tf.cast(y_true, tf.float32)) == 0 or K.sum(tf.cast(y_true, tf.float32)) == y_true.shape[0]:
        return K.sum(y_pred) * 1e-8
    pos = y_pred[y_true]
    neg = y_pred[~y_true]
    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]
    max_pos = 1000
    max_neg = 1000
    cap_pos = epoch_pos.shape[0]
    epoch_pos = epoch_pos[tf.random.uniform(epoch_pos.shape, minval=0, maxval=1, dtype=tf.float32) < max_pos / cap_pos]
    epoch_neg = epoch_neg[tf.random.uniform(epoch_neg.shape, minval=0, maxval=1, dtype=tf.float32) < max_neg / cap_pos]
    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]
    if ln_pos > 0:
        pos_expand = tf.reshape(tf.transpose(tf.tile(tf.reshape(pos, [1, pos.shape[0]]), [epoch_neg.shape[0], 1])),
                                pos.shape[0] * epoch_neg.shape[0])
        neg_expand = tf.reshape(tf.transpose(tf.tile(tf.reshape(epoch_neg, [epoch_neg.shape[0], 1]), [1, ln_pos])),
                                epoch_neg.shape[0] * ln_pos)
        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2 > 0]
        m2 = l2 * l2
    else:
        m2 = tf.constant([0], dtype=tf.float32)
    if ln_neg > 0:
        pos_expand = tf.reshape(tf.transpose(tf.tile(tf.reshape(epoch_pos, [1, epoch_pos.shape[0]]), [ln_neg, 1])),
                                epoch_pos.shape[0] * ln_neg)
        neg_expand = tf.reshape(tf.transpose(tf.tile(tf.reshape(neg, [neg.shape[0], 1]), [1, epoch_pos.shape[0]])),
                                neg.shape[0] * epoch_pos.shape[0])
        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3 > 0]
        m3 = l3 * l3
    else:
        m3 = tf.constant([0], dtype=tf.float32)
    if (K.sum(m2) + K.sum(m3)) != 0:
        res2 = K.sum(m2) / max_pos + K.sum(m3) / max_neg
    else:
        res2 = K.sum(m2) + K.sum(m3)
    res2 = tf.where(tf.math.is_nan(res2), tf.zeros_like(res2), res2)
    return res2


def roc_star_loss(_y_true, y_pred, gamma, _epoch_true, epoch_pred):
    """
    Nearly direct loss function for AUC.
    See article,
    C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
    https://github.com/iridiumblue/articles/blob/master/roc_star.md
        _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        gamma  : `Float` Gamma, as derived from last epoch.
        _epoch_true: `Tensor`.  Targets (labels) from last epoch.
        epoch_pred : `Tensor`.  Predicions from last epoch.
    """
    # convert labels to boolean
    y_true = (_y_true >= 0.50)
    epoch_true = (_epoch_true >= 0.50)

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true) == 0 or torch.sum(y_true) == y_true.shape[0]:
        return torch.sum(y_pred) * 1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]

    # Take random subsamples of the training set, both positive and negative.
    max_pos = 1000  # Max number of positive training samples
    max_neg = 1000  # Max number of positive training samples
    cap_pos = epoch_pos.shape[0]
    epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos / cap_pos]
    epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg / cap_pos]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    # sum positive batch elements agaionst (subsampled) negative elements
    if ln_pos > 0:
        pos_expand = pos.view(-1, 1).expand(-1, epoch_neg.shape[0]).reshape(-1)
        neg_expand = epoch_neg.repeat(ln_pos)

        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2 > 0]
        m2 = l2 * l2
    else:
        m2 = torch.tensor([0], dtype=torch.float).cuda()

    # Similarly, compare negative batch elements against (subsampled) positive elements
    if ln_neg > 0:
        pos_expand = epoch_pos.view(-1, 1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(epoch_pos.shape[0])

        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3 > 0]
        m3 = l3 * l3
    else:
        m3 = torch.tensor([0], dtype=torch.float).cuda()

    if (torch.sum(m2) + torch.sum(m3)) != 0:
        res2 = torch.sum(m2) / max_pos + torch.sum(m3) / max_neg
    else:
        res2 = torch.sum(m2) + torch.sum(m3)

    res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

    return res2


y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0], np.float32)
y_pred = np.array([0.8, 0.2, 0.9, 0.5, 0.3, 0.1, 0.45, 0.03], np.float32)
epoch_true = np.random.randint(0, 2, size=(16, 8))
epoch_pred = np.random.rand(16, 8)
result_torch = roc_star_loss(torch.from_numpy(y_true),
                             torch.from_numpy(y_pred),
                             0.5, torch.from_numpy(epoch_true),
                             torch.from_numpy(epoch_pred))
print(result_torch)
result_tf = roc_star_loss_tf(y_true, y_pred, 0.5, epoch_true, epoch_pred)
print(result_tf)
print("delta: ", result_torch.numpy() - result_tf.numpy())
