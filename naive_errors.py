#file for custom losses and metrics
#import math
#import tensorflow as tf
#import numpy as np
import tensorflow.math as tfm
import tensorflow.keras.losses as tkl
import numpy as np
import tensorflow as tf

#functions here operate on pairs of single normal vectors

def simpleAngLossOld(y_true, y_pred):
    cosSim = tfm.negative(tkl.cosine_similarity(y_true, y_pred))
    loss = tfm.acos(cosSim)
    loss = loss.numpy()
    return loss

def simpleAngLoss(y_true, y_pred):
    loss = np.arccos(np.divide(np.sum(np.multiply(y_true, y_pred), axis=-1), np.multiply(np.linalg.norm(y_true, axis=-1), np.linalg.norm(y_pred, axis=-1))))
    return loss


def simpleAngErrorMetric(y_true, y_pred):
    cosSim = tfm.negative(tkl.cosine_similarity(y_true, y_pred))
    if tfm.greater(cosSim, 0):
        cosSim = tfm.minimum(1, cosSim)
    else:
        cosSim = tfm.maximum(-1, cosSim)
    error = tfm.acos(cosSim)
    error = tf.make_ndarray(error)
    return error
