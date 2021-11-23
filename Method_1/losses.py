import numpy as np
from tensorflow.keras import backend as K

def rmse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.), dtype="float32")
    y_true_f = K.batch_flatten(y_true * mask)
    y_pred_f = K.batch_flatten(y_pred * mask)
    mask_f = K.batch_flatten(mask)
    mse = K.sum(K.square(y_true_f - y_pred_f)) / K.sum(mask)
    return K.sqrt(mse)

def truemae(y_true, y_pred):
  output = []
  for i in range(y_true.shape[0]):
    yt = y_true[i].flatten()
    yp = y_pred[i].flatten()
    sum = 0.0
    tot = 0
    for i in range(yt.shape[0]):
      if yt[i] == -1:
        continue
      sum += abs(yt[i] - yp[i])
      tot += 1
    output.append( sum/tot)
  return np.mean(output)
    
def npme(y_true, y_pred):
    mask = y_true != 0.
    y_true_f = y_true * mask
    y_pred_f = y_pred * mask
    return  (np.sum(y_pred_f - y_true_f)/np.sum(mask))

def mae(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1.), dtype="float32")
    y_true_f = K.batch_flatten(y_true * mask)
    y_pred_f = K.batch_flatten(y_pred * mask)
    mask_f = K.batch_flatten(mask)
    return  K.mean(K.sum(K.abs(y_true_f - y_pred_f), axis=1)/K.sum(mask_f, axis=1))


