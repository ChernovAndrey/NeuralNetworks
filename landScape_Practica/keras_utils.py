# -*- coding: utf-8 -*-
"""
Created on 2018.04.17

@author: poruss
"""

import numpy as np

import keras
from keras import backend as K


epsilon = K.epsilon()
def get_custom_crossentropy(k1 = 1.0, k2 = 1.0):
  def custom_crossentropy(target, output):
    output = K.clip(output, epsilon, 1 - epsilon)
    output = -k1*target*K.log(output)-k2*(1-target)*K.log(1-output)
    return output
  return custom_crossentropy


class LearningRateLogger(keras.callbacks.Callback):
  def on_train_begin(self, logs = {}):
    print('Optimizer:', self.model.optimizer.__class__.__name__, end = ' - ')
    print(self.model.optimizer.get_config())
    self.lr_logs = []
    if not hasattr(self.model.optimizer, 'lr'):
      print('Optimizer don\'t have a "lr" attribute.')
    if not hasattr(self.model.optimizer, 'decay'):
      print('Optimizer don\'t have a "decay" attribute.')
    if not hasattr(self.model.optimizer, 'iterations'):
      print('Optimizer don\'t have a "iterations" attribute.')
  def on_epoch_end(self, epoch, logs={}):
    lr = self.model.optimizer.lr
    decay = self.model.optimizer.decay
    iterations = self.model.optimizer.iterations
    lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
    self.lr_logs.append(K.eval(lr_with_decay))
    print(' - curent_lr: %.6f'%K.eval(lr_with_decay))

class LearningRateLoggerForNadam(keras.callbacks.Callback):
  def on_train_begin(self, logs = {}):
    print('Optimizer:', self.model.optimizer.__class__.__name__, end = ' - ')
    print(self.model.optimizer.get_config())
    self.lr_logs = []
    if not hasattr(self.model.optimizer, 'lr'):
      print('Optimizer don\'t have a "lr" attribute.')
  def on_epoch_end(self, epoch, logs={}):
    lr = self.model.optimizer.lr
    self.lr_logs.append(K.eval(lr))
    print(' - curent_lr: %.6f'%K.eval(lr))

def get_lr_logger(model):
  if model.optimizer.__class__.__name__ == 'Nadam':
    lr_logger = LearningRateLoggerForNadam()
  else:
    lr_logger = LearningRateLogger()
  return lr_logger


def get_lr_scheduler(lr_sched, batch_sched):
  if len(lr_sched) != len(batch_sched):
    print('len(lr_sched) != len(batch_sched)')
    return None
  lr_schedule = np.ones(np.sum(batch_sched), dtype = np.float32)
  lr_schedule[:batch_sched[0]] = np.full(batch_sched[0], lr_sched[0], dtype = np.float32)
  for i in range(1, len(batch_sched)):
    lr_schedule[np.sum(batch_sched[:i]):np.sum(batch_sched[:i+1])] = \
      np.full(batch_sched[i], lr_sched[i], dtype = np.float32)
  lr_schedule = lr_schedule.tolist()

  def scheduler(epoch):
    # print('\nScheduler set lr: %.6f'%lr_schedule[epoch])
    return lr_schedule[epoch]

  LRS = keras.callbacks.LearningRateScheduler(scheduler)
  return LRS


def get_model_checkpointer(ModelFileNameMask):
  MCheckpointer = keras.callbacks.ModelCheckpoint(ModelFileNameMask, monitor = 'val_loss', verbose = 0,
                                                save_best_only = False, save_weights_only = False, mode = 'auto', period = 1)
  return MCheckpointer

def get_layer_output(model, x_input, layers = 'all'): # для sequential     min(layers) = 0, для graphCNN =1(если будет ноль, то нулевой слой будет входом в сеть)
  if K.backend() == 'cntk':
    inp = model.input
    x_input = np.asarray(x_input)
    if x_input.shape == inp.shape:
      data_set = [x_input]
    elif x_input.shape[1:] == inp.shape:
      data_set = x_input
    else:
      print('check x_input shape, return None')
      return None
    data_set_outp = []
    n = len(model.layers)
    if layers == 'all':
      layers = range(0, n)
    if min(layers) < 0 or max(layers) > n:
      print('layers index out of range, return None')
      return None
    for i in layers:
      li = model.layers[i]
      l_outp = li.output
      functor = K.function([inp], [l_outp])
      data_set_out = functor([data_set])
      data_set_outp.append(np.asarray(data_set_out[0]))
    return data_set_outp
  else:
    print('%s backend!'%K.backend())
    print('get_layer_output return None')
    return None
#%%