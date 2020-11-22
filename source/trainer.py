"""
- Load and train a model

- Use of a generator to yield data with HDF5 database

- Output a H5 file containing weights and optimizers parameters to continue the training

- Use Tensorboard to watch training data

- Implement two additionnal loss function : DiceLoss and Weighted categorical cross entropy

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
import h5py
import tensorflow.keras.backend as K

NAME = "resunetv2"
CONTINUE=False

tensorboard = TensorBoard(log_dir='..\\tensorboard\\logs\\{}'.format(NAME))

#hyperparameters
width = 400
height = 560
nbr_mask = 10
epochs=150
Activation='softmax'
neurons=16
batch_size=16
class_weights=np.array([1.,0.1,8.,18.,850.,271.,1267.,325.,1104.,6.])

dir_process=dir+"dataseth5.h5"
files=os.listdir(dir+dir_img)
datasetsize=len(files)
dataset=h5py.File(dir_process,'r')

if datasetsize%batch_size==0:
    step_ep=int(datasetsize/batch_size)
else:
    step_ep = int(datasetsize / batch_size)+1

def generator_shuffle():
    while True:
        keys=np.arange(850)
        np.random.shuffle(keys)
        for i in range(step_ep):
            batch_keys=keys[i*batch_size:(i+1)*batch_size]
            x_yield=[]
            y_yield=[]
            for j in batch_keys:
                x_yield.append((dataset['x_train'])[j])
                y_yield.append((dataset['y_train'])[j])
            yield np.array(x_yield), np.array(y_yield)

def DiceLoss(y_true, y_pred):
    alpha,beta = 0.5,0.5
    ones = K.ones(K.shape(y_true))
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true
    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))
    T = K.sum(num / den)
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T

def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true=K.cast(y_true,dtype='float32')
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

if not CONTINUE :
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    my_model.compile(optimizer='adam',
                     loss='categorical_crossentropy', metrics=['accuracy'])
    my_model.fit(generator_shuffle(), steps_per_epoch=step_ep,
                 epochs=epochs,
                 callbacks=[checkpoint,tensorboard])

if CONTINUE:
    my_model=tf.keras.models.load_model(checkpoint_path)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,tensorboard]
    my_model.compile(optimizer='adam',
                     loss='categorical_crossentropy', metrics=['accuracy'])

    my_model.fit(generator_shuffle(),
                 epochs=epochs,steps_per_epoch=step_ep,
                 callbacks=callbacks_list)

my_model.save(archi+'/models/'+NAME+'-final.h5',overwrite=True)

