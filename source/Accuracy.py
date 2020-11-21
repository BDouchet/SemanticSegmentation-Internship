"""
Network Inference (and store the predictions).

Calculation of the mIoU and Pixel Accuracy
""

import numpy as np
import cv2
import os
import tensorflow as tf
import tensorflow.keras.backend as K

#Renseigner le nom du modèle et le type de modèle
model='resunet v2'
NAME='resunet v2.1.h5'

checkpoint_path=model+'/models/'+NAME
mode='test' #mode={'train','test'}

dir='../Evaluation/'

if mode=='test':
    dir_img=dir+'test_images/'
    dir_predictions=model+'/predictions/'+os.path.splitext(NAME)[0]+'/'
    if not os.path.exists(dir_predictions):
        os.makedirs(dir_predictions)
    dir_gt=dir+'test_gt/'

if mode=='train':
    dir_img=dir+'x_train/'
    dir_predictions=model+'/predictions/'+os.path.splitext(NAME)[0]+'-train/'
    if not os.path.exists(dir_predictions):
        os.makedirs(dir_predictions)
    dir_gt=dir+'y_train/'


width=400
height=560
nbr_mask=10

def max_transfo(tab):
    out=np.zeros((height,width,3), dtype='uint8')
    for i in range(height):
        for j in range(width):
            out[i][j][2]=np.argmax(tab[i][j])
    return out

def relu6(x):
  return K.relu(x, max_value=6)

#ecrit les images prédites dans le dossier correspondant
def get_pred():
    model=tf.keras.models.load_model(checkpoint_path,custom_objects={'relu6':relu6},compile=False)
    for file in os.listdir(dir_img):
        img = cv2.resize(cv2.imread(dir_img+file), (width, height))
        to_predict=np.array([img])/255
        predictions=model.predict(to_predict)
        predictions=max_transfo(predictions[0])
        print(dir_predictions+str(file))
        cv2.imwrite(dir_predictions + str(file),predictions)

def calculate_full():
    intersection = np.zeros(nbr_mask) # int = (A and B)
    den = np.zeros(nbr_mask) # den = A + B = (A or B) + (A and B)
    pixel_accuracy=0
    size=0
    for file in os.listdir(dir_predictions):
        size+=1
        pred=cv2.imread(dir_predictions +file)[:,:,2]
        gt=cv2.resize(cv2.imread(dir_gt+file), (width, height),interpolation=cv2.INTER_NEAREST)[:,:,2]
        for i in range(height):
            for j in range(width):
                if pred[i][j]==gt[i][j]:
                    pixel_accuracy+=1
                    intersection[gt[i][j]]+=1
                den[pred[i][j]] += 1
                if gt[i][j]<=9:
                    den[gt[i][j]] += 1
                if gt[i][j]>10:
                    den[0]+=1
    mIoU = 0
    for i in range(nbr_mask):
        if den[i]!=0:
            mIoU+=intersection[i]/(den[i]-intersection[i])
        else:
            mIoU+=1
    mIoU=mIoU/nbr_mask
    PA=pixel_accuracy / (560 * 400*size)
    print("Total : PA = " + str(PA)+" ; mIoU = "+str(mIoU))

get_pred()
calculate_full()
