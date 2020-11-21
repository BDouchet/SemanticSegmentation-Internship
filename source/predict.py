"""
Network inference

Comparison between ground truth and predicition to make a visual analysis

"""

import cv2
import tensorflow as tf
import numpy as np
import os
import time
import tensorflow.keras.backend as K

# necessary for deeplabv3plus
def relu6(x):
  return K.relu(x, max_value=6)

load = tf.keras.models.load_model(checkpoint_path, custom_objects={'relu6':relu6}, compile=False)

mode='train' #mode={'train','test'}

if mode=='test':
  dir = '../Evaluation/'
  dir_img = dir + 'test_images/'
  dir_masks = dir + 'test_gt/'

if mode=='train':
  dir = '../dataset/dataset85/'
  dir_img = dir + 'images/'
  dir_masks = dir + 'masks/'

nbr_mask=10
height=560
width=400

def max_transfo(tab):
  out=np.zeros((height,width,nbr_mask))
  for i in range(height):
    for j in range(width):
      out[i][j][np.argmax(tab[i][j])]=255
  return out

code_couleur_BGR = [[70, 70, 70], [200, 200, 200], [200, 0, 0], [0, 0, 200], [0, 200, 0], [200, 0, 200],
                      [200, 200, 0], [0, 200, 200], [0, 100, 255], [50, 0, 150]]
labels = ['class1', 'class2, 'class3 'cass4 'class5, 'class6,'class7, 'class8, 'class9','class10']

#visualize all the train dataset
def comparaison_all():
  for file in os.listdir(dir_img):
    comparaison_one(file)

#visualize one image from train dataset (ex : file = '3.tiff')
def comparaison_one(file):
  filename = dir_img + file
  filegt = dir_masks + file
  img = cv2.resize(cv2.imread(filename), (width, height))
  groundtruth = cv2.resize(cv2.imread(filegt), (width, height), interpolation=cv2.INTER_NEAREST)[:, :, 2]
  
  img_to_pred=np.array([img]) / 255
  start=time.time()
  prediction = load.predict(img_to_pred)
  print(time.time()-start)
  prediction = max_transfo(prediction[0])
  img_prediction = np.copy(img)
  img_gt = np.copy(img)
   
  #display of the legend     
  legend = np.zeros((height, 200, 3), dtype='uint8')
  for i in range(nbr_mask):
    cv2.rectangle(legend, (0, (i * 45)), (200, (i * 45) + 45), tuple(code_couleur_BGR[i]), -1)
    cv2.putText(legend, labels[i], (0, (i * 45) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
          
          
  filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  for m in filter:
    mask = prediction[:, :, m]
    img_prediction[:, :][mask[:, :] == 255] = code_couleur_BGR[m]
    img_gt[:, :][groundtruth[:, :] == m] = code_couleur_BGR[m]
  img[:, -1] = [0, 0, 0]
  img_gt[:, -1] = [0, 0, 0]
  img_prediction[:, -1] = [0, 0, 0]
  first_img = np.concatenate((img, img_gt), axis=1)
  second_img = np.concatenate((img_prediction, legend), axis=1)
  cv2.imshow("Visualization for image n: " + str(filename), np.concatenate((first_img, second_img), axis=1))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

#Overlay the prediction on the image
#selected filters (ex : [2,3]
#ex: file = '3.tiff'
def superpose(file,filter):
  filename = dir_img + file
  img = cv2.resize(cv2.imread(filename), (width, height))
  prediction = load.predict(np.array([img]) / 255)
  prediction = max_transfo(prediction[0])
  for m in filter :
    mask=prediction[:,:,m]
    img[:,:][mask[:,:]==255]=[0,0,255]
  cv2.imshow("superposition",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

#superpose('3.tiff',[3])
#comparaison_one('3.tiff')
comparaison_all()
