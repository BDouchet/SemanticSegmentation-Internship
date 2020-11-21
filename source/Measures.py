"""

Take an image as input and return the size of the objects in the image after a processing through a trained network

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import imutils
from imutils import contours, perspective
import tensorflow.keras.backend as K
import os

pixelratio= None # Nombre de pixel pour faire un mm, 'None' pour afficher les pixels

Save=True
model_path='resunet v1/models/resunet v1.1.h5'

Mode='pred' #gt ou pred
img_name='1.tiff'

height=560
width=400

nw_height=560
nw_width=400

dir_img='../Metrologie/x_metro/'
dir_gt='../Metrologie/y_gt_metro/'
img_path=dir_img+img_name
img=cv2.resize(cv2.imread(img_path),(width,height))
def relu6(x):
  return K.relu(x, max_value=6)
model=load_model(model_path,custom_objects={'relu6':relu6}, compile=False)
code_couleur_BGR = [[70, 70, 70], [200, 200, 200], [200, 0, 0], [0, 0, 200], [0, 200, 0], [200, 0, 200],
                      [200, 200, 0], [0, 200, 200], [0, 100, 255], [50, 0, 150]]
labels = ['Arriere-plan', 'Fuselage', 'Rivets/Boulons', 'Trous/Percages', 'Impacts', 'Rayures', 'Percage Parasite',
            'Ovalisation', 'Pointage Parasite', 'Autres']

## Prends une image en paramètre et renvoie les prédictions
# Enlève l'arrière plan (classes 0 et 1) ###
def get_pred(img):
    image=np.array([img])/255
    pred=model.predict(image)
    pred=pred[0]
    masks_1h=np.zeros((height,width,8), dtype='uint8')
    for i in range(height):
        for j in range(width):
            value=np.argmax(pred[i][j])
            if value!=0 and value!=1:
                masks_1h[i][j][value-2]=1
    return masks_1h

def centre(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def get_measures_pred(image,SAVE,pixelratio,nw_height=560, nw_width=400):
    mask=get_pred(image)
    mask=cv2.resize(mask,(nw_width,nw_height),interpolation=cv2.INTER_NEAREST)
    image=cv2.resize(image,(nw_width,nw_height))
    for i in range(8):
        if np.sum(mask[:,:,i])>0:
            color=code_couleur_BGR[i+2]
            process=np.zeros((nw_height,nw_width,3),dtype='uint8')
            process[:,:,2]=mask[:,:,i]*255
            edged=cv2.Canny(process,50,100)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts)>0:
                (cnts, _) = contours.sort_contours(cnts)
            for c in cnts:
                box=cv2.minAreaRect(c)
                box = cv2.boxPoints(box)
                box = perspective.order_points(box)
                box = np.int0(box)
                #cv2.drawContours(image, [box],0, color)
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = centre(tl, tr)
                (blbrX, blbrY) = centre(bl, br)
                (tlblX, tlblY) = centre(tl, bl)
                (trbrX, trbrY) = centre(tr, br)
                cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                         color, 1)
                cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                         color, 1)
                dA =cv2.norm((tltrX, tltrY),(blbrX, blbrY),cv2.NORM_L2)
                dB = cv2.norm((tlblX, tlblY), (trbrX, trbrY), cv2.NORM_L2)
                if pixelratio is None:
                    cv2.putText(image, "{:.1f}".format(dA),
                                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, color, 1)
                    cv2.putText(image, "{:.1f}".format(dB),
                                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, color, 1)
                else :
                    dA=dA/pixelratio
                    dB=dB/pixelratio
                    cv2.putText(image, "{:.1f}mm".format(dA),
                                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, color, 1)
                    cv2.putText(image, "{:.1f}mm".format(dB),
                                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, color, 1)
    if SAVE:
        cv2.imwrite('../Metrologie/y_pred_metro/'+str(os.path.splitext(img_name)[0]) +' - '+ str(os.path.splitext(os.path.basename(model_path))[0]+'.tiff'),image)
    cv2.imshow("canny",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_measures_gt(image,SAVE,pixelratio,nw_height=560, nw_width=400):
    mask=cv2.resize(cv2.imread(dir_gt+img_name),(nw_width,nw_height),interpolation=cv2.INTER_NEAREST)
    image=cv2.resize(image,(nw_width,nw_height))
    for i in range(8):
        mask_1h = np.array((mask[:, :, 2] == i+2))
        if np.sum(mask_1h) > 0:
            color = code_couleur_BGR[i + 2]
            process = np.zeros((nw_height, nw_width, 3), dtype='uint8')
            process[:, :, 2] = mask_1h * 255
            edged = cv2.Canny(process,50,100)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts)>0:
                (cnts, _) = contours.sort_contours(cnts)
            for c in cnts:
                box=cv2.minAreaRect(c)
                box = cv2.boxPoints(box)
                box = perspective.order_points(box)
                box = np.int0(box)
                #cv2.drawContours(image, [box],0, color)
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = centre(tl, tr)
                (blbrX, blbrY) = centre(bl, br)
                (tlblX, tlblY) = centre(tl, bl)
                (trbrX, trbrY) = centre(tr, br)
                cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                         color, 1)
                cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                         color, 1)
                dA =cv2.norm((tltrX, tltrY),(blbrX, blbrY),cv2.NORM_L2)
                dB = cv2.norm((tlblX, tlblY), (trbrX, trbrY), cv2.NORM_L2)
                if pixelratio is None:
                    cv2.putText(image, "{:.1f}".format(dA),
                                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, color, 1)
                    cv2.putText(image, "{:.1f}".format(dB),
                                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, color, 1)
                else :
                    dA=dA/pixelratio
                    dB=dB/pixelratio
                    cv2.putText(image, "{:.1f}mm".format(dA),
                                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, color, 1)
                    cv2.putText(image, "{:.1f}mm".format(dB),
                                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, color, 1)
    if SAVE:
        cv2.imwrite('../Metrologie/y_gt_metro/'+str(os.path.splitext(img_name)[0]) +' - pixels.tiff',image)
    cv2.imshow("canny",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if Mode=='pred':
    get_measures_pred(img,Save,pixelratio,nw_height,nw_width)
if Mode=='gt':
    get_measures_gt(img, Save, pixelratio, nw_height, nw_width)
