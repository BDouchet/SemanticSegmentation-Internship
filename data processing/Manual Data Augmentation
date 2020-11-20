"""
Random data augmentation :

- Add random noise
- Gamma Variation
- Color Variation
- Random rotation
- Random Crop
- Vertical and Horizontal Flips

"""

import numpy as np
from tqdm import tqdm
import cv2
import os
import random

dir_final='dataset/'

# add random noise
def noise(image):
    h, w, c=image.shape
    n=np.random.randn(h, w, c)*random.randint(1, 10)
    return np.clip(image+n, 0, 255).astype(np.uint8)

#change gamma
def change_gamma(image, alpha=1, beta=0):
    return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)

#change color
def color(image, alpha=10):
    n=[random.randint(-alpha, alpha), random.randint(-alpha, alpha),random.randint(-alpha, alpha)]
    return np.clip(image+n, 0, 255).astype(np.uint8)

#apply randomly the changes
def random_change(image):
    if np.random.randint(2):
        image=change_gamma(image, random.uniform(0.9, 1.1), np.random.randint(50)-25)
    if np.random.randint(2):
        image=bruit(image)
    if np.random.randint(2):
        image=color(image)
    return image

#rotate and crop image
def rotateImageandcropimg(image, angle, topleft,bottomright):
    image_center=tuple(np.array(image.shape[1::-1])/2)
    rot_mat=cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result=cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    result=result[topleft[1]:bottomright[1],topleft[0]:bottomright[0],:]
    return result

#rotate and crop mask
def rotateImageandcropmask(image, angle,topleft,bottomright):
    image_center=tuple(np.array(image.shape[1::-1])/2)
    rot_mat=cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result=cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST,borderValue=-1)
    result=result[topleft[1]:bottomright[1],topleft[0]:bottomright[0],:]
    result[:,:][result[:,:,0]==-1]=[0,0,0]
    return result

#operate the data augmentation and save it
def increase(n,dir_img,dir_mask,width,height,dir):
    name=1
    
    for file in tqdm(os.listdir(dir+dir_img)):
        img=cv2.resize(cv2.imread(dir+dir_img+file),(width,height))
        img_mask_result = cv2.resize(cv2.imread(dir+dir_mask + file), (width, height), interpolation=cv2.INTER_NEAREST)
        #First add the original image
        cv2.imwrite(dir_final+'images/'+str(name)+'.tiff',img) 
        cv2.imwrite(dir_final+'masks/' + str(name) + '.tiff', img_mask_result)
        name+=1
        
        for _ in range(n-1):
            process = img.copy()
            process_mask = img_mask_result.copy()
            
            #first apply non spatial modification only on Image
            process=random_change(process)
            
            #Apply angle and crop modification on Image and Mask
            angle=np.random.randint(0,20)
            scale = np.random.randint(200, 400)
            position1 = np.random.randint(0, 412 - scale)
            position2 = np.random.randint(0, 572 - int(scale * 1.38))
            top = [position1, position2]
            bottom = [position1 + scale, position2 + int(scale * 1.38)]
            process=rotateImageandcropimg(process,angle,top,bottom)
            trait_mask=rotateImageandcropmask(trait_mask,angle,top,bottom)
            
            #Apply randomly horizontal or vertical flips
            if np.random.randint(4)!=0:
                i = np.random.randint(3)
                trait=cv2.flip(trait,i-1) # 0 - vertical / 1 - horizontal / -1 : horizontal and vertical
                trait_mask=cv2.flip(trait_mask,i-1)
            
            #resize for the network input
            trait=cv2.resize(trait,(width,height))
            trait_mask=cv2.resize(trait_mask,(width,height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(dir_final+'images/' + str(name) + '.tiff', trait)
            cv2.imwrite(dir_final+'masks/' + str(name) + '.tiff', trait_mask)
            name += 1

dir_img='images/'
dir_mask='masks/'
width=412
height=572
increase(20,dir_img,dir_mask,width,height,'dataset85/')
