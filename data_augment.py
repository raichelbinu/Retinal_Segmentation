# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:49:47 2022

@author: TOSHIBA
"""
import os
import numpy as np
import albumentations as A
from skimage import io
import random
from data_prepare import one_hotencode
images_to_generate=10


#%%

images_path = r'C:/Users/TOSHIBA/Documents/Research/Research/FirstPapers/Datasets/RITE/AV_groundTruth/AV_groundTruth/TestSet/512images/'
masks_path = r'C:/Users/TOSHIBA/Documents/Research/Research/FirstPapers/Datasets/RITE/AV_groundTruth/AV_groundTruth/TestSet/512av/'

img_augmented_path = r'C:/Users/TOSHIBA/Documents/Research/Research/FirstPapers/Datasets/RITE/AV_groundTruth/AV_groundTruth/TestSet/512augimages/'       # path to store augmented images
msk_augmented_path = r'C:/Users/TOSHIBA/Documents/Research/Research/FirstPapers/Datasets/RITE/AV_groundTruth/AV_groundTruth/TestSet/512augav/'          #path to store augmented masks
 
images=[]   # to store paths of images from folder
masks=[]    # to store paths of masks from folder

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

for msk in os.listdir(masks_path):  # read mask name from folder and append its path into "masks" array     
    masks.append(os.path.join(masks_path,msk))

#%%
aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    
        # , alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    
    ]
)

#random.seed(42)
#%%
i=1   # variable to iterate till images_to_generate


while i<=images_to_generate: 
    number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
    image = images[number]
    mask = masks[number]
    print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    #img_num=image_name.split('_')[0]
    original_mask = io.imread(mask)
    
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

        
    new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.png" %(msk_augmented_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
    i =i+1
    
    #%%
aug_images=[]
aug_masks=[]
def get_augdata():
    
    for img in os.listdir(img_augmented_path):  # read image name from folder and append its path into "images" array     
        im = io.imread(os.path.join(img_augmented_path ,img))
        aug_images.append(np.array(im))

    for m in os.listdir(msk_augmented_path):  # read mask name from folder and append its path into "masks" array     
        msk = io.imread(os.path.join(msk_augmented_path ,m))
        aug_masks.append(msk)
        
    aug_images = np.array(aug_images)
    encoded_aug_masks = one_hotencode(aug_masks)
    encoded_aug_masks = np.array(encoded_aug_masks)
                          
    return aug_images, encoded_aug_masks
    

    #%%