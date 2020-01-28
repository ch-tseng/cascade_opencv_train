import glob, os
import os.path
import time
import cv2
from os.path import basename
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

imgFolder = "H:\\working\\cascade_indoor_face\\cascade_training\\positives"
outputSize = (80, 80)
outputFolder = "H:\\working\\cascade_indoor_face\\cascade_training\\aug_positives"
imageKeepType = "jpg"
numAugment = 5  #create how many images for 1 image?

aug_whitening = False
aug_rotation = 45
aug_w_shift = 0.15
aug_h_shift = 0.15
aug_shear = 0.2
aug_zoom = 0.05
aug_h_flip = True
aug_v_flip = False
aug_fillmode = "nearest"

def augImage(img_file):

    filepath, file_extension = os.path.splitext(img_file)
    #filename = filepath.split(folderCharacter)[-1]
    #print("read ", img_file)

    roi = cv2.imread(os.path.join(imgFolder ,img_file))
    roi = roi[...,::-1]

    datagen = ImageDataGenerator(
        zca_whitening=aug_whitening,
        rotation_range=aug_rotation,
        width_shift_range=aug_w_shift,
        height_shift_range=aug_h_shift,
        shear_range=aug_shear,
        zoom_range=aug_zoom,
        horizontal_flip=aug_h_flip,
        vertical_flip=aug_v_flip,
        fill_mode=aug_fillmode )

    x = img_to_array(roi)   # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape (( 1 ,) + x.shape)   # this is a Numpy array with shape (1, 3, 150, 150)
    i =  0

    for batch in datagen.flow(x, batch_size = 1 ,
            save_to_dir = outputFolder, save_prefix = "aug_", save_format = imageKeepType ):

        i +=  1
        if i >  numAugment:
            break   # otherwise the generator would loop indefinitely

#-----------------------------------------------------------

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

fileCount = 0
i = 0
for file in os.listdir(imgFolder):
    filename, file_extension = os.path.splitext(file)

    if(file_extension.lower()==".jpg" or file_extension.lower()==".jpeg" or 
            file_extension.lower()==".png" or file_extension.lower()==".bmp"):

        print("#{} {} is processing...".format(str(i), file))
        augImage(file)
        i += 1
