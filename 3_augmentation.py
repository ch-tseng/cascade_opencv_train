import glob, os
import os.path
import time
import cv2
from os.path import basename
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#專案目錄，所有產生的檔案或目錄皆會存於此
projFolder = "H:/working/cascade_cat_face/cascade_training"
#產生的正向圖片大小
outputSize = (60, 60)
#產生的圖片格式
imageKeepType = "jpg"
#每一個正向圖片要產生出幾張新圖片?
numAugment = 6

#Augmentation的設定
aug_whitening = False
aug_rotation = 16
aug_w_shift = 0.1
aug_h_shift = 0.1
aug_shear = 0.1
aug_zoom = 0.05
aug_h_flip = True
aug_v_flip = False
aug_fillmode = "nearest"

imgFolder = os.path.join(projFolder, "positives")
outputFolder = os.path.join(projFolder, "aug_positives")

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
