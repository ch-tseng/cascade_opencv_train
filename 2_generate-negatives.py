#python test-slidingwindow.py -c conf/office.json -i datasets/labelimages-1600/images/csi_1600x1600_20180124_130411.jpg

import time, os
import os.path
import cv2
import numpy as np
import imutils

movePixels = 120
resizeScale = 0.5
negSize = (240, 240)
imageKeepType = "jpg"
negSources = "H:\\working\\cascade_indoor_face\\cascade_training\\neg_bg"
negativeOutput = "H:\\working\\cascade_indoor_face\\cascade_training\\negatives"
negative_info = "H:\\working\\cascade_indoor_face\\cascade_training\\negatives.info"
resize_org_w = 1200  # 0--> keep the same

imagesCount = 6000  #how many negative images you want?
#-----------------------------------------------------------------------------------------

winW = negSize[0]
winH = negSize[1]

def imgPyramid(image, scale=0.5, minSize=[120,120], debug=False):
    yield image
 
        # keep looping over the pyramid
    while True:
        w = int(image.shape[1] * scale)
        h = int(image.shape[0] * scale)
        image = cv2.resize(image, (w, h))
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
 
        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

if not os.path.exists(negativeOutput):
    os.makedirs(negativeOutput)

i = 0
with open(negative_info, 'w') as the_file:

    for file in os.listdir(negSources):
        filename, file_extension = os.path.splitext(file)
        #if("." + imageKeepType == file_extension.lower()):
        if(file_extension.lower()==".jpg" or file_extension.lower()=="png" or file_extension.lower()=="jpeg"):
            print(os.path.join(negSources, file))
            image = cv2.imread(os.path.join(negSources, file))
            if(resize_org_w>0):
                image = imutils.resize(image, width=resize_org_w)

            # loop over the image pyramid
            for layer in imgPyramid(image, scale=resizeScale, minSize=[winW,winH]):
                # loop over the sliding window for each layer of the pyramid
                for (x, y, window) in sliding_window(layer, stepSize=movePixels, windowSize=(winW, winH)):
                    # if the current window does not meet our desired window size, ignore it
                    if window.shape[0] != winH or window.shape[1] != winW:
                        continue
     
                    #clone = layer.copy()
                    #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                    #cv2.imshow("Window", clone)

                    if(i < imagesCount):
                        print("     "+os.path.join(negativeOutput,str(time.time())+str(i)+"."+imageKeepType))
                        neg_imgname = os.path.join(negativeOutput, str(time.time())+str(i)+"."+imageKeepType)
                        cv2.imwrite(neg_imgname, window)
                        the_file.write( neg_imgname + '\n')

                        i += 1
                        print("     #{} negative image saved.".format(i))

            if( i >= imagesCount):
                print("image count reached.")
                break

the_file.close()