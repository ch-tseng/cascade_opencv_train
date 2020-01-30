import cv2
import os

projFolder = "H:/working/cascade_cat_face/cascade_training"


aug_positives = os.path.join(projFolder, "aug_positives")
positives_path = os.path.join(projFolder, "positives")
positiveDesc_file = os.path.join(projFolder, "positives.info")

with open(positiveDesc_file, 'a') as the_file:
    for file in os.listdir(aug_positives):
        filename, file_extension = os.path.splitext(file)
        if(file_extension.lower()==".jpg" or file_extension.lower()==".png" or file_extension.lower()==".jpeg"):
            #print(positiveFolder + "/" + file)
            img = cv2.imread(os.path.join(aug_positives ,file))
            cv2.imwrite(os.path.join(positives_path ,file), img)
            sizeimg = img.shape
            #the_file.write( os.path.join(saveROIsPath ,file) + '  1  0 0 ' + str(sizeimg[1]) + ' ' + str(sizeimg[0]) + '\n')
            the_file.write( 'positives/'+file + '  1  0 0 ' + str(sizeimg[1]) + ' ' + str(sizeimg[0]) + '\n')

the_file.close()