import glob, os
import os.path
import time
import cv2
from xml.dom import minidom
from os.path import basename
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#標記檔的path
xmlFolder = "H:/working/cascade_cat_face/voc_dataset/labels"
#圖片檔的path
imgFolder = "H:/working/cascade_cat_face/voc_dataset/images"
#要取出的標記名稱(class name)
labelName = "catface"
#專案目錄，所有產生的檔案或目錄皆會存於此
projFolder = "H:/working/cascade_cat_face/cascade_training"
#訓練的圖片大小(建議不要太大)
outputSize = (60, 60)
#產生的訓練圖片類型
imageKeepType = "jpg"
#去除標記區域的圖片，是否要作為negative圖片？
generateNegativeSource = True

#-------------------------------------------------------
saveROIsPath = os.path.join(projFolder, "positives")
positiveDesc_file = os.path.join(projFolder, "positives.info")
negOutput = os.path.join(projFolder, "neg_bg")

totalLabels = 0
wLabels = 0
hLabels = 0
w_h_bg_excgange = 0

def saveROI( roiSavePath, imgFolder, xmlFilepath, labelGrep="", generateNeg=False):
    global totalLabels, wLabels, hLabels, negOutput
    
    xml_filename, xml_file_extension = os.path.splitext(xmlFilepath)
    xml_filename = basename(xml_filename)
    img_filename = xml_filename + "." + imageKeepType

    labelXML = minidom.parse(xmlFilepath)
    labelName = []
    labelXstart = []
    labelYstart = []
    labelW = []
    labelH = []
    totalW = 0
    totalH = 0
    countLabels = 0

    tmpArrays = labelXML.getElementsByTagName("filename")
    for elem in tmpArrays:
        filenameImage = elem.firstChild.data
    #print ("Image file: " + filenameImage)

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        labelXstart.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        labelYstart.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        labelW.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        labelH.append(int(elem.firstChild.data))

    print(os.path.join(imgFolder ,img_filename))
    image = cv2.imread(os.path.join(imgFolder ,img_filename))
    image2 = image.copy()
    filepath = imgFolder
    filename = filenameImage

    for i in range(0, len(labelName)):
        if(labelGrep=="" or labelGrep==labelName[i]):
            countLabels += 1
            print(labelXstart[i], labelYstart[i], labelW[i], labelH[i])
            if(labelXstart[i]<0): labelXstart[i]=0
            if(labelYstart[i]<0): labelYstart[i]=0
            
            totalW = totalW + int(labelW[i]-labelXstart[i])
            totalH = totalH + int(labelH[i]-labelYstart[i])
            
            #roi = roi[...,::-1]
            #get the label image from the source image
            roi = image[labelYstart[i]:labelH[i], labelXstart[i]:labelW[i]]
            print(roi.shape)
            roi = cv2.resize(roi,outputSize)

            roiFile = os.path.join(roiSavePath ,xml_filename + '_' + str(countLabels)+"."+imageKeepType)
            cv2.imwrite(roiFile, roi)

            if(generateNegativeSource==True):
                if(w_h_bg_excgange==0):
                    bgColor = (0,0,0)
                    generateNegativeSource==False
                else:
                    bgColor = (255,255,255)
                    generateNegativeSource==True

                cv2.rectangle(image2, (labelXstart[i], labelYstart[i]), 
                    (labelXstart[i]+int(labelW[i]-labelXstart[i]), labelYstart[i]+int(labelH[i]-labelYstart[i])), bgColor, -1)

    if(generateNegativeSource==True):
       negFile = os.path.join(negOutput ,xml_filename + '_' + str(countLabels)+"."+imageKeepType)
       cv2.imwrite(negFile, image2)



    wLabels += totalW
    hLabels += totalH
    totalLabels += countLabels

    #if(countLabels>0): print("Average W, H: {}, {}".format(int(totalW/countLabels), int(totalH/countLabels)) )
    print("    find {}/{} labels.".format(countLabels, totalLabels) )



#Create all required folders
if not os.path.exists(saveROIsPath):
    os.makedirs(saveROIsPath)

if not os.path.exists(saveROIsPath):
    os.makedirs(saveROIsPath)

if(generateNegativeSource == True):
    if not os.path.exists(negOutput):
        os.makedirs(negOutput)

#make positive images
fileCount = 0
for file in os.listdir(xmlFolder):
    filename, file_extension = os.path.splitext(file)
    if(file_extension==".xml"):
        fileCount += 1
        print("processing XML: {}".format(filename))
        
        xmlfile = os.path.join(xmlFolder ,file) 

        saveROI(saveROIsPath, imgFolder, xmlfile, labelName, generateNegativeSource)


avgW = round(wLabels/totalLabels, 1)
avgH = round(hLabels/totalLabels,1)

with open(os.path.join(saveROIsPath ,"desc.txt"), 'a') as the_file:
    the_file.write("{} XML file processed \n".format(fileCount))
    the_file.write("Total labels: {} \n".format(totalLabels))
    the_file.write("Average W:H = {}:{} \n".format(avgW, avgH))	

print("----> Average W:H = {}:{}".format(avgW, avgH ))

with open(positiveDesc_file, 'a') as the_file:
    for file in os.listdir(saveROIsPath):
        filename, file_extension = os.path.splitext(file)
        if(file_extension.lower()==".jpg" or file_extension.lower()==".png" or file_extension.lower()==".jpeg"):
            #print(positiveFolder + "/" + file)
            img = cv2.imread(saveROIsPath + "/" + file)
            sizeimg = img.shape
            #the_file.write( os.path.join(saveROIsPath ,file) + '  1  0 0 ' + str(sizeimg[1]) + ' ' + str(sizeimg[0]) + '\n')
            the_file.write( 'positives/'+file + '  1  0 0 ' + str(sizeimg[1]) + ' ' + str(sizeimg[0]) + '\n')

the_file.close()
