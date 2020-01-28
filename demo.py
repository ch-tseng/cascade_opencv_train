import cv2
import imutils

face_cascade = cv2.CascadeClassifier('cascade_training\\LBP\\cascade.xml')
cascade_scale = 1.2
cascade_neighbors = 8
minFaceSize = (30,30)

def getFaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= cascade_scale,
        minNeighbors=cascade_neighbors,
        minSize=minFaceSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    bboxes = []
    for (x,y,w,h) in faces:
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            bboxes.append((x, y, w, h))

    return bboxes

pic = cv2.imread("demos\\demo4.jpg")
pic = imutils.resize(pic, width=640)
faces = getFaces(pic)

for (x,y,w,h) in faces:
    cv2.rectangle( pic,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("TEST", pic)
cv2.waitKey(0)
