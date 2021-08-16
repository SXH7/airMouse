import cv2 as cv
import numpy as np
import handTrackingModule as htm
import time
import mouse
#import autopy

width = 640
height = 480

cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.handDetector(max=1)

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if(len(lmList)!=0):
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[16][1:]

        #print(x1, y1, x2, y2)
        fingers = detector.fingersUp()
        #print(fingers)

        if(fingers == [1, 1, 0]):
            mouse.click('left')
        if(fingers == [1, 1, 1]):
            mouse.click('right')
        if(fingers == [1, 0, 0]):

            x = np.interp(x1, (0, width), (0,1920))
            y = np.interp(y1, (0, height), (0, 1080))

            mouse.move(x, y)

    cv.imshow('video', img)
    cv.waitKey(1)


