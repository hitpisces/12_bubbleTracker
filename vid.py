import cv2
import numpy as np
import time
from collections import Counter
import RPi.GPIO as GPIO
from threading import Thread
import matplotlib.pyplot as plt

fps = 3
prev = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, fps)
orb = cv2.ORB_create(nfeatures=500)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

ret, frame = cap.read()
frame_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
keypoints_old, descriptors_old = orb.detectAndCompute(frame_old, None)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(12,GPIO.IN)
GPIO.setup(16,GPIO.OUT)
running = True

speedRun = []

def waterLevel():
    while running:
        if GPIO.input(12) != 1:
            GPIO.output(16, 1)
            print("in water")
        else:
            GPIO.output(16, 0)
            print("out water")
        time.sleep(.5)
    
t = Thread(target = waterLevel)
t.start()

while True:
    time_elapsed = time.time() - prev
    
    if time_elapsed > 1./fps:
        prev = time.time()
    
        ret, frame = cap.read()
        if frame is None:
            GPIO.output(16, 0)
            running = False
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        
        ret, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours = contours[2:]
        b = []
        s = []
        i = 0
        for a in hierarchy[0]:
            s.append(a[3])
        avg = Counter(s).most_common(1)[0][0]
        for a in hierarchy[0]: 
            if a[3] == avg:
                b.append(contours[i])
            i += 1
        framec = frame
        framec = cv2.drawContours(frame.copy(), b, -1, (0,255,0), 1)
        
        
        if descriptors is None or descriptors_old is None:
            matching = cv2.drawMatches(frame_old, keypoints_old, framec, keypoints, None, None, flags = 2)      
        else:
            matches = bf.match(descriptors_old, descriptors)
            matches = sorted(matches, key = lambda x:x.distance)
            good = []
            listofSpeed = []
            for m in matches:
                if 15 < m.distance < 30:
                    good.append(m)
                    listofSpeed.append(m.distance)
            matching = cv2.drawMatches(frame_old, keypoints_old, framec, keypoints, good, None, flags = 2)
                     
        #matching = cv2.resize(matching, (1200,335))
        cv2.imshow("Matching", matching)
        #speedRun.append()
        #print(listofSpeed)
        #plt.figure()
        #plt.hist(x = listofSpeed, bins = 'auto', range = [0, 60])
        #plt.show()
        #plt.clf()
        #plt.show()
        
        
        frame_old = frame
        keypoints_old = keypoints
        descriptors_old = descriptors
        
    if cv2.waitKey(1) == 27:
        GPIO.output(16, 0)
        running = False
        break

cap.release()
cv2.destroyAllWindows()