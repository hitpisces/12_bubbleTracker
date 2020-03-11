import cv2
import numpy as np
import time
from collections import Counter
import RPi.GPIO as GPIO
from threading import Thread
import sys
import keyboard

fps = 3
prev = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, fps)
orb = cv2.ORB_create(nfeatures=50)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

ret, frame = cap.read()
frame_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
keypoints_old, descriptors_old = orb.detectAndCompute(frame_old, None)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(12,GPIO.IN)
GPIO.setup(16,GPIO.OUT)
running = True
speed = 0
data = ""
frameCount = 0

def storeData():
    def storing():
        global data
        print(data)
        with open("dataStore.txt","a") as store:
            store.write(data)
        data = ""
    s = Thread(target = storing)
    s.start()

def waterLevel():
    while running:
        if GPIO.input(12) != 1:
            GPIO.output(16, 1)
        else:
            GPIO.output(16, 0)
    
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
            storeData()
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        
        ret, binary = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        framec = cv2.drawContours(frame.copy(), b, -1, (0,255,0), 1)
        
        
        if descriptors is None or descriptors_old is None:
            matching = cv2.drawMatches(frame_old, keypoints_old, framec, keypoints, None, None, flags = 2)
            data = data + time.strftime("%Y-%m-%d %H:%M:%S") + " " + str(speed) + " " + str(i) + "\r\n"
        else:
            matches = bf.match(descriptors_old, descriptors)
            matches = sorted(matches, key = lambda x:x.distance)
            good = []
            sums = 0
            e = 0
            for m in matches:
                if m.distance < 40:
                    e += 1
                    sums += m.distance
                    good.append(m)
            matching = cv2.drawMatches(frame_old, keypoints_old, framec, keypoints, good, None, flags = 2)
            if e == 0:
                data = data + time.strftime("%Y-%m-%d %H:%M:%S") + " " + str(speed) + " " + str(i) + "\r\n"
            else:
                speed = sums/e
                data = data + time.strftime("%Y-%m-%d %H:%M:%S") + " " + str(speed) + " " + str(i) + "\r\n"    
        
        frame_old = frame
        keypoints_old = keypoints
        descriptors_old = descriptors
        
        frameCount += 1
        if frameCount * 1/fps > 4:
            storeData()
            frameCount = 0

    if keyboard.is_pressed('q'):
        running = False
        time.sleep(5)
        break

cap.release()
sys.exit()