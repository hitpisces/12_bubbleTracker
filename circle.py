import cv2
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt

fps = 5
prev = 0                                     

cap = cv2.VideoCapture("bubble.mp4")
cap.set(cv2.CAP_PROP_FPS, fps)

while True:
    time_elapsed = time.time() - prev
    
    if time_elapsed > 1./fps:
        prev = time.time()
    
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(frame, 135, 255, cv2.THRESH_BINARY)
        
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 25, param1 = 30, param2 = 10, minRadius = 5, maxRadius = 20)
        if circles is not None:
            for i in circles[0]:
                cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 1)

        framec = cv2.resize(frame, (710,600))
        cv2.imshow(cv2.__version__, framec)
     
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
