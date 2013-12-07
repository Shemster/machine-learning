import cv2
import numpy as np

img = cv2.imread('/home/ubuntu/data/test1/5282.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp, des = sift.detectAndCompute(gray,None)
print des

img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints_5282.jpg',img)
