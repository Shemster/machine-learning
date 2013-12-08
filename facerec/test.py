import numpy as np
import csv
import cv2


X = []
y = []
X_test = []
img1 = cv2.imread('/home/ubuntu/data/orig_detected_face/Adrian_Nastase/face_0_Adrian_Nastase_0001.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/ubuntu/data/orig_detected_face/Adrian_Nastase/face_0_Adrian_Nastase_0002.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('/home/ubuntu/data/orig_detected_face/Kate_Capshaw/face_0_Kate_Capshaw_0001.jpg', cv2.IMREAD_GRAYSCALE)
X.append(img1)
y.append(1)
X_test.append(img2)
X_test.append(img3)
model = cv2.createLBPHFaceRecognizer()
model.train(np.asarray(X), np.asarray(y))
[p_label1, p_confidence1] = model.predict(np.asarray(X_test[0]))
[p_label2, p_confidence2] = model.predict(np.asarray(X_test[1]))
print p_confidence1
print p_confidence2
