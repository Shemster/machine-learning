import numpy as np
import cv2
from multiprocessing import Pool

def import_files(file_list):

    with open(file_list) as img_file_list:
         imgs  = [line.rstrip('\n') for line in img_file_list]

    return imgs

def face_detect(img):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    total = len(faces)
    for i, (x, y, w, h) in enumerate(faces):
        roi_gray = gray[y:y+h, x:x+w]
        outfile = img[:-4] + "_face" + str(i) + ".jpg"
        cv2.imwrite(outfile, roi_gray)


def main():
    n_cpu = 8
    p = Pool(n_cpu)
    images = import_files('file_list_sample')
    p.map(face_detect, images)


if __name__ == "__main__":
    main()
