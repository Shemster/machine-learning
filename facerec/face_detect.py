import cv2
from multiprocessing import Pool

def import_files(file_list):

    with open(file_list) as img_file_list:
        imgs  = [line.rstrip('\n') for line in img_file_list]

    return imgs

def face_detect(imgfile):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, 1.1, 2, cv2.CASCADE_FIND_BIGGEST_OBJECT)
    total = len(faces)
    print "found" + str(total) + "faces"
    for i, (x, y, w, h) in enumerate(faces):
        roi = img[y:y+h, x:x+w]
        outfile = imgfile[:18] + "orig3_biggest/face_" + str(i) + "_" + imgfile[24:]
        cv2.imwrite(outfile, roi)


def main():
    n_cpu = 8
    p = Pool(n_cpu)
    images = import_files('orig3_list')
    p.map(face_detect, images)


if __name__ == "__main__":
    main()
