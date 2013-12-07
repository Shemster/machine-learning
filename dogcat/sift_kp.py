#!/usr/bin/python

import cv2
import numpy as np
#from multiprocessing import Pool

def import_files(dog_img_list_file, cat_img_list_file, test_img_list_file):

    with open(dog_img_list_file) as dog_img_list:
         dog_imgs  = [line.rstrip('\n') for line in dog_img_list]
    with open(cat_img_list_file) as cat_img_list:
         cat_imgs  = [line.rstrip('\n') for line in cat_img_list]
    with open(test_img_list_file) as test_img_list:
         test_imgs  = [line.rstrip('\n') for line in test_img_list]

    return dog_imgs, cat_imgs, test_imgs

def get_sift_desc(img):

    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, desc = sift.detectAndCompute(gray, None)

    return desc

def main():

#    n_cpu = 8
#    p = Pool(n_cpu)

    dog_images, cat_images, test_images = import_files('dog_img', 'cat_img',
    'test_img')

#    dog_sift_desc = p.map(get_sift_desc, dog_images)
#    cat_sift_desc = p.map(get_sift_desc, cat_images)
#    test_sift_desc = p.map(get_sift_desc, test_images)
    test_sift_desc = [get_sift_desc(img) for img in test_images]
#    dog_sift_features = np.vstack(dog_sift_desc)
#    np.save('dog_sift_desc', dog_sift_features)
#    cat_sift_features = np.vstack(cat_sift_desc)
#    np.save('cat_sift_desc', cat_sift_features)
    test_sift_features = np.vstack(test_sift_desc)
    np.save('test_sift_desc', test_sift_features)

if __name__ == "__main__":
    main()
