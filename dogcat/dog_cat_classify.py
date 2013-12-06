#!/usr/bin/python

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm, grid_search
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from multiprocessing import Pool
from itertools import islice
import time

def import_files(dog_img_list_file, cat_img_list_file, test_img_list_file):

    with open(dog_img_list_file) as dog_img_list:
         dog_imgs  = [line.rstrip('\n') for line in dog_img_list]
    with open(cat_img_list_file) as cat_img_list:
         cat_imgs  = [line.rstrip('\n') for line in cat_img_list]
    with open(test_img_list_file) as test_img_list:
         test_imgs  = [line.rstrip('\n') for line in test_img_list]

    return dog_imgs, cat_imgs, test_imgs

def map_sift_desc(img):

    raw = cv2.imread(img)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, desc = sift.detectAndCompute(gray, None)

    return desc

def online_mbk(train_sift_features, test_sift_features):
    n = train_sift_features.shape[0]
    rng = np.random.RandomState(0)
    kmeans = MiniBatchKMeans(n_clusters = 1000, batch_size = 1000, max_iter = 100, verbose = True)
    index = 0
    for _ in range(3):
        train_sift_features = shuffle(train_sift_features, n_samples = int(round(n*0.1)), random_state = rng)
        i = iter(train_sift_features)
        while True:
            index += 1
            sublist = list(islice(i, 2500))
            if len(sublist) > 0:
                sublist = np.vstack(sublist)
                kmeans.partial_fit(sublist)
            else:
                break

    print "finished training"
    train_predicted_labels = kmeans.predict(train_sift_features)
    test_predicted_labels = kmeans.predict(test_sift_features)
    return train_predicted_labels, test_predicted_labels

def get_hist_feature(sift_features, predicted_labels):

    feature_num = [f.shape[0] for f in sift_features]
    i = iter(predicted_labels)
    labels_list = [list(islice(i, num)) for num in feature_num]
    hist = np.zeros(shape = (len(feature_num), 1000))
    for i, labels in enumerate(labels_list):
        for label in labels:
            hist[i, label] += 1

    return hist

def classify(train_features, train_labels, test_features):
    clf = svm.SVC(C = 100, kernel = 'rbf', gamma = 0.001)
    clf.fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)
    return predicted_labels

def main():

    n_cpu = 8
    p = Pool(n_cpu)

    dog_images, cat_images, test_images = import_files('dog_img', 'cat_img',
    'test_img')
    n_dog = len(dog_images)
    n_cat = len(cat_images)
    n_all = n_dog + n_cat
    n_test = len(test_images)
    all_images = np.concatenate((dog_images, cat_images, test_images), axis = 0)
    train_labels = np.concatenate((np.ones(n_dog), np.zeros(n_cat)), axis = 0)
    print "begin sift feature extraction"
    sift_start = time.time()
    sift_features = p.map(map_sift_desc, all_images)
    sift_end = time.time()
    print (sift_end - sift_start)
    print "stacking features"
    stack_start = time.time()
    train_sift_features = np.vstack(sift_features[: n_train])
    test_sift_features = np.vstack(sift_features[n_train :])
    stack_end = time.time()
    print (stack_end - stack_start)
    print "begin mini batch kmeans"
    kmeans_start = time.time()
    train_predicted_labels, test_predicted_labels = online_mbk(train_sift_features, test_sift_features)
    kmeans_end = time.time() 
    print (kmeans_end - kmeans_start)
    print "begin histogram of features"
    hist_start = time.time()
    train_hist_features = get_hist_feature(sift_features[: n_train],
            train_predicted_labels)
    test_hist_features = get_hist_feature(sift_features[n_train :],
            test_predicted_labels)
    hist_end = time.time()
    print (hist_end - hist_start)
    print "begin prediction"
    svm_start = time.time()
    pred = classify(train_hist_features, train_labels, test_hist_features)
    svm_end = time.time()
    print (svm_end - svm_start)

    print "write submission file"
    out = pd.DataFrame(pred, columns = ['label'])
    out = out.astype(int)
    out.index += 1
    out.to_csv('sub1.csv', index_label = 'id')


if __name__ == "__main__":
    main()
