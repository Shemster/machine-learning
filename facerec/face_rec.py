import numpy as np
import pandas as pd
import csv
import cv2


def read_img(imgfile):
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    return img

def train_predict(X_train, y_train, X_test, y_test):
    model = cv2.createLBPHFaceRecognizer()
    model.train(np.asarray(X_train), np.asarray(y_train))
    hist = model.getMatVector("histograms")
    print len(hist)
    print hist[0].shape
    pred = []
    for i, x_test in enumerate(X_test):
        [p_label, dist] = model.predict(np.asarray(x_test))
        pred.append([y_test[i], dist])
    prediction = np.vstack(pred)
    print prediction.shape
    return prediction


def main():
    imgs = []
    labels = []
    with open ('orig3_big_db.csv') as db:
        reader = csv.reader(db, delimiter=";")
        for row in reader:
            imgs.append(read_img(row[0]))
            labels.append(int(row[1]))

    results = []

    for i in xrange(0, len(imgs)):
        if i % 3 == 0:
            print i
            tmp_imgs = imgs[:]
            tmp_labels = labels[:]
            true_label = (tmp_labels[i])*np.ones(shape = (len(imgs) - 2, 1))
            print tmp_labels[i]
            X_train = [tmp_imgs[i], tmp_imgs[i+1]]
            y_train = [tmp_labels[i], tmp_labels[i+1]]
            del tmp_imgs[i:i+2]
            del tmp_labels[i:i+2]
            X_test = tmp_imgs
            y_test = tmp_labels
            pred = train_predict(X_train, y_train, X_test, y_test)
            mask = true_label[:, 0] == pred[:, 0]
            match = 1*mask.reshape(pred.shape[0], 1)
            result = np.hstack((pred, true_label, match))
            print result.shape
            results.append(result)

    #df = np.vstack(results)
    #data = pd.DataFrame(df, columns = ['true_label_test', 'distance',
    #    'true_label', 'match'])
    #data.index += 1
    #data.to_csv('pred_3train_1_8.csv', index_label = 'index')
    


if __name__ == "__main__":
    main()

