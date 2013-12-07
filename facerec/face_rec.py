import numpy as np
import pandas as pd
import csv
import cv2


def train_predict(X_train, y_train, X_test):
    model = cv2.createLBPHFaceRecognizer()
    model.train(np.asarray(X_train), np.asarray(y_train))
    pred = []
    for x_test in X_test:
        [p_label, dist] = model.predict(np.asarray(x_test))
        pred.append([p_label, dist])
    prediction = np.vstack(pred)
    return prediction


def read_img(imgfile):
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def main():
    imgs = []
    labels = []
    with open ('db.csv') as db:
        reader = csv.reader(db, delimiter=";")
        for row in reader:
            imgs.append(read_img(row[0]))
            labels.append(int(row[1]))

    results = []

    for i in xrange(0, len(imgs)):
        print i
        tmp_imgs = imgs[:]
        tmp_labels = labels[:]
        X_train = [tmp_imgs[i]]
        y_train = [tmp_labels[i]]
        del tmp_imgs[i]
        del tmp_labels[i]
        X_test = tmp_imgs
        y_test = np.array([tmp_labels]).T
        pred = train_predict(X_train, y_train, X_test)
        mask = y_test[:,0] == pred[:, 0]
        match = 1*mask.reshape(mask.shape[0], 1)
        num = i*np.ones(shape=match.shape)
        result = np.hstack((num, pred, y_test, match))
        print result.shape
        results.append(result)

    df = np.vstack(results)
    data = pd.DataFrame(df, columns = ['image_num', 'pred_label', 'dist',
    'label', 'match'])
    data.index += 1
    data.to_csv('data.csv', index_label = 'index')
    


if __name__ == "__main__":
    main()

