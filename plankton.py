import os
import numpy as np

from nolearn.convnet import ConvNetFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

PROJECT_DIR = '/mnt/playground/'
TRAIN_DATA_DIR = PROJECT_DIR + 'train/'

def get_dataset():
    data = [(TRAIN_DATA_DIR + fn + '/' + img, fn) for fn in
            os.listdir(TRAIN_DATA_DIR) for img in os.listdir(TRAIN_DATA_DIR +
                fn)]
    X, y_label = zip(*data)
    le = LabelEncoder()
    le.fit(y_label)
    y = le.transform(y_label)
    X, y = shuffle(X, y, random_state=0)
    return X, y, le


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


def main():
    convnet = ConvNetFeatures(
            pretrained_params=PROJECT_DIR +
            'imagenet.decafnet.epoch90',
            pretrained_meta=PROJECT_DIR +
            'imagenet.decafnet.meta',
            center_only=False
            )
    # clf = LogisticRegression()
    clf = GradientBoostingClassifier(n_estimators=100,
            max_depth=2, subsample=0.8, random_state=0)
    pl = Pipeline([
        ('convnet', convnet),
        ('clf', clf),
        ])
    X, y, le = get_dataset()
    train_obs = int(len(y) * 0.8)
    X_train, y_train = X[:train_obs], y[:train_obs]
    X_test, y_test = X[train_obs:], y[train_obs:]

    print "Fitting..."
    pl.fit(X_train, y_train)
    print "Predicting..."
    y_pred = pl.predict_proba(X_test)
    print "Log Loss: %s" % multiclass_log_loss(y_test, y_pred)

if __name__ == "__main__":
    main()
