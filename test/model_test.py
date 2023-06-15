import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from model.model import model_example
import numpy as np
from sklearn import datasets


def test_model_predict():
    clf = model_example()
    X_test = datasets.load_iris().data[49:51]
    y_test = datasets.load_iris().target[49:51]
    y_pred = clf.predict(X_test)
    assert all(y_pred == y_test)
