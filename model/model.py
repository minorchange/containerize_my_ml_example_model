import ast
from model.create_and_train_clf import create_and_train_iris_clf
import numpy as np


class model_example:
    def __init__(self):
        self.clf = create_and_train_iris_clf()

    @property
    def name(self):
        return "SVM IRIS CLF"

    def fit(self, X, y):
        assert False, "This is a pretrained model"

    def predict(self, s):
        print("Look here")
        print(s)
        X = ast.literal_eval(s)
        X = np.array(X)
        print(X)
        if len(X) != 4:
            return "Unexpected shape."
        if len(np.shape(X)) == 1:
            X = np.array([X])
        return self.clf.predict(X).tolist()
