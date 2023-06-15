from model.create_and_train_clf import create_and_train_iris_clf


class model_example:
    def __init__(self):
        self.clf = create_and_train_iris_clf()

    @property
    def name(self):
        return "SVM IRIS CLF"

    def fit(self, X, y):
        assert False, "This is a pretrained model"

    def predict(self, X):
        return self.clf.predict(X)
