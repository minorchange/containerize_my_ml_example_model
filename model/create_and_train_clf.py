from sklearn import datasets
from sklearn.svm import SVC


def create_and_train_iris_clf():
    iris = datasets.load_iris()
    clf = SVC()
    clf.fit(iris.data, iris.target)
    return clf
