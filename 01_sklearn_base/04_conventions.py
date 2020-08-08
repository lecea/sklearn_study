import numpy as np
from sklearn import random_projection
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

# 强制转换为64
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
print(X.dtype)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)

# classification
iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)
print(list(clf.predict(iris.data[:3])))
clf.fit(iris.data, iris.target_names[iris.target])
print(list(clf.predict(iris.data[:3])))

# refit, 默认rbf
X, y = datasets.load_iris(return_X_y=True)
clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
print(clf.predict(X[:5]))
clf.set_params(kernel='rbf').fit(X, y)
print(clf.predict(X[:5]))

# multi-class multi-label
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X, y).predict(X)
y = LabelBinarizer().fit_transform(y)  # 二标签
print(y)  # 多标签格式
result = classif.fit(X, y).predict(X)
print(result)
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)  # 多标签
result2 = classif.fit(X, y).predict(X)
print(result2)
