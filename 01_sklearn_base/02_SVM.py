from sklearn import svm
from sklearn import datasets
# learning and predicting
# fit(X, y): label进行训练
# predict(T)
digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-2], digits.target[:-2])
# print(digits.images[-1])
res = clf.predict(digits.data[-2:])
print(res)
