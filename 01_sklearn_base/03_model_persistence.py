from sklearn import datasets
from sklearn import svm
import pickle
from joblib import dump, load

X, y = datasets.load_digits(return_X_y=True)
clf = svm.SVC()
clf.fit(X, y)
# method 1:pickle
# s = pickle.dumps(clf)
# clf2 = pickle.loads(s)
# res = clf2.predict(X[0:1])
# print(res)
# method 2:joblib
# dump(clf, "filename.joblib")
clf3 = load("filename.joblib")
res3 = clf3.predict(X[0:1])
print(res3)
