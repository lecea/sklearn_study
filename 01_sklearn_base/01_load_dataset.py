from sklearn import datasets
iris = datasets.load_iris()
digit = datasets.load_digits()
# print(digit.data)
# print(digit.target)
print(digit.images[0])
