from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
digits = datasets.load_digits()
iris_data = iris.data
digits_data = digits.images
print(iris_data.shape)
print(digits_data.shape)
