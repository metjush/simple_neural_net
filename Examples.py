__author__ = 'metjush'

import numpy as np
from sklearn.datasets import load_iris
from simpleNet import SimpleNet

data = load_iris()
X = data.data
y = data.target

net = SimpleNet(X.shape[1], X.shape[1]+5, 3)

net.train(X, y, iterations=10)

pred = net.predict(X)
print(pred)

import matplotlib.pyplot as plt
plt.plot(net.costs)
plt.show()
