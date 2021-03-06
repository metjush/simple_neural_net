__author__ = 'metjush'

import numpy as np
from sklearn.datasets import load_iris
from VanillaNet import VanillaNet

# classification test
'''
data = load_iris()
X = data.data
y = data.target

net = VanillaNet(X.shape[1], X.shape[1]+5, 3, activation='relu', alpha=0.005, batch=1.)

net.train(X, y, iterations=1000)

pred = net.predict(X)
print(pred)
evals = net.evaluate(X, y)
print(evals)

import matplotlib.pyplot as plt
plt.plot(net.costs)
plt.show()


#scores = net.cross_val(X, y, 100, 0.6, 5)
#print(scores)

# regression test
'''
from sklearn.datasets import load_boston
boston = load_boston()
X2 = boston.data
y2 = boston.target

regnet = VanillaNet(X2.shape[1], X2.shape[1]+5, 1, alpha=0.0001, activation='relu', goal="regression", batch=1.)
regnet.train(X2, y2, iterations=10000)

evals = regnet.evaluate(X2,y2)
print(evals)

print(regnet.costs)

import matplotlib.pyplot as plt
plt.plot(regnet.costs)
plt.show()
