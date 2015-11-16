# Vanilla Neural Net

This repository is a basic implementation of a "vanilla" neural network with one hidden layer. It is a basic feed-forward network trained with stochastic gradient descent. It's been built primarily as a self-learning exercise and as a tool for those who want to quickly try out a neural network model without coding it on their own or using a more heavy-weight library.

## Dependencies

This implementation is built on `numpy` (<http://www.numpy.org/>), which is the only dependency required for running the network. `matplotlib` (<http://matplotlib.org/>) is required for plotting learning curves. The example script in `Examples.py` also requires `scikit-learn` (<http://scikit-learn.org/stable/index.html>) as it uses their `datasets` functionality to load example data. However, this package is not required for running the network.

## Basic usage

The neural network has the following basic architecture: an *input layer*, a *hidden layer* with activation, and an *output layer*. The network is initialized by specifying the sizes of the three respective layers, the activation function of the hidden layer (defaults to a Rectified Linear Unit), the learning "goal" (classification or regression), the learning rate `alpha` (defaults to `0.01`) and the regularization parameter (defaults to `0.1`):

```python
from VanillaNet import *
net = VanillaNet(5, 15, 3, activation='relu', goal='classification', alpha=0.01, regularization=0.1)
```

The weights are randomly initialized upon creating the network instance. The network is then trained by calling the `train` function. This takes as arguments the feature matrix `X` and the target vector `y` (numpy arrays), which need to have appropriate dimensions (the number of features of `X` has to be equal to the input size and the number of classes equal to the output size). The only other parameter is the number of iterations (defaults to 100). As the network is trained by stochastic gradient descent, this is the number of iterations the whole dataset is cycled through (shuffling every iteration).

```python
from VanillaNet import *
net = VanillaNet(5, 15, 3, activation='relu', goal='classification', alpha=0.01, regularization=0.1)

# X is a feature matrix
# y is a target vector
net.train(X, y, iterations=100)
```

Upon training, the network can be used for prediction by calling the `predict` function. This takes one argument (the feature matrix to predict/classify) and returns a numpy array of predictions.

```python
from VanillaNet import *
net = VanillaNet(5, 15, 3, activation='relu', goal='classification', alpha=0.01, regularization=0.1)

# X is a feature matrix
# y is a target vector
net.train(X, y, iterations=100)

predictions = net.predict(X)
```

## Evaluation

An alternative to the `predict` function is the `evaluate` function, which, given a ground-truth vector `y`, scores the network prediction. For classification problems, it uses the *F1-score*. For regression problems, it uses the *root mean squared error*. It returns the score of the prediction.

```python
from VanillaNet import *
net = VanillaNet(5, 15, 3, activation='relu', goal='classification', alpha=0.01, regularization=0.1)

# X is a feature matrix
# y is a target vector
net.train(X, y, iterations=100)

score = net.evaluate(X, y)
```

Similarly, one can use k-fold cross-validation with the `cross_val` function. This function trains the network `k` times, each time with a random subsample of the training data, leaving out the remainder for scoring. It returns an array of scores for each fold.

```python
from VanillaNet import *
net = VanillaNet(5, 15, 3, activation='relu', goal='classification', alpha=0.01, regularization=0.1)

# X is a feature matrix
# y is a target vector
net.train(X, y, iterations=100)

cross_val_scores = net.cross_val(self, X, y, iterations=100, fraction=0.7, folds=5)
```

## Updating

Instead of training the network with a full sample, it can also be trained "progressively" (or *online*), one observation at a time. This can be done with the `update` function, which takes a single observation vector `x` and its correpsonding ground truth `y`, and updates the weights of the network after predicting `yhat` given the current trained weights.

```python
from VanillaNet import *
net = VanillaNet(5, 15, 3, activation='relu', goal='classification', alpha=0.01, regularization=0.1)

# X is a feature matrix
# y is a target vector
net.train(X, y, iterations=100)

# x_new is one-dimensional feature vector
# y_new is a single value, correpsonding to x_new ground truth
net.update(x_new, y_new)
```
 
