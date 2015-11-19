__author__ = "metjush"

# This is an implementation of a simple vanilla feed-forward neural network for supervised learning.
# It is limited to one hidden layer.
# The layers can be activated with Softmax (ReLU), hyperbolic tangent or sigmoid activation functions.
# The outcome can be either a classification or a regression.
# The network is trained with stochastic gradient descent.

import numpy as np
import warnings

warnings.filterwarnings('error')

class VanillaNet:

    def __init__(self, inputs, hidden, outputs, activation='relu', goal='classification', alpha=0.01, regular=0.1, batch=None):
        # Initialize properties of the net

        # Dimensions
        self.input = inputs
        self.hidden = hidden
        self.samples = 0
        self.batch_size = 1
        # Check the number of classes:
        # If the number of classes is 2, it's binary classification,
        # so we only need one output neuron
        self.output = outputs if outputs > 2 else 1
        # Learning parameters
        self.alpha = alpha
        self.regularization = regular
        # Activation and goal
        self.activation = activation
        self.classification = goal == 'classification'
        # Whether this is batch or stochastic
        self.batch = batch


        # Create weight matrices
        self.__rand_init()

    # Initialize weights
    # Separated as a function for untraining the net for cross_validation
    def __rand_init(self):
        # Reset costs
        self.costs = []
        # Don't forget to add one parameter for bias units
        self.theta_hidden = 2.*np.random.rand(self.input + 1, self.hidden) - 1.
        self.theta_final = 2.*np.random.rand(self.hidden + 1, self.output) - 1.
         # initialize deltas
        self.delta_final = np.zeros_like(self.theta_final)
        self.delta_hidden = np.zeros_like(self.theta_hidden)

    # Define activation functions
    # All are implemented such that they can also return their derivatives

    # Sigmoid activation function (logistic)
    def __sigmoid(self, x, dx=False):
        if dx:
            return self.__sigmoid(x, False) * (1. - self.__sigmoid(x, False))
        else:
            out = np.empty_like(x)
            idx = x>0
            out[idx] = 1. / (1. + np.exp(-x[idx]))
            invexp = np.exp(x[~idx])
            out[~idx] = invexp / (1. + invexp)
            return out

    # ReLU activation function
    def __relu(self, x, dx=False):
        if dx:
            dfdx = np.zeros_like(x)
            dfdx[x > 0] = 1.
            dfdx[x <= 0] = 0.
            return dfdx
        else:
            return np.maximum(x, np.zeros_like(x))

    # Softplus activation
    def __softplus(self, x, dx=False):
        if dx:
            return self.__sigmoid(x)
        else:
            return np.log(1. + np.exp(x))

    # Hyperbolic tangent activation function
    def __tanh(self, x, dx=False):
        if dx:
            return 1. - np.tanh(x)**2
        else:
            return np.tanh(x)

    #Linear activation
    def __linear(self, x, dx=False):
        if dx:
            return np.ones_like(x)
        else:
            return x

    # General activation function wrapper
    def __activate(self, x, dx=False, method='softplus'):
        if method == 'softplus':
            return self.__softplus(x, dx)
        elif method == 'relu':
            return self.__relu(x, dx)
        elif method == 'tanh':
            return self.__tanh(x, dx)
        elif method == 'sigmoid':
            return self.__sigmoid(x, dx)
        elif method == 'linear':
            return self.__linear(x, dx)
        else:
            # Default to softmax if invalid argument is passed
            return self.__relu(x, dx)

    # Feed forward through the network
    def __forward(self, x):
        d = len(x.shape)
        ax = 1
        if d == 1:
            x = x.reshape(1, len(x))
        m = x.shape[0]
        # Add bias unit to x
        z0 = np.concatenate((np.ones((m,1)), x), axis=ax)
        # Pass signal to the hidden layer
        z1 = np.dot(z0, self.theta_hidden)
        # Activate the hidden neurons
        a1 = self.__activate(z1, False, self.activation)
        # Add bias unit to the activations
        a1 = np.concatenate((np.ones((a1.shape[0],1)), a1), axis=ax)
        # Pass signal to the output layer
        z2 = np.dot(a1, self.theta_final)
        # Depending on whether the goal is classification or regression
        # Either activate the final layer or leave it as is
        a2 = self.__sigmoid(z2) if self.classification else z2
        # Return all calculations for the gradient
        return [z0, z1, a1, z2, a2]

    # Cost functions depending on type of learning
    def __regularization_cost(self, gradient=False):
        # substitute bias coefficients for zeros
        # as bias weights are not regularized by convention
        theta10 = np.copy(self.theta_hidden)
        theta10[:,0] = 0.
        theta20 = np.copy(self.theta_final)
        theta20[0,:] = 0.
        # compute regularization cost
        if not gradient:
            return (self.regularization/(2.*self.samples)) * (np.sum(np.multiply(theta10,theta10)) + np.sum(np.multiply(theta20,theta20)))
        else:
            return [(self.regularization/self.samples) * theta10, (self.regularization/self.samples) * theta20]

    # Classification log loss
    def __log_loss(self, truth, yhat):
        # Initiate cost at 0
        J = 0
        truth = truth.reshape(yhat.shape)
        # For each class, calculate log loss and sum
        if self.output == 1:
            J -= truth*np.log(yhat) + (1-truth)*np.log(1-yhat)
        else:
            for c in xrange(self.output):
                J -= truth[:,c] * np.log(yhat[:,c]+0.00000000000001)
        J = np.sum(J)
        J /= 2.*len(yhat)
        # Add regularization
        J += self.__regularization_cost()
        return J

    # Regression Root squared mean error
    def __square_loss(self, truth, yhat):
        # calculate error
        error = yhat - truth.reshape(yhat.shape)
        loss = np.sqrt(np.dot(error.T, error))/(2.*self.batch_size)
        return loss + self.__regularization_cost()

    # Wrapper for loss
    def __loss(self, truth, yhat):
        if self.classification:
            return self.__log_loss(truth, yhat)
        else:
            return self.__square_loss(truth, yhat)

    # Backwards propagation back

    def __gradients(self, truth, forward_pass):
        # compute individual gradients
        d_final = (forward_pass[-1] - truth).reshape((1, self.output))
        dz1 = self.__activate(np.concatenate(([[1]],forward_pass[1].T)), True)
        d_hidden = np.multiply( np.dot(self.theta_final, d_final.T), dz1 ).T
        a1 = forward_pass[2].reshape((self.hidden+1, 1))

        # return individual gradients to add to the total gradient function
        return np.dot(forward_pass[0].reshape(self.input + 1, 1), d_hidden[:,1:]), np.dot(a1, d_final)

    def __backprop(self):
        # initialize deltas
        self.delta_final = np.zeros_like(self.theta_final)
        self.delta_hidden = np.zeros_like(self.theta_hidden)

        # get regularizations
        regulars = self.__regularization_cost(True)

        # add regularizatio
        self.delta_hidden += regulars[0]
        self.delta_final += regulars[1]

    def __cost_check(self):
        cost_diff = np.log10(self.costs[-1]/self.costs[-2])
        #if cost_diff > 2:
        #    print("Gradient possibly exploding, try lowering learning rate")
        if cost_diff > 4:
            raise OverflowError("Gradient Exploding, Lower Learning Rate From %f To %f" % (self.alpha, self.alpha * 0.1))

    # Stochastic descent function
    def __stochastic_descent(self, X, y, iterations):
        # start the outer iteration over the whole sample
        indices = range(len(X))
        for iter in xrange(iterations):
            # shuffle sample
            np.random.shuffle(indices)
            shuffledX = X[indices]
            shuffledy = y[indices]
            # start inner loop within the sample
            for k, sample in enumerate(shuffledX):
                self.batch_size = 1
                # get prediction
                yhat = self.__forward(sample)
                # calculate cost
                self.costs.extend([self.__loss(shuffledy[k], yhat[-1])])
                # check if costs aren't exploding
                if k > 2 and k % 3 == 0:
                    self.__cost_check()
                # initialize gradients with regularization costs
                self.__backprop()
                # compute gradients
                gradient_hidden, gradient_final = self.__gradients(shuffledy[k], yhat)
                self.delta_final += gradient_final
                self.delta_hidden += gradient_hidden
                # update weights
                self.theta_final -= self.alpha * self.delta_final
                self.theta_hidden -= self.alpha * self.delta_hidden

    # Batch gradient descent
    # batch size determines the size of the batch to be passed into forward/backward prop
    # it is expressed in percentage terms, defaults to 100% (1.0)
    def __batch_descent(self, X, y, iterations, batch_frac = 1.):
        indices = range(len(X))
        batch_size = int(len(X)*batch_frac)
        n_batches = int(1/batch_frac)

        for iter in xrange(iterations):
            if batch_frac < 1.:
                # shuffle observations
                np.random.shuffle(indices)
                shuffledX = X[indices]
                shuffledy = y[indices]
                for b in xrange(n_batches):
                    # get the bth batch of the sample
                    Xb = shuffledX[b*batch_size:(b+1)*batch_size]
                    yb = shuffledy[b*batch_size:(b+1)*batch_size]
                    self.batch_size = Xb.shape[0]
                    # forward pass
                    yhatb = self.__forward(Xb)
                    # costs
                    self.costs.extend([self.__loss(yb, yhatb[-1])])
                    # check costs
                    if b > 2 and b % 3 == 0:
                        self.__cost_check()
                    # compute gradients
                    self.__backprop()
                    # add up gradients
                    for i in xrange(len(Xb)):
                        yhat = self.__forward(Xb[i,:])
                        g0,g1 = self.__gradients(yb[i], yhat)
                        self.delta_hidden += g0
                        self.delta_final += g1
                    # update parameters
                    self.theta_final -= self.alpha * self.delta_final
                    self.theta_hidden -= self.alpha * self.delta_hidden
            else:
                self.batch_size = X.shape[0]
                # forward pass
                yhat = self.__forward(X)
                # cost
                self.costs.extend([self.__loss(y, yhat[-1])])
                # check costs
                if iter > 2 and iter % 3 == 0:
                    self.__cost_check()
                # compute gradients
                self.__backprop()
                # add up gradients
                for i in xrange(len(X)):
                    yhat0 = self.__forward(X[i,:])
                    g0,g1 = self.__gradients(y[i], yhat0)
                    self.delta_hidden += g0
                    self.delta_final += g1
                # update parameters
                self.theta_final -= self.alpha * self.delta_final
                self.theta_hidden -= self.alpha * self.delta_hidden


    # Vectorize the label vector
    def __vectorize(self, y):
        if self.output == 1:
            return y
        else:
            yvec = np.zeros((len(y),self.output))
            for i in xrange(len(y)):
                yvec[i, y[i]] = 1
            return yvec

    def __feature_scale(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Train function
    def train(self, X, y, iterations = 100):
        # TODO: check X and Y are valid arrays
        # set sample size
        self.samples = len(X)
        # vectorize y if there are more classes
        y = self.__vectorize(y)
        # scale features
        X = self.__feature_scale(X)
        if self.batch is None:
            self.__stochastic_descent(X, y, iterations)
        else:
            self.__batch_descent(X, y, iterations, self.batch)
        print("Neural network trained")
        # flatten costs so that they can be plotted
        self.costs = np.array(self.costs).flatten()

    # Update function
    def update(self, x, y):
        # vectorize y
        y = self.__vectorize(y)
        # run forward pass
        forward = self.__forward(x)
        # get gradients
        gradients = self.__backprop(y, forward)
        # update parameters
        self.theta_final -= self.alpha * gradients[1]
        self.theta_hidden -= self.alpha * gradients[0]

    # Predict function
    def predict(self, X):
        Yhat = np.zeros(len(X))
        # for each observation, run forward pass and get prediction
        X = self.__feature_scale(X)
        for s, sample in enumerate(X):
            forward = self.__forward(sample)
            yhat = forward[-1]
            # unravel vectorized output if classification
            if self.output >= 2:
                yhat = np.argmax(yhat)
            elif self.classification:
                yhat = np.int(yhat[0] > 0.5)
            # update prediction vector
            Yhat[s] = yhat
        return Yhat

    # F1 score
    def __f1(self, truth, yhat, recourse=False):
        # check if this is a binary problem or a multiclass problem
        accurate = truth == yhat
        if self.output == 1 or recourse:
            # binary f1
            positive = np.sum(truth == 1)
            hat_positive = np.sum(yhat == 1)
            tp = np.sum(yhat[accurate] == 1)
            recall = 1.*tp/positive if positive > 0 else 0.
            precision = 1.*tp/hat_positive if hat_positive > 0 else 0.
            f1 = (2.*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.
            return f1
        else:
            # iterate over all classes and return weighted average of individual F1 scores
            # weighting is based on true instances of class
            f1 = 0
            for label in xrange(self.output):
                # create binary vectors to pass to self recursively
                truth_binary = np.copy(truth)
                truth_binary[truth == label] = 1
                truth_binary[truth != label] = 0

                yhat_binary = np.copy(yhat)
                yhat_binary[yhat == label] = 1
                yhat_binary[yhat != label] = 0

                f1 += np.sum(truth == label) * self.__f1(truth_binary, yhat_binary, True)
            return f1 / len(yhat)
        pass

    # RMSE for a vector
    def __rmse(self, truth, yhat):
        # calculate error
        error = yhat - truth
        # square and root
        return np.sqrt(np.dot(error.T, error)) / len(yhat)

    # Evaluate function
    # For classification problems, calculate the F1 score
    # For regression problems, calculate root mean squared error
    def evaluate(self, X, y):
        Yhat = self.predict(X)
        if self.classification:
            return self.__f1(y, Yhat)
        else:
            return self.__rmse(y, Yhat)

    # Cross validate function
    def cross_val(self, X, y, iterations=100, fraction=0.7, folds=1):
        indices = np.arange(len(X))
        set_ind = set(indices)
        size = np.int(len(X)*(1-fraction))
        scores = np.zeros(folds)
        for f in xrange(folds):
            train = np.random.choice(indices, size, replace=False)
            set_train = set(train)
            set_test = list(set_ind.difference(set_train))
            Xtrain = X[train, :]
            ytrain = y[train]
            Xtest = X[set_test, :]
            ytest = y[set_test]
            self.train(Xtrain, ytrain, iterations)
            scores[f] = self.evaluate(Xtest, ytest)
            self.__rand_init()
            print(scores[f])
        return scores

    # To JSON function
    def to_json(self, filename):
        pass

    # From JSON function
    def from_json(self, filename):
        pass
