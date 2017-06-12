# -*- coding: utf-8 -*-
"""
Created on Thu May 25 06:13:08 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=ho4ms9gVjKE&index=14&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T
https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano11_classification_nn/full_code.py

Python 3.5.2 |Anaconda 4.1.1 (64-bit)
Theano version: theano-0.9.0
numpy>=1.9.1
scipy>=0.14
six>=1.9.0

"""

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

print('*** theano_tu07.py ***')
print('*** theano_tu07, theano classification ***')


def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy

rng = np.random

N = 400                                   # training sample size
feats = 784                               # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weights and biases
W = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0.1, name="b")


# Construct Theano expression graph
p_1 = T.nnet.sigmoid(T.dot(x, W) + b)   # Logistic Probability that target = 1 (activation function)
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
# or
# xent = T.nnet.binary_crossentropy(p_1, y) # this is provided by theano
cost = xent.mean() + 0.01 * (W ** 2).sum()# The cost to minimize (l2 regularization)
gW, gb = T.grad(cost, [W, b])             # Compute the gradient of the cost


# Compile
learning_rate = 0.1
train = theano.function(
          inputs=[x, y],
          outputs=[prediction, xent.mean()],
          updates=((W, W - learning_rate * gW), (b, b - learning_rate * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Training
for i in range(500):
    pred, err = train(D[0], D[1])
    if i % 50 == 0:
        print('cost:', err)
        print("accuracy:", compute_accuracy(D[1], predict(D[0])))

print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

