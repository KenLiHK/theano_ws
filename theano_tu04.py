# -*- coding: utf-8 -*-
"""
Created on Fri May 19 05:15:44 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=lWvlKqvvXyw&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=10
https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano8_Layer_class.py

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

print('*** theano_tu04.py ***')
print('*** theano_tu04, theano layer class ***')



class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size, )) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)

"""
to define the layer like this:
l1 = Layer(inputs, 1, 10, T.nnet.relu)
l2 = Layer(l1.outputs, 10, 1, None)
"""