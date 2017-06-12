# -*- coding: utf-8 -*-
"""
Created on Wed May 17 06:24:27 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=je2oHCX5m74&index=5&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T
https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk11_save.py

Python 3.5.2 |Anaconda 4.1.1 (64-bit)
Theano version: theano-0.9.0
numpy>=1.9.1
scipy>=0.14
six>=1.9.0

"""

from __future__ import print_function
import numpy as np
import theano.tensor as T
from theano import function

import theano

print('*** theano_tu01.py ***')
print('*** theano_tu01, theano basic ***')
print('theano version:', theano.__version__)

# basic
x = T.dscalar('x')
y = T.dscalar('y')
z = x+y     # define the actual function in here
f = function([x, y], z)  # the inputs are in [], and the output in the "z"

print(f(2,3))  # only give the inputs "x and y" for this function, then it will calculate the output "z"

# to pretty-print the function
from theano import pp
print(pp(z))

# how about matrix
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print(np.arange(12).reshape((3,4)))
print(10*np.ones((3,4)))
print(f(np.arange(12).reshape((3,4)), 10*np.ones((3,4))))



