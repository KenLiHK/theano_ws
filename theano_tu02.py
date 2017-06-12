# -*- coding: utf-8 -*-
"""
Created on Thu May 18 05:40:00 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=2exmT0L-QV0&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=6&spfreload=5
https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano5_function.py

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

print('*** theano_tu02.py ***')
print('*** theano_tu02, theano basic ***')
print('theano version:', theano.__version__)


# activation function example
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))    # logistic or soft step
logistic = theano.function([x], s)
print(logistic([[0, 1],[-1, -2]]))

# multiply outputs for a function
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f = theano.function([a, b], [diff, abs_diff, diff_squared])
print( f(np.ones((2, 2)), np.arange(4).reshape((2, 2))) )

# default value and name for a function
x, y, w = T.dscalars('x', 'y', 'w')
z = (x+y)*w
f = theano.function([x,
                     theano.In(y, value=1),
                     theano.In(w, value=2, name='weights')],
                   z)
print(f(23, 2, weights=4))

