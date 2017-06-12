# -*- coding: utf-8 -*-
"""
Created on Fri May 19 05:01:31 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=GbYWEOjjsAI&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=7
https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano6_shared_variable.py

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

print('*** theano_tu03.py ***')
print('*** theano_tu03, theano shared variable ***')



state = theano.shared(np.array(0,dtype=np.float64), 'state') # inital state = 0
inc = T.scalar('inc', dtype=state.dtype)
accumulator = theano.function([inc], state, updates=[(state, state+inc)])

# to get variable value
print(state.get_value())
accumulator(1)   # return previous value, 0 in here
print(state.get_value())
accumulator(10)  # return previous value, 1 in here
print(state.get_value())

# to set variable value
state.set_value(-1)
accumulator(3)
print(state.get_value())

# temporarily replace shared variable with another value in another function
tmp_func = state * 2 + inc
a = T.scalar(dtype=state.dtype)
skip_shared = theano.function([inc, a], tmp_func, givens=[(state, a)]) # temporarily use a's value for the state
print(skip_shared(2, 3))
print(state.get_value()) # old state value