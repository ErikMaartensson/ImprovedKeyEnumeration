#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:22:06 2022

@author: ema103
"""

# A very primitive, but simple to understand implementation which can be used a a sanity-check for small values

import numpy as np
import math

import matplotlib.pyplot as plt

def entropy(p):
    return sum([-k*math.log(k, 2) for k in p])

p = np.array([1, 4, 6, 4, 1])/16 # Kyber768/Kyber1024

# p = np.array([1, 6, 15, 20, 15, 6, 1])/64 # Kyber512/FireSaber

# p = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1])/256 # Saber
    
# p = np.array([1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1])/1024 # LightSaber

p = np.flip(np.sort(p))

pN = p.copy()

n = len(p)

guesses = np.array(range(1, n + 1))

quantum = 0

print(np.dot(pN, guesses))


N = 11 # this primitive implementation crashes beyond N = 11 for Kyber768/Kyber1024
#N = 9 # this primitive implementation crashes beyond N = 9 for Kyber512/FireSaber
#N = 8 # this primitive implementation crashes beyond N = 8 for Saber ...
#N = 7 # this primitive implementation crashes beyond N = 7 for LightSaber ...

meanNumberOfGuesses = [0]*N
meanNumberOfGuesses[0] = np.dot(pN, guesses)

for k in range(N - 1):
    pN = np.outer(p, pN)
    pN = pN.flatten()
    pN = np.flip(np.sort(pN))
    n = n * len(p)
    if quantum:
        guesses = np.array([math.sqrt(k) for k in range(1, n + 1)])
    else:
        guesses = np.array(range(1, n + 1))
    meanNumberOfGuesses[k + 1] = np.dot(pN, guesses)



H = entropy(p)

pessimisticModel = [len(p)**k for k in range(1, N + 1)]

MatzovModel = [2**(H*k) for k in range(1, N + 1)]

approximateASModel = [(2/math.sqrt(math.e)) * 2**(H*k) for k in range(1, N + 1)] # See Quantum Augmented Dual Attack paper - p. 14 and see Lemma 4 for a more precise estimate

# Compare the four different models!
plt.scatter(list(range(1, N + 1)), meanNumberOfGuesses, label = 'Exact number')
plt.scatter(list(range(1, N + 1)), pessimisticModel, label = 'Exhaustive')
plt.scatter(list(range(1, N + 1)), MatzovModel, label = 'Matzov')
# plt.scatter(list(range(1, N + 1)), approximateASModel)
plt.title('Expected number of secrets to enumerate')
plt.xlabel('Number of positions')
plt.ylabel('Number of guesses')

plt.yscale("log")

plt.legend()

print(meanNumberOfGuesses)
    

