#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:22:06 2022

@author: ema103
"""

import numpy as np
import math

import matplotlib.pyplot as plt

from scipy.special import zeta

# Gets the non-zero indices of a list
def nonZeroIndices(l):
    return list(np.nonzero(l)[0])

# Calculates the mean of a distribution D which takes the values -(q - 1)/2, ..., (q - 1)/2
# Assumes that q is odd
def meanSymmetric(D, q):
    theValues = list(range((q  + 1)//2)) + list(range(-(q - 1)//2, 0))  
    return sum([D[k] * theValues[k] for k in range(q)])

# Calculates the standard deviation of a distribution D which takes the values -(q - 1)/2, ..., (q - 1)/2
# Assumes that q is odd - what happens if q is even?
def standardDeviationSymmetric(D, q):
    mu = meanSymmetric(D, q)
    # print(mu)
    return math.sqrt(sum([D[k] * (min(k, q - k) - mu)**2 for k in range(q)]))

def entropyRepeatedProbabilites(p, freq):
    return sum([-freq[k]*p[k]*math.log(p[k], 2) for k in range(len(p))])

def entropy(p):
    return sum([-k*math.log(k, 2) for k in p])

# Computes the sum of sqrt(k) where k goes from 1 to n
def sumOfSqrts(n):
    # Use the exact expression for very small values of n
    if n < 40:
        return sum ([math.sqrt(k) for k in range(1, n + 1)])
    return 2/3*n**1.5 + zeta(-0.5) + 0.5*n**0.5 + 1/24*n**(-0.5) - 1/1920*n**(-5/2) + 1/9216*n**(-9/2) # For larger values of n we use an approximation formula
        

# Iterate over all length-tuples of non-negative integers that sum to total_sum - see https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation

# Calculates the expected number of guesses of the secret to iterate over - the general case - quantum
def exactNumberOfEnumerationQuantum(p, freq, n):
    
    allTuples = list(sums(len(p), n))
    numberOfTerms = len(allTuples)
    
    sumP = len(p)
    
    sortedP = np.zeros(numberOfTerms)
    sortedFreq = np.zeros(numberOfTerms)
    
    for k in range(numberOfTerms):
        currentTuple = allTuples[k]
        coeff = math.factorial(n) / math.prod(math.factorial(l) for l in currentTuple)
        coeff = coeff * math.prod(freq[l]**currentTuple[l] for l in range(sumP))
        sortedFreq[k] = coeff
        sortedP[k] = math.prod(p[l]**currentTuple[l] for l in range(sumP))
    
    inds = np.argsort(-sortedP) # Sort in decreasing order
    sortedP = sortedP[inds]
    sortedFreq = sortedFreq[inds]
    
    partialSum = 0
    estimatedEnumeration = 0
    for k in range(numberOfTerms):
        fk = int(sortedFreq[k])
        estimatedEnumeration = estimatedEnumeration + (sumOfSqrts(partialSum + fk) - sumOfSqrts(partialSum))*sortedP[k]
        partialSum = partialSum + fk
    
    return estimatedEnumeration

# Calculates the expected number of guesses of the secret to iterate over - the general case - classical
def exactNumberOfEnumeration(p, freq, n):
    
    allTuples = list(sums(len(p), n))
    numberOfTerms = len(allTuples)
    
    sumP = len(p)
    
    sortedP = np.zeros(numberOfTerms)
    sortedFreq = np.zeros(numberOfTerms)
    
    for k in range(numberOfTerms):
        currentTuple = allTuples[k]
        coeff = math.factorial(n) / math.prod(math.factorial(l) for l in currentTuple)
        coeff = coeff * math.prod(freq[l]**currentTuple[l] for l in range(sumP))
        sortedFreq[k] = coeff
        sortedP[k] = math.prod(p[l]**currentTuple[l] for l in range(sumP))
    
    inds = np.argsort(-sortedP) # Sort in decreasing order
    sortedP = sortedP[inds]
    sortedFreq = sortedFreq[inds]
    
    partialSum = 0
    estimatedEnumeration = 0
    for k in range(numberOfTerms):
        fk = sortedFreq[k]
        estimatedEnumeration = estimatedEnumeration + (partialSum*fk + fk*(fk + 1)/2)*sortedP[k]
        partialSum = partialSum + fk
    
    return estimatedEnumeration

# Calculates the expected number of guesses of the secret to iterate over - Assumes for now that p has 3 unique values
def exactNumberOfEnumeration3Values(p, freq, n):
    # p - the list of unique probabilities
    # freq - the number of each unique probability
    # n - the number of positions
    
    numberOfTerms = (n + 2)*(n + 1)//2
    
    sortedP = np.zeros(numberOfTerms)
    sortedFreq = np.zeros(numberOfTerms)
    index = 0
    
    for k in range(n + 1):
        for l in range(0, n - k + 1):
            coeff = math.factorial(n) / (math.factorial(k) * math.factorial(l) * math.factorial(n - k - l))
            # coeff = coeff * math.prod([freq[k]** for k in len(p)]) - something like this in the general case
            coeff = coeff * freq[0]**k * freq[1]**l * freq[2]**(n - k - l)
            sortedFreq[index] = coeff
            sortedP[index] = p[0]**k * p[1]**l * p[2]**(n - k - l)
            index = index + 1
    
    inds = np.argsort(-sortedP) # Sort in decreasing order
    sortedP = sortedP[inds]
    sortedFreq = sortedFreq[inds]
    
    partialSum = 0
    estimatedEnumeration = 0
    for k in range(numberOfTerms):
        fk = sortedFreq[k]
        estimatedEnumeration = estimatedEnumeration + (partialSum*fk + fk*(fk + 1)/2)*sortedP[k]
        partialSum = partialSum + fk
    
    return estimatedEnumeration


# Calculates the expected number of guesses of the secret to iterate over - Assumes for now that p has 4 unique values
def exactNumberOfEnumeration4Values(p, freq, n):
    # p - the list of unique probabilities
    # freq - the number of each unique probability
    # n - the number of positions
    
    numberOfTerms = math.comb(4 + n - 1, n)
    
    sortedP = np.zeros(numberOfTerms)
    sortedFreq = np.zeros(numberOfTerms)
    index = 0
    
    for k in range(n + 1):
        for l in range(0, n - k + 1):
            for m in range(0, n - k - l + 1):
                coeff = math.factorial(n) / (math.factorial(k) * math.factorial(l) * math.factorial(m) * math.factorial(n - k - l - m))
                # coeff = coeff * math.prod([freq[k]** for k in len(p)]) - something like this in the general case
                coeff = coeff * freq[0]**k * freq[1]**l * freq[2]**m * freq[3]**(n - k - l - m)
                sortedFreq[index] = coeff
                sortedP[index] = p[0]**k * p[1]**l * p[2]**m * p[3]**(n - k - l - m)
                index = index + 1
    
    inds = np.argsort(-sortedP) # Sort in decreasing order
    sortedP = sortedP[inds]
    sortedFreq = sortedFreq[inds]
    
    # print(sortedP)
    # print(sortedFreq)
    
    partialSum = 0
    estimatedEnumeration = 0
    for k in range(numberOfTerms):
        fk = sortedFreq[k]
        estimatedEnumeration = estimatedEnumeration + (partialSum*fk + fk*(fk + 1)/2)*sortedP[k]
        partialSum = partialSum + fk
    
    return estimatedEnumeration

# Estimates the expected cost of enumeration classically according to Albrecht-Shen
def enumerationASModel(s, q, k_enum):
    sigmaS = standardDeviationSymmetric(s, q)
    ind = nonZeroIndices(s)
    H = entropy([s[k] for k in ind])
    coeff = 1 / (1 - math.exp(-1 / 2 / sigmaS**2))
    tmp_alpha = math.pi**2 * sigmaS**2
    tmp_a = math.exp(8 * tmp_alpha * math.exp(-2 * tmp_alpha) * math.tanh(tmp_alpha))#.n(30) # Does this expression need extra precision?
    return coeff * ((2 * tmp_a / math.sqrt(math.e)) ** k_enum) * (2 ** (k_enum * H))
    # return coeff * ((2 / math.sqrt(math.e)) ** k_enum) * (2 ** (k_enum * H))

# Estimates the expected cost of enumeration quantumly according to Albrecht-Shen
def enumerationASModelQuantum(s, q, k_enum):
    sigmaS = standardDeviationSymmetric(s, q)
    ind = nonZeroIndices(s)
    H = entropy([s[k] for k in ind])
    coeff = 7/ 6/(1 - math.exp(-1 / 3 / sigmaS**2))**(3/2)
    tmp_alpha = math.pi**2 * sigmaS**2
    tmp_a = math.exp(8 * tmp_alpha * math.exp(-2 * tmp_alpha) * math.tanh(tmp_alpha))
    # return coeff * ((27 * tmp_a**2 / math.sqrt(8 * math.e)) ** (k_enum / 4)) * math.sqrt(2 ** (k_enum * H))
    return coeff * ((27 * tmp_a**2 / (8 * math.e)) ** (k_enum / 4)) * math.sqrt(2 ** (k_enum * H))

# Creates an estimation of y on the form exp(ax + b)
def exponentialApproximation(x, y):
    N = len(y)
    logY = [math.log(y[k]) for k in range(N)]
    # x = list(range(1, N + 1))
    fit = np.polyfit(x, logY, 1)
    return fit

q = 3329 # Kyber
# q = 2**13 # Saber

# Kyber768/Kyber1024
#p = np.array([1, 4, 6])/16
#freq = np.array([2, 2, 1])
#s = [0]*q
#s[0] = 6/16
#s[1] = 4/16
#s[2] = 1/16
#s[-1] = 4/16
#s[-2] = 1/16

# Kyber512/FireSaber
p = np.array([1, 6, 15, 20])/64
freq = np.array([2, 2, 2, 1])
s = [0]*q
s[0] = 20/64
s[1] = 15/64
s[2] = 6/64
s[3] = 1/64
s[-1] = 15/64
s[-2] = 6/64
s[-3] = 1/64

# Saber

#p = np.array([1, 8, 28, 56, 70])/256
#freq = np.array([2, 2, 2, 2, 1])
#s = [0]*q
#s[0] = 70/256
#s[1] = 56/256
#s[2] = 28/256
#s[3] = 8/256
#s[4] = 1/256
#s[-1] = 56/256
#s[-2] = 28/256
#s[-3] = 8/256
#s[-4] = 1/256

# LightSaber
#p = np.array([1, 10, 45, 120, 210, 252])/1024
#freq = np.array([2, 2, 2, 2, 2, 1])
#s = [0]*q
#s[0] = 252/1024
#s[1] = 210/1024
#s[2] = 120/1024
#s[3] = 45/1024
#s[4] = 10/1024
#s[5] = 1/1024
#s[-1] = 210/1024
#s[-2] = 120/1024
#s[-3] = 45/1024
#s[-4] = 10/1024
#s[-5] = 1/1024

# print(standardDeviationSymmetric(s, q))

N = 40
meanNumberOfGuesses = [0]*(N + 1)
meanNumberOfGuessesQuantum = [0]*(N + 1)

for n in range(N + 1):
    print(n) # To track progress for slow parameter settings :-)
    meanNumberOfGuesses[n] = exactNumberOfEnumeration(p, freq, n)
    meanNumberOfGuessesQuantum [n] = exactNumberOfEnumerationQuantum(p, freq, n)

H = entropyRepeatedProbabilites(p, freq)

lenS = int(sum(freq))

pessimisticModel = [lenS**k for k in range(N + 1)]

MatzovModel = [2**(H*k) for k in range(N + 1)]

approximateASModel = [(2/math.sqrt(math.e))**k * 2**(H*k) for k in range(N + 1)] # See Quantum Augmented Dual Attack paper - p. 14 and see Lemma 4 for a more precise estimate
preciseASModel = [enumerationASModel(s, q, k) for k in range(N + 1)]
preciseASModelQuantum = [enumerationASModelQuantum(s, q, k) for k in range(N + 1)]

# Compare the four different models!
plt.plot(list(range(N + 1)), meanNumberOfGuesses, label = 'Exact number')
plt.plot(list(range(N + 1)), pessimisticModel, label = 'Exhaustive')
plt.plot(list(range(N + 1)), MatzovModel, label = 'Matzov')
plt.plot(list(range(N + 1)), preciseASModel, label = 'AlbrechtShen')
# plt.plot(list(range(1, N + 1)), approximateASModel, label = 'Approximate AlbrechtShen')
# plt.title('Expected number of secrets to enumerate - classic')
plt.xlabel('Number of positions')
plt.ylabel('Number of guesses')

plt.yscale("log", base = 2)

plt.legend()

plt.savefig('Classical.pdf')

# The quantum plot comparison
plt.show()
plt.plot(list(range(N + 1)), meanNumberOfGuessesQuantum, label = 'Exact number')
plt.plot(list(range(N + 1)), preciseASModelQuantum, label = 'AlbrechtShen')
# plt.title('Expected number of secrets to enumerate - quantum')
plt.yscale("log", base = 2)
plt.xlabel('Number of positions')
plt.ylabel('Number of guesses')

plt.legend()

plt.savefig('Quantum.pdf')

# Create an approximation formula classical case
# logY = [math.log(meanNumberOfGuesses[k]) for k in range(N)]
x = list(range(N + 1))
# fit = np.polyfit(x, logY, 1)
fit = exponentialApproximation(x, meanNumberOfGuesses)
approxY = [math.exp(fit[1] + fit[0]*x[k]) for k in range(N + 1)]

plt.show()
plt.plot(list(range(N + 1)), meanNumberOfGuesses, label = 'Exact number')
plt.plot(list(range(N + 1)), approxY, label = 'Exponential regression')
plt.title('Expected number of secrets to enumerate - classical')
plt.yscale("log", base = 2)
plt.xlabel('Number of positions')
plt.ylabel('Number of guesses')

plt.legend()

# Create an approximation formula quantum case
# logYQ = [math.log(meanNumberOfGuessesQuantum[k]) for k in range(N)]
x = list(range(N + 1))
# fitQ = np.polyfit(x, logYQ, 1)
fitQ = exponentialApproximation(x, meanNumberOfGuessesQuantum)
approxYQ = [math.exp(fitQ[1] + fitQ[0]*x[k]) for k in range(N + 1)]

plt.show()
plt.plot(list(range(N + 1)), meanNumberOfGuessesQuantum, label = 'Exact number')
plt.plot(list(range(N + 1)), approxYQ, label = 'Exponential regression')
plt.title('Expected number of secrets to enumerate - quantum')
plt.yscale("log", base = 2)
plt.xlabel('Number of positions')
plt.ylabel('Number of guesses')

plt.legend()