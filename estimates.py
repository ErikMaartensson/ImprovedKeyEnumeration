# -*- coding: utf-8 -*-
"""
Run like this::

    sage: attach("estimates.py")
    sage: %time results = runall()
    sage: save(results, "estimates.sobj") # To save the results as an object
    sage: print(results_table(results))

To load the results from a saved object run:
    sage: load(results, "estimates.sobj") # Load results from a saved object

"""

import tabulate

# Some extra imports needed to calculate the expected cost of enumeration
import numpy as np
import math
from scipy.special import zeta

from sage.all import sqrt, log, exp, coth, e, pi, RR, ZZ#, tanh

from estimator.estimator.cost import Cost
from estimator.estimator.lwe_parameters import LWEParameters
from estimator.estimator.reduction import delta as deltaf
from estimator.estimator.reduction import RC, ReductionCost
from estimator.estimator.conf import red_cost_model as red_cost_model_default
from estimator.estimator.util import local_minimum, early_abort_range
from estimator.estimator.io import Logging
from estimator.estimator.schemes import (
    Kyber512,
    Kyber768,
    Kyber1024,
    LightSaber,
    Saber,
    FireSaber,
)
from estimator.estimator.schemes import TFHE630, TFHE1024


class ChaLoy21(ReductionCost):

    __name__ = "ChaLoy21"
    short_vectors = ReductionCost._short_vectors_sieve

    def __call__(self, beta, d, B=None):
        """
        :param beta: Block size ≥ 2.
        :param d: Lattice dimension.
        :param B: Bit-size of entries.
        """

        return ZZ(2) ** RR(0.2570 * beta)


class MATZOV:
    """ """

    C_prog = 1.0 / (1 - 2.0 ** (-0.292))  # p.37
    C_mul = 32**2  # p.37
    C_add = 5 * 32  # guessing based on C_mul

    @classmethod
    def T_fftf(cls, k, p):
        """
        The time complexity of the FFT in dimension `k` with modulus `p`.

        :param k: Dimension
        :param p: Modulus ≥ 2

        """
        return cls.C_mul * k * p ** (k + 1)  # Theorem 7.6, p.38

    @classmethod
    def T_tablef(cls, D):
        """
        Time complexity of updating the table in each iteration.

        :param D: Number of nonzero entries

        """
        return 4 * cls.C_add * D  # Theorem 7.6, p.39

    @classmethod
    def Nf(cls, params, m, beta_bkz, beta_sieve, k_enum, k_fft, p):
        """
        Required number of samples to distinguish with advantage.

        :param params: LWE parameters
        :param m:
        :param beta_bkz: Block size used for BKZ reduction
        :param beta_sieve: Block size used for sampling
        :param k_enum: Guessing dimension
        :param k_fft: FFT dimension
        :param p: FFT modulus

        """
        mu = 0.5
        k_lat = params.n - k_fft - k_enum  # p.15

        # p.39
        lsigma_s = (
            params.Xe.stddev ** (m / (m + k_lat))
            * (params.Xs.stddev * params.q) ** (k_lat / (m + k_lat))
            * sqrt(4 / 3.0)
            * sqrt(beta_sieve / 2 / pi / e)
            * deltaf(beta_bkz) ** (m + k_lat - beta_sieve)
        )

        # p.29, we're ignoring O()
        N = (
            exp(4 * (lsigma_s * pi / params.q) ** 2)
            * exp(k_fft / 3.0 * (params.Xs.stddev * pi / p) ** 2)
            * (k_enum * cls.Hf(params.Xs) + k_fft * log(p) + log(1 / mu))
        )

        return RR(N)

    @staticmethod
    def Hf(Xs):
        return RR(
            1 / 2
            + log(sqrt(2 * pi) * Xs.stddev)
            + log(coth(pi**2 * Xs.stddev**2))
        ) / log(2.0)

    @classmethod
    def cost(
        cls,
        beta,
        params,
        m=None,
        p=2,
        k_enum=0,
        k_fft=0,
        enumerationCostList=None,
        expFit=None,
        beta_sieve=None,
        red_cost_model=red_cost_model_default,
    ):
        """
        Theorem 7.6

        """

        if m is None:
            m = params.n

        k_lat = params.n - k_fft - k_enum  # p.15

        # We assume here that β_sieve ≈ β
        N = cls.Nf(
            params,
            m,
            beta,
            beta_sieve if beta_sieve else beta,
            k_enum,
            k_fft,
            p,
        )
        rho, T_sample, _, beta_sieve = red_cost_model.short_vectors(
            beta, N=N, d=k_lat + m, sieve_dim=beta_sieve
        )

        # H = cls.Hf(params.Xs)
        
        # Hard-coded cost for Kyber512 - classical seeting
#        enumerationCost = [1, 2.53125, 10.072265625, 48.558837890625,250.01622772216797, 1321.592160642147, 7082.739818695933, 38293.15297269059, 208342.01823904362, 1139014.9008441723, 6251129.409546113,
# 34416606.72080199, 189993676.831495, 1051245078.569889,5828144794.728788,32367696628.317352,180036756003.4318,1002780833562.2446,5592234404725.372,31221040428280.375,174480766243802.62,
# 975994689518323.2,5464035718301343.0,3.0613739441597736e+16,1.7164457077164262e+17,9.630117628579959e+17,5.40630371518026e+18,3.0368144509137805e+19,1.7067446152356138e+20,9.597019966231212e+20,
# 5.398937996991184e+21,3.0385900365984553e+22,1.710868125195202e+23,9.636768015021387e+23,5.430094106325939e+24,3.060805888489751e+25,1.7258727674833738e+26, 9.734620012052489e+26,5.492368928737431e+27,3.099735192537661e+28,1.7498783227749864e+29]

        # expFit = [ 1.71470387, -1.40229344] # Exponential regression model used when k_enum > 40
        
        enumerationFactor = 1
        
        if k_enum < len(enumerationCostList):
            enumerationFactor = enumerationCostList[k_enum]
        else:
            enumerationFactor = exp(expFit[0]*k_enum + expFit[1])

        T_guess = enumerationFactor*((cls.T_fftf(k_fft, p) + cls.T_tablef(N)))

#        coeff = 1 / (1 - exp(-1 / 2 / params.Xs.stddev**2))
#        tmp_alpha = pi**2 * params.Xs.stddev**2
#        tmp_a = exp(8 * tmp_alpha * exp(-2 * tmp_alpha) * tanh(tmp_alpha)).n(30)
#        T_guess = coeff * (
#            ((2 * tmp_a / sqrt(e)) ** k_enum)
#            * (2 ** (k_enum * H))
#            * (cls.T_fftf(k_fft, p) + cls.T_tablef(N))
#        )

        # print(T_guess)

        cost = Cost(rop=T_sample + T_guess, problem=params)
        cost["red"] = T_sample
        cost["guess"] = T_guess
        cost["beta"] = beta
        cost["p"] = p
        cost["zeta"] = k_enum
        cost["t"] = k_fft
        cost["beta_"] = beta_sieve
        cost["N"] = N
        cost["m"] = m

        cost.register_impermanent(
            {"β'": False, "ζ": False, "t": False}, rop=True, p=False, N=False
        )
        return cost

    def __call__(
        self,
        params: LWEParameters,
        enumerationCostList=None,
        expFit=None,
        red_cost_model=red_cost_model_default,
        log_level=1,
    ):
        """
        Optimizes cost of dual attack as presented in [Matzov22]_.

        :param params: LWE parameters
        :param red_cost_model: How to cost lattice reduction

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``red``: Number of word operations in lattice reduction and
                   short vector sampling.
        - ``guess``: Number of word operations in guessing and FFT.
        - ``β``: BKZ block size.
        - ``ζ``: Number of guessed coordinates.
        - ``t``: Number of coordinates in FFT part mod `p`.
        - ``d``: Lattice dimension.

        """
        params = params.normalize()

        for p in early_abort_range(2, 10 + 1):
        #for p in early_abort_range(2, params.q, 20):
        #for p in early_abort_range(2, params.q):
            # print(p[0])
            for k_enum in early_abort_range(0, params.n, 5):
            #for k_enum in early_abort_range(0, params.n, 2):
                # print(k_enum[0])
                for k_fft in early_abort_range(0, params.n - k_enum[0], 5):
                #for k_fft in early_abort_range(0, params.n - k_enum[0], 2):
                    with local_minimum(
                        40, params.n, log_level=log_level + 4
                    ) as it:
                        for beta in it:
                            cost = self.cost(
                                beta,
                                params,
                                p=p[0],
                                k_enum=k_enum[0],
                                k_fft=k_fft[0],
                                enumerationCostList=enumerationCostList,
                                expFit=expFit,
                                red_cost_model=red_cost_model,
                            )
                            # print(type(cost))
                            it.update(cost)
                            # print("Testing...")
                        Logging.log(
                            "dual",
                            log_level + 3,
                            f"t: {k_fft[0]}, {repr(it.y)}",
                        )
                        k_fft[1].update(it.y)
                Logging.log(
                    "dual", log_level + 2, f"ζ: {k_enum[0]}, {repr(k_fft[1].y)}"
                )
                k_enum[1].update(k_fft[1].y)
                #if t == 0 then p is irrelevant, so we early abort that loop if that's the case once we hit t==0 twice.
#                if p[1].y["t"] == 0 and p[0] > 2:
#                    break
#                Logging.log(
#                    "dual", log_level + 2, f"ζ: {k_enum[0]}, {repr(k_fft[1].y)}"
#                )
#                k_enum[1].update(k_fft[1].y)
#                if p[1].y["t"] == 0 and p[0] > 2:
#                    break
            Logging.log("dual", log_level + 1, f"p:{p[0]}, {repr(k_enum[1].y)}")
            p[1].update(k_enum[1].y)
        Logging.log("dual", log_level, f"{repr(p[1].y)}")
        return p[1].y


class QMATZOV(MATZOV):
    @classmethod
    def cost(
        cls,
        beta,
        params,
        m=None,
        p=2,
        k_enum=0,
        k_fft=0,
        enumerationCostList=None,
        expFit=None,
        beta_sieve=None,
        red_cost_model=red_cost_model_default,
    ):
        """
        Theorem 7.6

        """

        if m is None:
            m = params.n

        k_lat = params.n - k_fft - k_enum  # p.15

        # We assume here that β_sieve ≈ β
        N = cls.Nf(
            params,
            m,
            beta,
            beta_sieve if beta_sieve else beta,
            k_enum,
            k_fft,
            p,
        )
        rho, T_sample, _, beta_sieve = red_cost_model.short_vectors(
            beta, N=N, d=k_lat + m, sieve_dim=beta_sieve
        )

        # H = cls.Hf(params.Xs)
        
        # enumerationCost = [1, 1.5266502260621266, 2.893536614521863, 6.092746109993902, 13.365189613776643, 29.81695884755893, 67.09106790289934, 151.76424691820338, 344.5877582400435,784.6278544031683, 1790.6270494721282, 4093.9681405933466, 9374.530820797194, 21494.12890643851,49337.46499689194,113359.46310474361,260682.4128233466,599924.550872062,1381584.1429151213,3183641.3473206647,7340263.7356094,16932383.248319335,39077292.82384563,90222374.8143642,208388091.9837503,481490299.49685,1112875955.5623553,2573003340.266295,5950578217.703946,13765594764.823517,31852267024.477028,73720542954.20332,170660642840.46555,395156660558.13007,915147699348.9127, 2119797308922.306, 4911050717799.14, 11379612100721.598, 26372407355477.234, 61127687225544.54, 141706204842195.03]

        # expfit = [ 0.83175539, -0.76988718] # Exponential regression model used when k_enum > 40
        
        enumerationFactor = 1
        
        if k_enum < len(enumerationCostList):
            enumerationFactor = enumerationCostList[k_enum]
        else:
            enumerationFactor = exp(expFit[0]*k_enum + expFit[1])
            
        # print(enumerationFactor * sqrt(p ** (k_fft) * cls.T_tablef(N)))
        T_guess = enumerationFactor * sqrt(p ** (k_fft) * cls.T_tablef(N))
        
#        coeff = 7 / (
#            ZZ(6) * (1 - exp(-1 / (3 * params.Xs.stddev**2))) ** (3 / ZZ(2))
#        )
#        tmp_alpha = pi**2 * params.Xs.stddev**2
#        tmp_a = exp(8 * tmp_alpha * exp(-2 * tmp_alpha) * tanh(tmp_alpha)).n(30)
#        T_guess = (
#            coeff
#            * ((27 * tmp_a**2 / sqrt(8 * e)) ** (k_enum / 4))
#            * sqrt(2 ** (k_enum * H) * p ** (k_fft) * cls.T_tablef(N))
#            + N
#        )
        
        # print(T_guess)

        cost = Cost(rop=T_sample + T_guess, problem=params)
        cost["red"] = T_sample
        cost["guess"] = T_guess
        cost["beta"] = beta
        cost["p"] = p
        cost["zeta"] = k_enum
        cost["t"] = k_fft
        cost["beta_"] = beta_sieve
        cost["N"] = N
        cost["m"] = m

        cost.register_impermanent(
            {"β'": False, "ζ": False, "t": False}, rop=True, p=False, N=False
        )
        return cost



# Some functions used to compute the expected cost of enumeration
        
# Iterate over all length-tuples of non-negative integers that sum to total_sum - see https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation

# Computes the sum of sqrt(k) where k goes from 1 to n
def sumOfSqrts(n):
    # Use the exact expression for very small values of n
    if n < 40:
        return sum ([math.sqrt(k) for k in range(1, n + 1)])
    return 2/3*n**1.5 + zeta(-0.5) + 0.5*n**0.5 + 1/24*n**(-0.5) - 1/1920*n**(-5/2) + 1/9216*n**(-9/2) # For larger values of n we use an approximation formula

# Calculates the expected number of guesses of the secret to iterate over - the general case - classical - with aborted enumeration
def exactNumberOfEnumerationQuantumAborted(p, freq, n, successProbability):
    
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
    
    partialProbability = 0
    partialSum = 0
    estimatedEnumeration = 0
    for k in range(numberOfTerms):
        fk = int(sortedFreq[k])
        pk = sortedP[k]
        if partialProbability > successProbability: # Abort here if we go beyond the success probability requirement
            break    
        estimatedEnumeration = estimatedEnumeration + (sumOfSqrts(partialSum + fk) - sumOfSqrts(partialSum))*pk
        partialSum = partialSum + fk
        partialProbability = partialProbability + fk*pk
    
    estimatedEnumeration = (estimatedEnumeration + (1 - partialProbability)*math.sqrt(partialSum)) # / partialProbability# Calculate the total cost - abortion cost
    
    # We divide by p twice - fix this!
    
    return estimatedEnumeration 

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

# Calculates the expected number of guesses of the secret to iterate over - the general case - classical - with aborted enumeration
def exactNumberOfEnumerationAborted(p, freq, n, successProbability):
    
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
    
    partialProbability = 0
    partialSum = 0
    estimatedEnumeration = 0
    for k in range(numberOfTerms):
        fk = int(sortedFreq[k])
        pk = sortedP[k]
        if partialProbability > successProbability: # Abort here if we go beyond the success probability requirement
            break    
        estimatedEnumeration = estimatedEnumeration + (partialSum*fk + fk*(fk + 1)/2)*pk
        partialSum = partialSum + fk
        partialProbability = partialProbability + fk*pk
    
    estimatedEnumeration = (estimatedEnumeration + (1 - partialProbability)*partialSum) # / partialProbability# Calculate the total cost - abortion cost
    
    # We divide by p twice - fix this!
    
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

def exponentialApproximation(x, y):
    N = len(y)
    logY = [math.log(y[k]) for k in range(N)]
    # x = list(range(1, N + 1))
    fit = np.polyfit(x, logY, 1)
    return fit

def getQuantumEnumerationCost(p, freq, N, successProbability):
    enumerationList = [exactNumberOfEnumerationQuantumAborted(p, freq, k, successProbability) for k in range(N + 1)]
    x = list(range(N + 1))
    fit = exponentialApproximation(x, enumerationList)
    return enumerationList, fit

def getClassicalEnumerationCost(p, freq, N, successProbability):
    enumerationList = [exactNumberOfEnumerationAborted(p, freq, k, successProbability) for k in range(N + 1)]
    x = list(range(N + 1))
    fit = exponentialApproximation(x, enumerationList)
    return enumerationList, fit

def runall(
    schemes=(
        [Kyber512, np.array([1, 6, 15, 20])/64, np.array([2, 2, 2, 1])],  
        [Kyber768, np.array([1, 4, 6])/16, np.array([2, 2, 1])],
        [Kyber1024, np.array([1, 4, 6])/16, np.array([2, 2, 1])],
        [LightSaber, np.array([1, 10, 45, 120, 210, 252])/1024, np.array([2, 2, 2, 2, 2, 1])],
        [Saber, np.array([1, 8, 28, 56, 70])/256, np.array([2, 2, 2, 2, 1])], 
        [FireSaber, np.array([1, 6, 15, 20])/64, np.array([2, 2, 2, 1])]
        #[TFHE630, np.array([1/2, 1/2]), np.array([1, 1])],
        #[TFHE1024, np.array([1/2, 1/2]), np.array([1, 1])]
    ),
    # schemes=(Kyber512, Kyber768, Kyber1024, LightSaber, Saber, FireSaber),
    nns=(
        "list_decoding-naive_classical",
        "list_decoding-classical",
        "list_decoding-naive_quantum",
        "list_decoding-ge19",
    ),
):

    results = {}
    successProbability = 0.5 # The success probability for the enumeration part - set this to 1 for full enumeration

    for scheme in schemes:

        results[scheme[0]] = {}
        
        # Compute the cost of enumeration for the scheme, classically and quantumly
        prob = scheme[1]
        freq = scheme[2]
        enumCost, fit = getClassicalEnumerationCost(prob, freq, 40, successProbability)
        # enumCost, fit = getClassicalEnumerationCost(prob, freq, 40, 1)
        enumCostQ, fitQ = getQuantumEnumerationCost(prob, freq, 40, successProbability)
        
#        print(scheme)
#        for k in range(len(enumCost)):
#            print("n= " + str(k) + ", enum[n] = " + str(round(math.log(enumCost[k], 2), 1)))

        for nn in nns:
            cost = MATZOV()(scheme[0], enumerationCostList = enumCost, expFit = fit, red_cost_model=RC.MATZOV.__class__(nn=nn))
            results[scheme[0]][(nn, "classical")] = cost
            print(f" nn: {nn},  cost: {repr(cost)}")
            cost = QMATZOV()(scheme[0], enumCostQ, fitQ, red_cost_model=RC.MATZOV.__class__(nn=nn))
            print(f" nn: {nn}, qcost: {repr(cost)}")
            results[scheme[0]][(nn, "quantum")] = cost

        cost = MATZOV()(scheme[0], enumCost, fit, red_cost_model=RC.ADPS16)
        print(f" C0, cost: {repr(cost)}")
        results[scheme[0]][("C0", "classical")] = cost

        cost = MATZOV()(scheme[0], enumCost, fit, red_cost_model=ChaLoy21())
        print(f" Q0, cost: {repr(cost)}")
        results[scheme[0]][("Q0", "classical")] = cost

        cost = QMATZOV()(scheme[0], enumCostQ, fitQ, red_cost_model=ChaLoy21())
        print(f"Q0, qcost: {repr(cost)}")
        results[scheme[0]][("Q0", "quantum")] = cost
        print("Testing to print the cost")
        print(round(math.log(cost["rop"]/successProbability, 2), 1))

    return results


def results_table(results, fmt=None):
    # import tabulate

    rows = []
    successProbability = 0.5 # Set this one to the same value as in runall() to get results printed correctly

    def pp(cost):
        return round(math.log(cost["rop"]/successProbability, 2), 1)

    for scheme, costs in results.items():
        # row = scheme.tag
        # print(costs[("list_decoding-classical", "classical")]["rop"])
        row = [
            scheme.tag,
            pp(costs[("list_decoding-classical", "classical")]),
            pp(costs[("list_decoding-naive_classical", "classical")]),
            pp(costs[("C0", "classical")]),
            pp(costs[("list_decoding-ge19", "classical")]),
            pp(costs[("list_decoding-naive_quantum", "classical")]),
            pp(costs[("Q0", "classical")]),
            pp(costs[("list_decoding-naive_quantum", "quantum")]),
            pp(costs[("Q0", "quantum")]),
        ]
        rows.append(row)
    if fmt is None:
        return rows
    else:
        return tabulate.tabulate(
            results_table(results),
            headers=[
                "Scheme",
                "CC",
                "CN",
                "C0",
                "GE19",
                "QN",
                "Q0",
                "This work (QN)",
                "This work (Q0)",
            ],
            tablefmt="latex_booktabs",
            floatfmt=".1f",
        )
