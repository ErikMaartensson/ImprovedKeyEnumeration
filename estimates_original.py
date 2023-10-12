# -*- coding: utf-8 -*-
"""
Run like this::

    sage: attach("estimates_original.py")
    sage: %time results = runall()
    sage: save(results, "estimates.sobj") # To save the results as an object
    sage: print(results_table(results))

To load the results from a saved object run:
    sage: load(results, "estimates.sobj") # Load results from a saved object

"""
from sage.all import sqrt, log, exp, tanh, coth, e, pi, RR, ZZ

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

        H = cls.Hf(params.Xs)

        coeff = 1 / (1 - exp(-1 / 2 / params.Xs.stddev**2))
        tmp_alpha = pi**2 * params.Xs.stddev**2
        tmp_a = exp(8 * tmp_alpha * exp(-2 * tmp_alpha) * tanh(tmp_alpha)).n(30)
        T_guess = coeff * (
            ((2 * tmp_a / sqrt(e)) ** k_enum)
            * (2 ** (k_enum * H))
            * (cls.T_fftf(k_fft, p) + cls.T_tablef(N))
        )

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

        #for p in early_abort_range(2, params.q):
        for p in early_abort_range(2, 20):
            # print(p[0])
            for k_enum in early_abort_range(0, params.n, 5):
                for k_fft in early_abort_range(0, params.n - k_enum[0], 5):
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
                                red_cost_model=red_cost_model,
                            )
                            it.update(cost)
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

        H = cls.Hf(params.Xs)

        coeff = 7 / (
            ZZ(6) * (1 - exp(-1 / (3 * params.Xs.stddev**2))) ** (3 / ZZ(2))
        )
        tmp_alpha = pi**2 * params.Xs.stddev**2
        tmp_a = exp(8 * tmp_alpha * exp(-2 * tmp_alpha) * tanh(tmp_alpha)).n(30)
        T_guess = (
            coeff
            * ((27 * tmp_a**2 / (8 * e)) ** (k_enum / 4))
            * sqrt(2 ** (k_enum * H) * p ** (k_fft) * cls.T_tablef(N))
            + N
        )

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


def runall(
    schemes=(
        Kyber512,
        Kyber768,
        Kyber1024,
        LightSaber,
        Saber,
        FireSaber,
        TFHE630,
        TFHE1024,
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

    for scheme in schemes:
        results[scheme] = {}
        print(f"{repr(scheme)}")
        for nn in nns:
            cost = MATZOV()(scheme, red_cost_model=RC.MATZOV.__class__(nn=nn))
            results[scheme][(nn, "classical")] = cost
            print(f" nn: {nn},  cost: {repr(cost)}")
            cost = QMATZOV()(scheme, red_cost_model=RC.MATZOV.__class__(nn=nn))
            print(f" nn: {nn}, qcost: {repr(cost)}")
            results[scheme][(nn, "quantum")] = cost

        cost = MATZOV()(scheme, red_cost_model=RC.ADPS16)
        print(f" C0, cost: {repr(cost)}")
        results[scheme][("C0", "classical")] = cost

        cost = MATZOV()(scheme, red_cost_model=ChaLoy21())
        print(f" Q0, cost: {repr(cost)}")
        results[scheme][("Q0", "classical")] = cost

        cost = QMATZOV()(scheme, red_cost_model=ChaLoy21())
        print(f"Q0, qcost: {repr(cost)}")
        results[scheme][("Q0", "quantum")] = cost

    return results


def results_table(results, fmt=None):
    import tabulate

    rows = []

    def pp(cost):
        return round(log(cost["rop"], 2), 1)

    for scheme, costs in results.items():
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
