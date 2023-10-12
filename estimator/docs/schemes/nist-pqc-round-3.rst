NIST PQC Round 3 Finalists
==========================

`Kyber <https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf>`__

::

    >>> from estimator import *
    >>> Kyber512
    LWEParameters(n=512, q=3329, Xs=D(σ=1.22), Xe=D(σ=1.22), m=512, tag='Kyber 512')
    >>> LWE.primal_bdd(Kyber512)
    rop: ≈2^140.3, red: ≈2^139.7, svp: ≈2^138.8, β: 391, η: 421, d: 1013, tag: bdd

::

    >>> from estimator import *
    >>> Kyber768
    LWEParameters(n=768, q=3329, Xs=D(σ=1.00), Xe=D(σ=1.00), m=768, tag='Kyber 768')
    >>> LWE.primal_bdd(Kyber768)
    rop: ≈2^201.0, red: ≈2^200.0, svp: ≈2^200.0, β: 606, η: 641, d: 1425, tag: bdd

::

    >>> from estimator import *
    >>> Kyber1024
    LWEParameters(n=1024, q=3329, Xs=D(σ=1.00), Xe=D(σ=1.00), m=1024, tag='Kyber 1024')
    >>> LWE.primal_bdd(Kyber1024)
    rop: ≈2^270.8, red: ≈2^269.9, svp: ≈2^269.7, β: 855, η: 890, d: 1873, tag: bdd

`Saber <https://www.esat.kuleuven.be/cosic/pqcrypto/saber/files/saberspecround3.pdf>`__

::

    >>> from estimator import *
    >>> LightSaber
    LWEParameters(n=512, q=8192, Xs=D(σ=1.58), Xe=D(σ=2.00), m=512, tag='LightSaber')
    >>> LWE.primal_bdd(LightSaber)
    rop: ≈2^137.8, red: ≈2^137.3, svp: ≈2^136.3, β: 382, η: 412, d: 1024, tag: bdd

::

    >>> from estimator import *
    >>> Saber
    LWEParameters(n=768, q=8192, Xs=D(σ=1.41), Xe=D(σ=2.00), m=768, tag='Saber')
    >>> LWE.primal_bdd(Saber)
    rop: ≈2^204.9, red: ≈2^203.9, svp: ≈2^204.0, β: 620, η: 655, d: 1475, tag: bdd

::

    >>> from estimator import *
    >>> FireSaber
    LWEParameters(n=1024, q=8192, Xs=D(σ=1.22), Xe=D(σ=2.00), m=1024, tag='FireSaber')
    >>> LWE.primal_bdd(FireSaber)
    rop: ≈2^271.8, red: ≈2^270.7, svp: ≈2^270.8, β: 858, η: 894, d: 1886, tag: bdd


`NTRU <https://ntru.org/f/ntru-20190330.pdf>`__

::

    >>> from estimator import *
    >>> NTRUHPS2048509Enc
    LWEParameters(n=508, q=2048, Xs=D(σ=0.82), Xe=D(σ=0.71), m=508, tag='NTRUHPS2048509Enc')
    >>> LWE.primal_bdd(NTRUHPS2048509Enc)
    rop: ≈2^131.1, red: ≈2^130.1, svp: ≈2^130.2, β: 357, η: 390, d: 916, tag: bdd

::

    >>> from estimator import *
    >>> NTRUHPS2048677Enc
    LWEParameters(n=676, q=2048, Xs=D(σ=0.82), Xe=D(σ=0.61), m=676, tag='NTRUHPS2048677Enc')
    >>> LWE.primal_bdd(NTRUHPS2048677Enc)
    rop: ≈2^170.8, red: ≈2^169.6, svp: ≈2^169.9, β: 498, η: 533, d: 1179, tag: bdd

::

    >>> from estimator import *
    >>> NTRUHPS4096821Enc
    LWEParameters(n=820, q=4096, Xs=D(σ=0.82), Xe=D(σ=0.79), m=820, tag='NTRUHPS4096821Enc')
    >>> LWE.primal_bdd(NTRUHPS4096821Enc)
    rop: ≈2^199.7, red: ≈2^198.7, svp: ≈2^198.6, β: 601, η: 636, d: 1485, tag: bdd

::

    >>> from estimator import *
    >>> NTRUHRSS701Enc
    LWEParameters(n=700, q=8192, Xs=D(σ=0.82), Xe=D(σ=0.82), m=700, tag='NTRUHRSS701')
    >>> LWE.primal_bdd(NTRUHRSS701Enc)
    rop: ≈2^158.9, red: ≈2^157.9, svp: ≈2^158.0, β: 455, η: 490, d: 1294, tag: bdd
