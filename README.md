# ImprovedKeyEnumeration
The source code for the paper "Further Improvements of the Estimation of Key Enumeration with Applications to Solving LWE", which in turn is an extension of the paper "Improved Estimation of Key Enumeration with Applications to Solving LWE", both by Alessandro Budroni and Erik Mårtensson.

Basic Run Instructions

The 3 enumeration files are pure Python files. Run EnumerationLarge.py to reproduce Figure 1 and Figure 2 from the paper. Run EnumerationLargeNewIdeas.py to reproduce Figure 3. EnumerationSmall.py is mainly added for completeness and implements enumeration in the naive way on a small scale. It can be used as a sanity-check when testing the output of the more complicated but efficient enumeration algorithms of the other files.

The 3 estimates files are Sage files. Other than Sage, they require the lattice estimator (https://github.com/malb/lattice-estimator), included in this repository. Make sure to use the lattice estimator from this repository! To integrate our enumeration efficiently we needed to make minor changes to one of the lattice estimator files. estimates.py can be used to reproduce Table 4 of our paper. estimates_original.py can be used to reproduce Table 1 of our paper. estimates_ISIT_version.py can be used to reproduce Table 2 of our paper. Basic run instructions for each files are included at the top of each file.

estimates_original.py is taken from the source code of "Quantum Augmented Dual Attack" by Martin Albrecht and Yixin Shen (https://eprint.iacr.org/2022/656). The other 2 estimates files added our improved enumeration algorithms on top of their implementation.
