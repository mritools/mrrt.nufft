NUFFT
=====
High performance NUFFT based on a port of Jeff Fessler's Matlab NUFFT code
from his [image reconstruction toolbox](https://web.eecs.umich.edu/~fessler/code/).

Much of the code in this module is a based on Matlab routines originally
created by Jeff Fessler and his students at the University of Michigan.  See
COPYING for more license info.

"""

Basic Usage
===========
TODO

Documentation
=============
TODO

Required Dependencies
---------------------
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://scipy.org)
- [mrrt.utils](https://github.com/grlee77/mrrt.utils)

Recommended Dependencies
------------------------
- [matplotlib](https://matplotlib.org)  (for plotting)
- [pyFFTW](https://github.com/pyFFTW/pyFFTW) (enable faster FFTS than numpy.fft)
- [mrrt.cuda](https://github.com/grlee77/mrrt.cuda)  (code to support the GPU implementation is here)
