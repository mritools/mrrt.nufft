NUFFT
=====
High performance CPU/GPU NUFFT operator for Python.

This library started as a port of the Matlab NUFFT code in the
[Michigan image reconstruction toolbox](https://web.eecs.umich.edu/~fessler/code/),
but has been substantially overhauled and GPU support has been added.

See COPYING for more license info.

Basic Usage
===========
It is recommended to use the simplified interface provided by:

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
- [cupy](https://github.com/cupy/cupy)  (required for the GPU implementation)
