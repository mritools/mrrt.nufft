Non-uniform fast Fourier transform in Python
============================================
This library provides a higher performance CPU/GPU NUFFT for Python.

This library started as a port of the Matlab NUFFT code in the
[Michigan image reconstruction toolbox] written by Jeff Fessler and his
students, but has been substantially overhauled and GPU support has been added.

The library does not implement all NUFFT variants, but only the following two
cases:

1.) transformation from a uniform spatial grid to non-uniformly sampled
frequency domain.

2.) Inverse transformation from non-uniform Fourier samples to a uniformly
spaced spatial grid.

Those interested in other NUFFT types may want to consider the
[NFFT library] which has an unofficial python wrapper via [pyNFFT].

The transforms are implemented in both single and double precision variants.

Both a low memory lookup table-based implementation and a fully precomputed
sparse matrix-based implementations are available.

See [Copying] and [LICENSES_bundled.txt] for full license info.

Related Software
================

Another Python-based implementation that has both CPU and GPU support is
available in the [sigpy]() package. The sigpy implementation of the NUFFT is
fairly compact as it uses [Numba]() to provide just-in-time compilation for
both the CPU and GPU variants from a common code base.

In contrast ``mrrt.nufft`` uses pre-compiled C code for the CPU variant and the
GPU kernels are compiled at run time using NVIDIA's run-time compilation
(NVRTC) as provided by [cupy.RawKernel](https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.RawKernel.html).

The [NFFT library] implements a more extensive set of non-uniform Fourier
transform variants. It has an unofficial python wrapper via [pyNFFT]. At the
time of writing it is CPU only.

A Matlab-based CPU-based implementation of the NUFFT is available in the
[Michigan image reconstruction toolbox]

A GPU based implementation with a Matlab interface is avialable as [gpuNUFFT].

The Flatiron Institute implemented [FINUFFT] which is a C++ library with
Fortran, Matlab and Python interfaces.

Some C/C++ MRI image reconstruction toolboxes also provide NUFFT
implementations: [Gadgetron] and the Berkley Advanced Reconstruction Toolbox
([BART]).


Basic Usage
===========
For those interested in iterative MR image reconstruction it is recommended to
use the simplified interface provided by:

TODO


Documentation
=============
TODO


Installation
=============

Binary packages have not yet been built and uploaded to PyPI or conda-forge,
but the package can be built from source tarballs hosted on PyPI.

```
pip install mrrt.utils
```

Required Dependencies
---------------------
- [NumPy]  (>=1.14)
- [SciPy]  (>=0.19)
- [Cython]  (>=0.29.13)
- [mrrt.utils]

Recommended Dependencies
------------------------
- [Matplotlib]  (for plotting)
- [pyFFTW]  (>=0.11) (enable faster FFTS than numpy.fft)
- [CuPy]  (>=6.1) (required for the GPU implementation)
- [jinja2]  (required for GPU implementation)


[BART]: https://mrirecon.github.io/bart/
[Copying]: https://github.com/mritools/mrrt.nufft/blob/master/COPYING
[CuPy]: https://github.com/cupy/cupy
[Cython]: https://cython.org/
[FINUFFT]: https://finufft.readthedocs.io/en/latest/index.html
[Gadgetron]: https://gadgetron.github.io/
[gpuNUFFT]: https://github.com/andyschwarzl/gpuNUFFT
[jinja2]: https://palletsprojects.com/p/jinja/
[LICENSES_bundled.txt]: https://github.com/mritools/mrrt.nufft/blob/master/LICENSES_bundled.txt
[Matplotlib]: https://matplotlib.org
[Michigan image reconstruction toolbox]: https://web.eecs.umich.edu/~fessler/code/
[mrrt.utils]: https://github.com/mritools/mrrt.utils
[NFFT library]: https://www-user.tu-chemnitz.de/~potts/nfft/
[NumPy]: https://github.com/numpy/numpy
[pyFFTW]: https://matplotlib.org
[pyNFFT]: https://github.com/pyNFFT/pyNFFT
[pytest]: https://docs.pytest.org/en/latest/
[SciPy]: https://scipy.org
