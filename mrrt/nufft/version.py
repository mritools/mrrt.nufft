from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 2
_version_micro = 2  # use "" for first of series, number for 1 and above
_version_extra = ""  # use "dev0" for developemnt, "" for full release

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = ".".join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = "Non-uniform FFT in 1D, 2D and 3D for CPU and GPU (CUDA)"
# Long description will go up on the pypi page
long_description = """

mrrt.nufft
==========
mrrt.nufft includes:

- 1D, 2D and 3D Transform from uniformly spaced spatial grid to non-uniformly spaced Fourier samples.
- 1D, 2D and 3D Inverse Transform non-uniformly spaced Fourier samples to uniformly sampled spatial.
- All transforms have both low memory and sparse-matrix (precomputed) variants.
- Transforms can be applied to NumPy arrays (CPU) or to CuPy arrays (GPU).

.. _README: https://github.com/mritools/mrrt.nufft/blob/master/README.md

License
=======
``mrrt.nufft`` is licensed under the terms of the BSD 3-clause license. See
the file "LICENSE.txt" and "LICENSES_bundled.txt" for information on the
history of this software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2014-2020,
Gregory R. Lee, Cincinnati Children's Hospital Medical Center.
"""

NAME = "mrrt.nufft"
MAINTAINER = "Gregory R. Lee"
MAINTAINER_EMAIL = "grlee77@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/mritools/mrrt.nufft"
DOWNLOAD_URL = ""
LICENSE = "BSD"
AUTHOR = "Gregory R. Lee"
AUTHOR_EMAIL = "grlee77@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {
    "mrrt.nufft": [
        pjoin("tests", "*"),
        pjoin("tests", "data", "*"),
        pjoin("cuda", "jinja", "*"),
    ]
}
REQUIRES = ["numpy", "cython", "mrrt.utils"]
PYTHON_REQUIRES = ">= 3.6"
