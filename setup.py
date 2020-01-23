import os
import platform
import sys
from os.path import join as pjoin
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

from setup_helpers import add_flag_checking, make_np_ext_builder

PACKAGES = find_packages()

# Get version and release info, which is all stored in mrrt/nufft/version.py
ver_file = os.path.join("mrrt", "nufft", "version.py")
with open(ver_file) as f:
    exec(f.read())
# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
SETUP_REQUIRES = ["setuptools >= 24.2.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []


# Note if rpath errors related to libgomp occur on OsX, try adding the path
# containing libgomp to the link arguments.  e.g.
# LDFLAGS="$LDFLAGS -Wl,-rpath,/Users/lee8rx/anaconda/lib" pip install -e . -v
#
# On OS X, recent clang does have OpenMP support.
# From conda-forge, one can install the clangdev and openmp packages to get a
# version supporting it.

# Add openmp flags if they work
simple_test_c = """int main(int argc, char** argv) { return(0); }"""
omp_test_c = """#include <omp.h>
int main(int argc, char** argv) { return(0); }"""

msc_flag_defines = []
gcc_flag_defines = []
# TODO: commented out SSE2 stuff. I don't think it was being used
# gcc_flag_defines = [
#     [["-msse2", "-mfpmath=sse"], [], simple_test_c, "USING_GCC_SSE2"]
# ]
if "NUFFT_DISABLE_OPENMP" not in os.environ:
    msc_flag_defines += [[["/openmp"], [], omp_test_c, "HAVE_VC_OPENMP"]]
    gcc_flag_defines += [
        [["-fopenmp"], ["-fopenmp"], omp_test_c, "HAVE_OPENMP"]
    ]

# # Test if it is a 32 bits version
# if not sys.maxsize > 2 ** 32:
#     # This flag is needed only on 32 bits
#     msc_flag_defines += [[["/arch:SSE2"], [], simple_test_c, "USING_VC_SSE2"]]

flag_defines = (
    msc_flag_defines
    if "msc" in platform.python_compiler().lower()
    else gcc_flag_defines
)

extbuilder = add_flag_checking(build_ext, flag_defines, pjoin("mrrt", "nufft"))

# add np.get_include() at build time, not during setup.py execution.
extbuilder = make_np_ext_builder(extbuilder)

# cython check
try:
    import cython

    # check that cython version is > 0.21
    cython_version = cython.__version__
    if float(cython_version.partition(".")[2][:2]) < 21:
        raise ImportError
    build_cython = True
except ImportError:
    build_cython = False
    # TODO: allow a python-only installation with reduced functionality
    raise EnvironmentError(
        """
        cython could not be found.  Compilation of mrrt.nufft requires Cython
        version >= 0.21.
        Install or upgrade cython via:
        pip install cython --upgrade
        """
    )


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


extra_compile_args = ["-ffast-math"]
# if use_OpenMP:
cmdclass = {"build_ext": extbuilder, "test": PyTest}

src_path = pjoin("mrrt", "nufft")

# C extensions for table-based NUFFT
ext_nufft = Extension(
    "mrrt.nufft._extensions._nufft_table",
    sources=[
        "mrrt/nufft/_extensions/c/nufft_table.c",
        "mrrt/nufft/_extensions/_nufft_table.pyx",
    ],
    depends=[
        "mrrt/nufft/_extensions/c/nufft_table.template.c",
        "mrrt/nufft/_extensions/c/nufft_table.template.h",
        "mrrt/nufft/_extensions/c/templating.h"
        "mrrt/nufft/_extensions/c/nufft_table.h",
        "mrrt/nufft/_extensions/_nufft_table.pxd",
    ],
    language="c",
    extra_compile_args=extra_compile_args,
    extra_link_args=[],  # extra_link_args,
    include_dirs=[pjoin(src_path, "_extensions", "c")],
)  # numpy_include


ext_modules = [ext_nufft]

c_macros = [("PY_EXTENSION", None)]
cython_macros = []
cythonize_opts = {}
if os.environ.get("CYTHON_TRACE"):
    cythonize_opts["linetrace"] = True
    cython_macros.append(("CYTHON_TRACE_NOGIL", 1))

ext_modules = cythonize(
    ext_modules, compiler_directives=cythonize_opts, language_level=2
)

# setup(
#     name="mrrt.nufft",
#     author="Gregory R. Lee",
#     version=VERSION,
#     ext_modules=ext_modules,
#     cmdclass=cmdclass,
#     packages=find_packages(),
#     namespace_package=["mrrt"],
#     # scripts=[],
#     # since the package has c code, the egg cannot be zipped
#     zip_safe=False,
#     # data_files=[('mrrt.nufft/data', glob('mrrt.nufft/data/*.npy')), ],
#     package_data={"mrrt.nufft": [pjoin("tests", "data", "*")]},
#     # maintainer="",
#     # maintainer_email="grlee77@gmail.com",
#     url="https://github.com/mritools/mrrt.nufft",
#     download_url="https://github.com/mritools/mrrt.nufft/releases",
#     license="BSD-3",
#     description="Non-uniform FFT in 1D, 2D and 3D for CPU and GPU (CUDA)",
#     long_description="""\
#         mrrt.nufft includes:

#         * 1D, 2D and 3D Transform from uniformly spaced spatial grid to
#         non-uniformly spaced Fourier samples.
#         * 1D, 2D and 3D Inverse Transform non-uniformly spaced Fourier samples
#         to uniformly sampled spatial.
#         * All transforms have both low memory and sparse-matrix (precomputed)
#         variants.
#         * Transforms can be applied to NumPy arrays (CPU) or to CuPy arrays
#         (GPU).
#         """,
#     keywords=[
#         "non-uniform fast Fourier transform",
#         "NFFT",
#         "NUFFT",
#         "scientific",
#         "non-cartesian MRI",
#         "magnetic resonance imaging",
#     ],
#     classifiers=[
#         "Development Status :: 4 - Beta",
#         "Intended Audience :: Science/Research",
#         "License :: OSI Approved :: BSD 3-Clause 'New' or 'Revised' License",
#         "Operating System :: OS Independent",
#         "Programming Language :: C",
#         "Programming Language :: CUDA",
#         "Programming Language :: Python",
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3.6",
#         "Programming Language :: Python :: 3.7",
#         "Programming Language :: Python :: 3.8",
#         "Programming Language :: Python :: 3 :: Only",
#         "Programming Language :: Python :: Implementation :: CPython",
#         "Topic :: Software Development :: Libraries :: Python Modules",
#     ],
#     platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
#     tests_require=["pytest"],
#     install_requires=["numpy>=1.14.5"],
#     setup_requires=["numpy>=1.14.5"],
#     python_requires=">=3.6",
# )

opts = dict(
    name=NAME,
    namespace_package=["mrrt"],
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    install_requires=REQUIRES,
    python_requires=PYTHON_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    requires=REQUIRES,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    keywords=[
        "non-uniform fast Fourier transform",
        "NFFT",
        "NUFFT",
        "scientific",
        "non-cartesian MRI",
        "magnetic resonance imaging",
    ],
)

if __name__ == "__main__":
    setup(**opts)
