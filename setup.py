import os
import platform
import sys
from os.path import join as pjoin
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import versioneer

from setup_helpers import add_flag_checking, make_np_ext_builder

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
gcc_flag_defines = [
    [["-msse2", "-mfpmath=sse"], [], simple_test_c, "USING_GCC_SSE2"]
]
if "NUFFT_DISABLE_OPENMP" not in os.environ:
    msc_flag_defines += [[["/openmp"], [], omp_test_c, "HAVE_VC_OPENMP"]]
    gcc_flag_defines += [
        [["-fopenmp"], ["-fopenmp"], omp_test_c, "HAVE_OPENMP"]
    ]

# Test if it is a 32 bits version
if not sys.maxsize > 2 ** 32:
    # This flag is needed only on 32 bits
    msc_flag_defines += [[["/arch:SSE2"], [], simple_test_c, "USING_VC_SSE2"]]

flag_defines = (
    msc_flag_defines
    if "msc" in platform.python_compiler().lower()
    else gcc_flag_defines
)

extbuilder = add_flag_checking(build_ext, flag_defines, pjoin("pyir", "nufft"))

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
except:
    build_cython = False
    # TODO: allow a python-only installation with reduced functionality
    raise EnvironmentError(
        """
        cython could not be found.  Compilation of pyir.nufft requires Cython
        version >= 0.21.
        Install or upgrade cython via:
        pip install cython --upgrade
        """
    )

extra_compile_args = ["-ffast-math"]
# if use_OpenMP:
cmdclass = {"build_ext": extbuilder}
cmdclass.update(versioneer.get_cmdclass())

src_path = pjoin("pyir", "nufft")

# C extensions for table-based NUFFT
ext_nufft = Extension(
    "pyir.nufft._extensions._nufft_table",
    sources=[
        "pyir/nufft/_extensions/c/nufft_table.c",
        "pyir/nufft/_extensions/_nufft_table.pyx",
    ],
    depends=[
        "pyir/nufft/_extensions/c/nufft_table.template.c",
        "pyir/nufft/_extensions/c/nufft_table.template.h",
        "pyir/nufft/_extensions/c/templating.h"
        "pyir/nufft/_extensions/c/nufft_table.h",
        "pyir/nufft/_extensions/_nufft_table.pxd",
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

setup(
    name="pyir.nufft",
    author="Gregory R. Lee",
    version=versioneer.get_version(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
    namespace_package=["pyir"],
    # scripts=[],
    # since the package has c code, the egg cannot be zipped
    zip_safe=False,
    # data_files=[('pyir.nufft/data', glob('pyir.nufft/data/*.npy')), ],
    package_data={"pyir.nufft": [pjoin("tests", "data", "*")]},
)
