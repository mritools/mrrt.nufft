import os
import sys
import shutil
from os.path import join as pjoin
from setuptools import setup, find_packages
import distutils
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import tempfile
import subprocess
import versioneer

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

# Note if rpath errors related to libgomp occur on OsX, try adding the path
# containing libgomp to the link arguments.  e.g.
# LDFLAGS="$LDFLAGS -Wl,-rpath,/Users/lee8rx/anaconda/lib" pip install -e . -v

if sys.platform == 'darwin':
    if 'ENABLE_OPENMP' in os.environ:
        use_OpenMP = True
    else:
        use_OpenMP = False
    extra_link_args = []
    # extra_link_args = ['-Wl,-rpath,{}'.format(os.path.expanduser('~/anaconda/lib'))]
else:
    use_OpenMP = True
    extra_link_args = []


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


# OpenMP check routine adapted from pynbody/yt
def check_for_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from yt, thanks to Nathan
    Goldbaum. See https://github.com/pynbody/pynbody/issues/124"""

    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    # TODO: fix proper finding of compiler
    try:
        # Get compiler invocation
        compiler = os.environ.get('CC',
                                  distutils.sysconfig.get_config_var('CC'))

        # make sure to use just the compiler name without flags
        compiler = compiler.split()[0]
    except:
        compiler = 'gcc'

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    filename = r'test.c'
    with open(filename, 'w') as f:
        f.write(
            "#include <omp.h>\n"
            "#include <stdio.h>\n"
            "int main() {\n"
            "#pragma omp parallel\n"
            "printf(\"Hello from thread %d, nthreads %d\\n\", omp_get_thread_num(), omp_get_num_threads());\n"
            "}"
        )

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call([compiler, '-fopenmp', filename],
                                        stdout=fnull, stderr=fnull)
    except OSError:
        exit_code = 1

    # Clean up
    os.chdir(curdir)
    shutil.rmtree(tmpdir)

    if exit_code == 0:
        return True
    else:
        import multiprocessing
        import platform
        cpus = multiprocessing.cpu_count()
        if cpus > 1:
            print(
    """
    WARNING:
    OpenMP support is not available in your default C compiler, even though
    your machine has more than one core available. Some routines in pyir.nufft are
    parallelized using OpenMP and these will only run on one core with your
    current configuration.
    """)
            if platform.uname()[0] == 'Darwin':
                print(
    """
    Since you are running on Mac OS, it's likely that the problem here is
    Apple's Clang, which does not support OpenMP at all. The easiest way to get
    around this is to download the latest version of gcc from here:
    http://hpc.sourceforge.net.
    After downloading, just point the CC environment variable to the real gcc
    and OpenMP support should get enabled automatically. Something like this:
    sudo tar -xzf /path/to/download.tar.gz /
    export CC='/usr/local/bin/gcc'
    python setup.py clean
    python setup.py build
    """)
            print("""Continuing your build without OpenMP...\n""")
            # time.sleep(2)
        return False

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


openmp_flags = {}
openmp_libs = {}
openmp_link = {}

if use_OpenMP and check_for_openmp():
    openmp_flags['msvc'] = ['/fopenmp', ]
    openmp_flags['mingw32'] = ['-fopenmp', ]
    openmp_flags['unix'] = ['-fopenmp', ]
    openmp_flags['cygwin'] = ['-fopenmp', ]
    openmp_libs['unix'] = ['gomp', ]         # unnecessary?
    openmp_libs['mingw32'] = ['gomp', ]      # unnecessary?
    openmp_libs['cygwin'] = ['gomp', ]       # unnecessary?
    openmp_libs['unix'] = []         # unnecessary?
    openmp_libs['mingw32'] = []      # unnecessary?
    openmp_libs['cygwin'] = []       # unnecessary?
    openmp_link['unix'] = ['-fopenmp', ]     # unnecessary?
    openmp_link['mingw32'] = ['-fopenmp', ]  # unnecessary?
    openmp_link['cygwin'] = ['-fopenmp', ]   # unnecessary?
else:
    use_OpenMP = False
    openmp_flags['msvc'] = []
    openmp_flags['mingw32'] = []
    openmp_flags['unix'] = []
    openmp_flags['cygwin'] = []
    openmp_libs['unix'] = []         # unnecessary?
    openmp_libs['mingw32'] = []      # unnecessary?
    openmp_libs['cygwin'] = []       # unnecessary?
    openmp_link['unix'] = []     # unnecessary?
    openmp_link['mingw32'] = []  # unnecessary?
    openmp_link['cygwin'] = []   # unnecessary?


class build_ext_openmp(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        print("detected compiler_type = {}".format(compiler_type))
        if use_OpenMP:
            if compiler_type in openmp_flags:
                for e in self.extensions:
                    # print("e.extra_compile_args = {}".format(
                    #     e.extra_compile_args))
                    for flag in openmp_flags[compiler_type]:
                        if flag not in e.extra_compile_args:
                            e.extra_compile_args += [flag, ]
            if compiler_type in openmp_libs:
                for e in self.extensions:
                    for ext in openmp_libs[compiler_type]:
                        if ext not in e.libraries:
                            e.libraries += [ext, ]
            if compiler_type in openmp_link:
                for e in self.extensions:
                        for link in openmp_link[compiler_type]:
                            if link not in e.extra_link_args:
                                e.extra_link_args += [link, ]
        build_ext.build_extensions(self)


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
        """)


extra_compile_args = ['-ffast-math']
# if use_OpenMP:
cmdclass = {'build_ext': build_ext_openmp}

cmdclass.update(versioneer.get_cmdclass())

src_path = pjoin('pyir', 'nufft')


# C extensions for table-based NUFFT
ext_nufft = Extension(
    'pyir.nufft._extensions._nufft_table',
    sources=['pyir/nufft/_extensions/c/nufft_table.c',
             'pyir/nufft/_extensions/_nufft_table.pyx'],
    depends=['pyir/nufft/_extensions/c/nufft_table.template.c',
             'pyir/nufft/_extensions/c/nufft_table.template.h',
             'pyir/nufft/_extensions/c/templating.h'
             'pyir/nufft/_extensions/c/nufft_table.h',
             'pyir/nufft/_extensions/_nufft_table.pxd'],
    language='c',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[numpy_include, pjoin(src_path, '_extensions', 'c')])


# C extensions for brute-force discrete-time Fourier transform
ext_dtft = Extension(
    'pyir.nufft._extensions._dtft',
    sources=['pyir/nufft/_extensions/_dtft.pyx'],
    depends=['pyir/nufft/_extensions/c/_complexstuff.h'],
    language='c',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[numpy_include, pjoin(src_path, '_extensions', 'c')])

ext_modules = [ext_nufft, ext_dtft]

c_macros = [("PY_EXTENSION", None)]
cython_macros = []
cythonize_opts = {}
if os.environ.get("CYTHON_TRACE"):
    cythonize_opts['linetrace'] = True
    cython_macros.append(("CYTHON_TRACE_NOGIL", 1))

if USE_CYTHON:
    ext_modules = cythonize(ext_modules, compiler_directives=cythonize_opts)


setup(name='pyir.nufft',
      author='Gregory R. Lee',
      version=versioneer.get_version(),
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      packages=find_packages(),
      namespace_package=['pyir'],
      # scripts=[],
      # since the package has c code, the egg cannot be zipped
      zip_safe=False,
      # data_files=[('pyir.nufft/data', glob('pyir.nufft/data/*.npy')), ],
      package_data={'pyir.nufft':
                    [pjoin('tests', 'data', '*'), ]},
      )
