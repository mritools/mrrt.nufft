from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
# import os

include_dirs = [np.get_include(), ]
# library_dirs = [] #pjoin(dname, 'lib'), pjoin(openblas_dir,'lib')]

# don't have to specify -lgomp explicitly.  -fopenmp causes automatic
# linking of -lgomp

# from cyarma import include_dir as arma_inc_dir  #path to cyarma.pyx
# arma_inc_dir = '.'
# arma_include_path = '/home/lee8rx/local_libs/include'   #TODO: fix hardcode
# if not os.path.exists(arma_include_path):
#   arma_include_path = '/usr/include'

# arma_lib_path = '/home/lee8rx/local_libs'   #TODO: fix hardcode
# if not os.path.exists(arma_lib_path):
#   arma_lib_path = '/usr/include'

ext_modules = [
    # Extension("interp_table_new",['interp_table_new.pyx'],
    #           include_dirs=include_dirs,
    #           extra_compile_args=['-fopenmp','-O3',],
    #           extra_link_args=['-fopenmp'],
    #           language='c'),
    Extension("interp_table_wrap", ['interp_table_wrap.pyx'],
              include_dirs=include_dirs,
              extra_compile_args=['-fopenmp', '-O3', ],
              extra_link_args=['-fopenmp'],
              library_dirs=[
        '/home/lee8rx/src_repositories/my_git/pyrecon/PyIRT/nufft',
    ],
        libraries=["interp_table"],
        language='c'),
    Extension("cy_dtft", ['cy_dtft.pyx'],
              include_dirs=include_dirs,
              extra_compile_args=['-fopenmp', '-O3', ],
              extra_link_args=['-fopenmp'],
              library_dirs=[
        '/home/lee8rx/src_repositories/my_git/pyrecon/PyIRT/nufft',
    ],
        libraries=["interp_table"],
        language='c'),
]

# ext_modules_v2=[
#    Extension("denoising_all",['copy_block.pyx','padding.pyx','gausswin.pyx','meanvar.pyx','LUT.pyx','nystrom.pyx','nlm.pyx'],
#              include_dirs=include_dirs,
#              extra_compile_args=['-fopenmp','-O3',],
#              extra_link_args=['-fopenmp']),
# ]


setup(
    name="nufft_table",
    # package_data={'cyarma': ['*.pyx','*.pxd']},
    ext_modules=cythonize(ext_modules),
)

# Note: this simple form of setup.py relies on the comments at the top of
# the .pyx files

# gcc -pthread -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall
# -Wstrict-prototypes -fPIC -I/home/lee8rx/anaconda/include/python2.7 -c
# copy_block.c -o build/temp.linux-x86_64-2.7/copy_block.o

# gcc -pthread -shared build/temp.linux-x86_64-2.7/copy_block.o
# -L/home/lee8rx/anaconda/lib -lpython2.7 -o
# build/lib.linux-x86_64-2.7/denoise/cython/copy_block.so

# python setup.py build_ext --inplace
