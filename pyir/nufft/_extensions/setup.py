#!/usr/bin/env python
from __future__ import division, print_function, absolute_import


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy as np

    config = Configuration('_extensions', parent_package, top_path)

    sources = ["c/nufft_table", ]
    source_templates = ["c/nufft_table", ]
    headers = ["c/templating", "c/nufft_table"]
    header_templates = ["c/nufft_table", ]

    config.add_extension(
        '_nufft_table',
        sources=["{0}.c".format(s) for s in ["_nufft_table"] + sources],
        depends=(["{0}.template.c".format(s) for s in source_templates]
                 + ["{0}.template.h".format(s) for s in header_templates]
                 + ["{0}.h".format(s) for s in headers]
                 + ["{0}.h".format(s) for s in sources]),
        include_dirs=["c", np.get_include()],
        extra_compile_args=['-fopenmp', '-O3', ],
        extra_link_args=['-fopenmp'],
        define_macros=[("PY_EXTENSION", None)],
    )

    config.add_extension(
        "_dtft",
        sources=['_dtft.c'],
        depends=['c/_complexstuff.h'],
        include_dirs=[np.get_include(), ],
        extra_compile_args=['-fopenmp', '-O3', ],
        extra_link_args=['-fopenmp'],
        language='c'
    )

    config.make_config_py()
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
