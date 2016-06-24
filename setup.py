#!/usr/bin/env python
import os
import sys
import subprocess
import versioneer


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


if sys.platform == "darwin":
    # Don't create resource files on OS X tar.
    os.environ["COPY_EXTENDED_ATTRIBUTES_DISABLE"] = "true"
    os.environ["COPYFILE_DISABLE"] = "true"


setup_args = {}

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

if sys.platform == "darwin":
    # Don't create resource files on OS X tar.
    os.environ["COPY_EXTENDED_ATTRIBUTES_DISABLE"] = "true"
    os.environ["COPYFILE_DISABLE"] = "true"

setup_args = {}


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'util', 'cythonize.py'),
                         os.path.join('pyir', 'nufft')],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # main modules
    config.add_subpackage('pyir.nufft')
    config.add_subpackage('pyir.nufft.tests')
    config.add_data_dir('pyir/nufft/tests/data')
    config.add_data_dir('pyir/nufft/tests/data/mat_files')
    config.get_version('pyir.nufft/version.py')

    return config


def setup_package():
    metadata = dict(
        name="pyir.nufft",
        maintainer="Gregory R. Lee",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        url='https://bitbucket.org/grlee77/pyir.nufft',
        # maintainer_email="",
        # download_url="url='https://bitbucket.org/grlee77/pyir.nufft/releases",
        include_package_data=True,
        license="TODO",
        description="High Performance NUFFT in Python",
        long_description="""\
        High Performance NUFFT in Python
        """,
        keywords=["MRI", "magnetic resonance imaging", "image reconstruction",
                  "nufft", "nfft", "nonuniform fft"],
        classifiers=[
            # TODO: "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            # TODO: "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Topic :: Software Development :: Libraries :: Python Modules"
        ],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        test_suite='nose.collector',
        **setup_args
    )
    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                            'clean')):
        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install PyWavelets when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

    else:
        if (len(sys.argv) >= 2 and sys.argv[1] == 'bdist_wheel') or (
                'develop' in sys.argv):
            # bdist_wheel needs setuptools
            import setuptools
        from numpy.distutils.core import setup

    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Generate Cython sources, unless building from source release
        generate_cython()

    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
