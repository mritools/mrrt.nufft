"""Distutils / setuptools helpers.

The functions in this file were borrowed from DIPY (see LICENSE)
"""
import os
from os.path import join as pjoin, dirname, exists
import tempfile
import shutil

from distutils.errors import CompileError, LinkError

from distutils import log


# Path of file to which to write C conditional vars from build-time checks
CONFIG_H = pjoin("build", "config.h")
# File name (no directory) to which to write Python vars from build-time checks
CONFIG_PY = "__config__.py"
# Directory to which to write libraries for building
LIB_DIR_TMP = pjoin("build", "extra_libs")


def add_flag_checking(build_ext_class, flag_defines, top_package_dir=""):
    """ Override input `build_ext_class` to check compiler `flag_defines`

    Parameters
    ----------
    build_ext_class : class
        Class implementing ``distutils.command.build_ext.build_ext`` interface,
        with a ``build_extensions`` method.
    flag_defines : sequence
        A sequence of elements, where the elements are sequences of length 4
        consisting of (``compile_flags``, ``link_flags``, ``code``,
        ``defvar``). ``compile_flags`` is a sequence of compiler flags;
        ``link_flags`` is a sequence of linker flags. We
        check ``compile_flags`` to see whether a C source string ``code`` will
        compile, and ``link_flags`` to see whether the resulting object file
        will link.  If both compile and link works, we add ``compile_flags`` to
        ``extra_compile_args`` and ``link_flags`` to ``extra_link_args`` of
        each extension when we build the extensions.  If ``defvar`` is not
        None, it is the name of C variable to be defined in ``build/config.h``
        with 1 if the combination of (``compile_flags``, ``link_flags``,
        ``code``) will compile and link, 0 otherwise. If None, do not write
        variable.
    top_package_dir : str
        String giving name of top-level package, for writing Python file
        containing configuration variables.  If empty, do not write this file.
        Variables written are the same as the Cython variables generated via
        the `flag_defines` setting.

    Returns
    -------
    checker_class : class
        A class with similar interface to
        ``distutils.command.build_ext.build_ext``, that adds all working
        ``compile_flags`` values to the ``extra_compile_args`` and working
        ``link_flags`` to ``extra_link_args`` attributes of extensions, before
        compiling.
    """

    class Checker(build_ext_class):
        flag_defs = tuple(flag_defines)

        def can_compile_link(self, compile_flags, link_flags, code):
            cc = self.compiler
            fname = "test.c"
            cwd = os.getcwd()
            tmpdir = tempfile.mkdtemp()
            try:
                os.chdir(tmpdir)
                with open(fname, "wt") as fobj:
                    fobj.write(code)
                try:
                    objects = cc.compile([fname], extra_postargs=compile_flags)
                except CompileError:
                    return False
                try:
                    # Link shared lib rather then executable to avoid
                    # http://bugs.python.org/issue4431 with MSVC 10+
                    cc.link_shared_lib(
                        objects, "testlib", extra_postargs=link_flags
                    )
                except (LinkError, TypeError):
                    return False
            finally:
                os.chdir(cwd)
                shutil.rmtree(tmpdir)
            return True

        def build_extensions(self):
            """ Hook into extension building to check compiler flags """
            def_vars = []
            good_compile_flags = []
            good_link_flags = []
            config_dir = dirname(CONFIG_H)
            for compile_flags, link_flags, code, def_var in self.flag_defs:
                compile_flags = list(compile_flags)
                link_flags = list(link_flags)
                flags_good = self.can_compile_link(
                    compile_flags, link_flags, code
                )
                if def_var:
                    def_vars.append((def_var, flags_good))
                if flags_good:
                    good_compile_flags += compile_flags
                    good_link_flags += link_flags
                else:
                    log.warn(
                        "Flags {0} omitted because of compile or link "
                        "error".format(compile_flags + link_flags)
                    )
            if def_vars:  # write config.h file
                if not exists(config_dir):
                    self.mkpath(config_dir)
                with open(CONFIG_H, "wt") as fobj:
                    fobj.write("/* Automatically generated; do not edit\n")
                    fobj.write("   C defines from build-time checks */\n")
                    for v_name, v_value in def_vars:
                        fobj.write(
                            "int {0} = {1};\n".format(
                                v_name, 1 if v_value else 0
                            )
                        )
            if def_vars and top_package_dir:  # write __config__.py file
                config_py_dir = (
                    top_package_dir
                    if self.inplace
                    else pjoin(self.build_lib, top_package_dir)
                )
                if not exists(config_py_dir):
                    self.mkpath(config_py_dir)
                config_py = pjoin(config_py_dir, CONFIG_PY)
                with open(config_py, "wt") as fobj:
                    fobj.write("# Automatically generated; do not edit\n")
                    fobj.write("# Variables from compile checks\n")
                    for v_name, v_value in def_vars:
                        fobj.write("{0} = {1}\n".format(v_name, v_value))
            if def_vars or good_compile_flags or good_link_flags:
                for ext in self.extensions:
                    ext.extra_compile_args += good_compile_flags
                    ext.extra_link_args += good_link_flags
                    if def_vars:
                        ext.include_dirs.append(config_dir)
            build_ext_class.build_extensions(self)

    return Checker


def make_np_ext_builder(build_ext_class):
    """ Override input `build_ext_class` to add numpy includes to extension

    This is useful to delay call of ``np.get_include`` until the extension is
    being built.

    Parameters
    ----------
    build_ext_class : class
        Class implementing ``distutils.command.build_ext.build_ext`` interface,
        with a ``build_extensions`` method.

    Returns
    -------
    np_build_ext_class : class
        A class with similar interface to
        ``distutils.command.build_ext.build_ext``, that adds libraries in
        ``np.get_include()`` to include directories of extension.
    """

    class NpExtBuilder(build_ext_class):
        def build_extensions(self):
            """ Hook into extension building to add np include dirs
            """
            # Delay numpy import until last moment
            import numpy as np

            for ext in self.extensions:
                ext.include_dirs.append(np.get_include())
            build_ext_class.build_extensions(self)

    return NpExtBuilder
