"""Functions for runtime generation of CUDA-based NUFFT kernels."""
import os

import numpy as np
from jinja2 import Template

from mrrt.utils import config

template_path = os.path.join(os.path.dirname(__file__), "jinja")


def _template_from_filename(filename):
    with open(filename, "r") as f:
        fstr = f.read()
    return Template(fstr)


grid_includes_template = _template_from_filename(
    os.path.join(template_path, "gridding_kernel_includes.jinja")
)

gridding_kernel_templates = {}
for ndim in [1, 2, 3]:
    gridding_kernel_templates[(ndim, "forward")] = _template_from_filename(
        os.path.join(template_path, "table_{}d_forward.jinja".format(ndim))
    )
    gridding_kernel_templates[(ndim, "adjoint")] = _template_from_filename(
        os.path.join(template_path, "table_{}d_adjoint.jinja".format(ndim))
    )


def apply_astyle(code):
    """Auto-indent generated code by calling astyle from the shell.

    This function requires that the astyle binary is on the path.

    Parameters
    ----------
    code : str
        A string representing the code to be formatted.

    Returns
    -------
    code : str
        The reformatted code
    """
    import subprocess
    import tempfile

    # create a temporary file
    fid, fname = tempfile.mkstemp(".cu")

    try:
        # write to temporary file
        with open(fid, "wt") as f:
            f.write(code)

        # call astyle to reformat the file in-place
        cmd = "astyle --suffix=none --delete-empty-lines {}".format(fname)
        subprocess.check_output(cmd, shell=True)

        # read the reformatted file
        code = open(fname, "rt").read()
    finally:
        # delete the temporary file
        os.remove(fname)
    return code


def _get_gridding_funcs(
    Kd,
    M,
    J,
    L,
    precision="single",
    order=1,
    is_complex_kernel=False,
    compile_options=("--use_fast_math",),
    render_template_only=False,
):
    """Compile a CUDA GPU kernel for the NUFFT.

    Paramters
    ---------
    Kd : tuple of int
        The oversampled image size. len(Kd) must be 1, 2 or 3.
    M : int
        The total number of k-space samples.
    J : int
        The extent of the NUFFT kernel in number of samples.
    L : int
        The size of the lookup table (the actual number of precomputed values
        int the table is J*L.
    precision : {'single', 'double'}, optional
        Floating point precision to use during the computations.
    order : {0, 1}, optional
        The order of interpolation to use for the lookup table. 0 means nearest
        neighbord and 1 means linear interpolation.
    is_complex_kernel : bool, optional
        If true, the NUFFT kernel has complex-valued elements. This can be used
        to build a spatial shift into the kernel itself.
    compile_options : tuple of str, optional
        Any additional options to pass to the NVCC compiler.
    render_template_only : bool, optional
        If True, a string containing the CUDA code is returned instead of a
        compiled CuPy RawKernel object.

    Returns
    -------
    forward_kernel, adjoint_kernel : cupy.core.RawKernel or str
        cupy.core.RawKernel objects corresponding to the forward and adjoint
        NUFFT. If ``render_template_only`` is ``True``, then a string
        containing the CUDA C++ code is returned instead.
    """
    if config.have_cupy:
        import cupy
    else:
        raise ImportError("This function requries CuPy")

    precision = precision.lower()
    if precision == "single":
        real_type = "float"
    elif precision == "double":
        real_type = "double"
    else:
        raise ValueError("precision must be single or double")

    # cuda_version = cupy.cuda.driver.get_build_version()
    compile_options = tuple(compile_options)

    """ compile with many things hardcoded to reduce the number of
    registers needed."""
    ndim = len(Kd)
    if ndim < 3:
        Kd = Kd + ((1,) * (3 - ndim))

    ncenter = int(np.floor(J * L / 2))

    template_kwargs = dict(
        grid_includes_template=grid_includes_template,
        real_type=real_type,
        complex_kernel=is_complex_kernel,
        order=order,
        J=J,
        L=L,
        ncenter=ncenter,
        J_2="%.1f" % (J / 2),
        M=M,
        M2=2 * M,
        K1="%d" % Kd[0],
        K2="%d" % Kd[1],
        K3="%d" % Kd[2],
    )

    template_forward = gridding_kernel_templates[(ndim, "forward")]
    template_adjoint = gridding_kernel_templates[(ndim, "adjoint")]
    rendered_template_forward = template_forward.render(**template_kwargs)
    rendered_template_adjoint = template_adjoint.render(**template_kwargs)

    if render_template_only:
        # return strings containing the kernel source
        return rendered_template_forward, rendered_template_adjoint

    if is_complex_kernel:
        cplx_str = "complex"
    else:
        cplx_str = "real"

    forward_name = "interp{}_table{}_{}_{}_per_GPUkernel".format(
        ndim, order, cplx_str, real_type
    )
    adjoint_name = "interp{}_table{}_{}_{}_per_adj_GPUkernel".format(
        ndim, order, cplx_str, real_type
    )

    forward_kernel = cupy.core.RawKernel(
        rendered_template_forward, name=forward_name, options=compile_options
    )

    adjoint_kernel = cupy.core.RawKernel(
        rendered_template_adjoint, name=adjoint_name, options=compile_options
    )

    return forward_kernel, adjoint_kernel
