"""

have_pyfftw will be True if the PyFFTW module is available and the user has
not defined the environment variable NUFFT_DISABLE_PYFFTW.

have_cupy will be True if the CuPy module is available and the user has
not defined the environment variable NUFFT_DISABLE_CUPY.

cupy_has_fftn_planning will be True when the version of CuPy is new enough
that n-dimensional CUFFT planning is supported.

pyfftw_config is the pyfftw.config module when PyFFTW is available.

"""
import multiprocessing
import os
import subprocess
import sys
import warnings


__all__ = ["have_cupy", "have_pyfftw", "pyfftw_config"]


# status of optional third party libraries
global have_cupy
global have_pyfftw
global pyfftw_config
global cupy_has_fftn_planning

have_pyfftw = None
pyfftw_config = None
have_cupy = None
cupy_has_fftn_planning = False


def _check_pyfftw():
    global have_pyfftw
    global pyfftw_config

    if "NUFFT_DISABLE_PYFFTW" not in os.environ:
        try:
            import pyfftw

            have_pyfftw = True
            from pyfftw import config

            if pyfftw.__version__ < "0.11":
                warnings.warn(
                    (
                        "pyFFTW version {} found, but (>=0.11 required). "
                        "pyFFTW will not be used"
                    ).format(numba.__version__)
                )
                have_pyfftw = False

            if "PYFFTW_PLANNER_EFFORT" not in os.environ:
                # default effort for numpy interfaces and FFTW builiders
                config.PLANNER_EFFORT = "FFTW_ESTIMATE"

            if "PYFFTW_NUM_THREADS" not in os.environ:
                # This cpu_count includes the number of threads available with
                # "hyperthreading" when present.
                cpu_count = multiprocessing.cpu_count()

                # FFTW planning takes longer with more cores and there are
                # diminishing returns to using more threads than the number of
                # physical cores (e.g. hyperthreading).  Here we try to
                # determine the actual number of physical cores and use that
                # instead.
                try:
                    # try using cross-platform method from psutil
                    import psutil

                    pyfftw_threads = psutil.cpu_count(logical=False)
                except ImportError:
                    # specialized solutions with fallback to cpu_count
                    if sys.platform == "linux":
                        try:
                            hwinfo = subprocess.check_output(
                                "lscpu", shell=True
                            )
                            hwinfo = stdout.decode("ascii", "ignore").split(
                                "\n"
                            )
                            for line in hwinfo:
                                if "Core(s)" in line:
                                    cores_per_socket = int(
                                        line.split(":")[1].strip()
                                    )
                                if "Socket(s)" in line:
                                    sockets = int(line.split(":")[1].strip())
                            num_cores = sockets * cores_per_socket
                            pyfftw_threads = num_cores
                        except (subprocess.SubprocessError, NameError):
                            pyfftw_threads = max(cpu_count // 2, 1)
                    elif sys.platform == "darwin":
                        try:
                            hwinfo = subprocess.check_output(
                                "sysctl hw", shell=True
                            )
                            hwinfo = hwinfo.decode("ascii", "ignore").split(
                                "\n"
                            )[:20]
                            for line in hwinfo:
                                if "hw.physicalcpu" in line:
                                    num_cores = int(line.split(":")[1].strip())
                            pyfftw_threads = num_cores
                        except (subprocess.SubprocessError, NameError):
                            pyfftw_threads = max(cpu_count // 2, 1)
                    else:
                        pyfftw_threads = max(cpu_count // 2, 1)
                config.NUM_THREADS = pyfftw_threads
            pyfftw_config = config
        except ImportError:
            print("pyFFTW not available")
            have_pyfftw = False
            pyfftw_config = None
    else:
        have_pyfftw = False
        pyfftw_config = None


_check_pyfftw()


def _check_cupy():
    # delay check for cupy until the first time it is requested.
    # This is done to reduce import time for pyframelets.
    global have_cupy
    global cupy_has_fftn_planning

    if have_cupy is None and "NUFFT_DISABLE_CUPY" not in os.environ:
        try:
            import cupy

            have_cupy = True
            try:
                # try a basic GPU operation to test CuPy functionality
                cupy.arange(5)
            except cupy.cuda.runtime.CUDARuntimeError:
                warnings.warn(
                    "cupy imports, but does not seem functional. "
                    "Disabling CuPy-based features."
                )
                have_cupy = False
            try:
                from cupy.cuda.cufft import PlanNd

                cupy_has_fftn_planning = True
            except ImportError:
                cupy_has_fftn_planning = False

        except ImportError:
            have_cupy = False
            cupy_has_fftn_planning = False
            print("Cupy not available ")
    else:
        have_cupy = False
        cupy_has_fftn_planning = False
    return


_check_cupy()
