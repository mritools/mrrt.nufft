# -*- coding: utf-8 -*-
import os
import platform
import ctypes
import inspect
import numpy as np
#from scipy import weave
import PyIRT.nufft


def import_ctypes_lib():
    """import interp_table library via ctypes
    """

    sys_name = platform.system()

    if sys_name.lower() == 'windows':
        interplib = ctypes.cdll['interp_table.dll']
    # linux of Mac
    elif ((sys_name.lower() == 'linux') or (sys_name.lower() == 'darwin')):
        interp_lib_fname = os.path.abspath(
            os.path.join(
                os.path.dirname(
                    inspect.getsourcefile(
                        PyIRT.nufft)),
                'interp_table_lib.so'))
        # interp_lib_fname='interp_table_lib.so'
        #interplib = ctypes.cdll.LoadLibrary(interp_lib_fname)
        interplib = ctypes.cdll.LoadLibrary(interp_lib_fname)
    else:
        raise Exception("Unsupported system type")
    return interplib


def interp1_table_adj(fm, h1, J1, L1, tm, K1, order=None, flips=None):

    M = fm.shape[0]
    N = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, None]

    if (h1.shape[0] != J1 * L1 + 1) | (h1.shape[1] != 1):
        print(("J = %d, L=%d, tablelength=%d" % J1, L1, h1.shape[0]))
        raise ValueError("h size problem")

    if tm.ndim == 1:
        tm = tm[:, None]

    if not tm.shape == (M, 1):  # (M != tm.shape[0]) | (1 != tm.shape[1]):
        raise ValueError("tm must be Mx1 col vector")

    J1 = int(J1)
    L1 = int(L1)
    ck = _interp1_table_adj(fm, K1, h1, J1, L1, tm, M, N, order, flips)
    return ck


# 'weave'): #'python1'
def _interp1_table_adj(
        fm, K1, h1, J1, L1, tm, M, N, order, flips, alg='ctypes'):
    #ncenter1 = math.floor(Jd[0]*Ld[0]/2)
    #ncenter2 = math.floor(Jd[1]*Ld[1]/2)

    fm = np.asfortranarray(fm)  # from Matrix to array
    if fm.ndim == 1:
        fm = fm[..., np.newaxis]

    kernel_dtype = h1.dtype
    ck = np.asfortranarray(
        np.zeros(
            (K1, N), dtype=np.result_type(
                kernel_dtype, np.complex64)))
    complex_kernel = (
        kernel_dtype == np.complex64) or (
        kernel_dtype == np.complex128)
    tm = tm.astype(h1.real.dtype)
    if alg == 'ctypes':
        interplib = import_ctypes_lib()  # ctypes.cdll['interp_table.dll']

        array1d_float_c = np.ctypeslib.ndpointer(
            dtype=h1.real.dtype,
            flags='F_CONTIGUOUS')
        #array2d_float_c = np.ctypeslib.ndpointer(dtype=float64, ndim=2, flags='C_CONTIGUOUS')

        if order == 0:
            if kernel_dtype == np.complex64:
                interp_func = interplib.interp1f_table0_complex_per_adj
            elif kernel_dtype == np.complex128:
                interp_func = interplib.interp1_table0_complex_per_adj
            elif kernel_dtype == np.float64:
                interp_func = interplib.interp1_table0_real_per_adj
            elif kernel_dtype == np.float32:
                interp_func = interplib.interp1f_table0_real_per_adj
            else:
                raise ValueError("ck has unsupported dtype")
        else:
            if kernel_dtype == np.complex64:
                interp_func = interplib.interp1f_table1_complex_per_adj
            elif kernel_dtype == np.complex128:
                interp_func = interplib.interp1_table1_complex_per_adj
            elif kernel_dtype == np.float64:
                interp_func = interplib.interp1_table1_real_per_adj
            elif kernel_dtype == np.float32:
                interp_func = interplib.interp1f_table1_real_per_adj
            else:
                raise ValueError("ck has unsupported dtype")

        if complex_kernel:
            interp_func.argtypes = [array1d_float_c, array1d_float_c,
                                    ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    array1d_float_c,
                                    ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int]
        else:
            interp_func.argtypes = [array1d_float_c, array1d_float_c,
                                    ctypes.c_int,
                                    array1d_float_c,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    array1d_float_c,
                                    ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int]

        K1 = np.int32(K1)
        J1 = np.int32(J1)
        L1 = np.int32(L1)
        N = np.int32(N)
        tm = np.asfortranarray(tm)
        #r_fm = np.asfortranarray(np.zeros((M,N)))
        #i_fm = np.asfortranarray(np.zeros((M,N)))
        r_fm = np.asfortranarray(fm.real)
        i_fm = np.asfortranarray(fm.imag)

        r_ck = np.asfortranarray(np.zeros_like(ck.real))
        i_ck = np.asfortranarray(np.zeros_like(ck.real))
        r_h1 = np.asfortranarray(h1.real)

        if complex_kernel:
            i_h1 = np.asfortranarray(h1.imag)
        #array2d_float_c = np.ctypeslib.ndpointer(dtype=float64, ndim=2, flags='C_CONTIGUOUS')

        # for nn in range(0,N): #never tested for N>1!
        if (alg == 'ctypes'):

            if complex_kernel:
                interp_func(
                    r_ck,
                    i_ck,
                    K1,
                    r_h1,
                    i_h1,
                    J1,
                    L1,
                    tm,
                    M,
                    r_fm,
                    i_fm,
                    N)
            else:
                interp_func(r_ck, i_ck, K1, r_h1, J1, L1, tm, M, r_fm, i_fm, N)

            ck.real = np.asfortranarray(r_ck)
            ck.imag = np.asfortranarray(i_ck)

        else:
            raise ValueError("invalid algorithm specified")

        return ck


# [False, False]):
def interp2_table_adj(fm, h1, h2, Jd, Ld, tm, Kd, order=None, flips=None):

    M = fm.shape[0]
    N = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, None]
    if h2.ndim == 1:
        h2 = h2[:, None]

    Jd = np.asanyarray(Jd).astype(np.int32)
    Ld = np.asanyarray(Ld).astype(np.int32)
    flips = np.asanyarray(flips)
    if flips.any():
        if not (flips.size is 2):
            raise ValueError("flips must be length 2 array")
        flips = flips.astype(np.int32)

    if not ((len(Jd) == 2) & (len(Ld) == 2) & (len(Kd) == 2)):
        raise ValueError("Error:  J, K and L must all be length 2")

    if (h1.shape[0] != Jd[0] * Ld[0] + 1) | (h1.shape[1] != 1):
        print(("J = %d, L=%d, tablelength=%d" % Jd[0], Ld[0], h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != Jd[1] * Ld[1] + 1) | (h2.shape[1] != 1):
        print(("J = %d, L=%d, tablelength=%d" % Jd[1], Ld[1], h2.shape[0]))
        raise ValueError("h2 size problem")

    if not tm.shape == (M, 2):  # (M != tm.shape[0]) | (2 != tm.shape[1]):
        raise ValueError("tm must be Mx2")

    ck = _interp2_table_adj(fm, Kd, h1, h2, Jd, Ld, tm, M, N, order, flips)
    return ck


# 'weave'): #'python1'
def _interp2_table_adj(
        fm, Kd, h1, h2, Jd, Ld, tm, M, N, order, flips, alg='ctypes'):
    #ncenter1 = math.floor(Jd[0]*Ld[0]/2)
    #ncenter2 = math.floor(Jd[1]*Ld[1]/2)

    fm = np.asarray(fm)  # from Matrix to array
    if fm.ndim == 1:
        fm = fm[..., np.newaxis]

    kernel_dtype = h1.dtype
    ck = np.asfortranarray(
        np.zeros(
            (np.prod(Kd), N), dtype=np.result_type(
                kernel_dtype, np.complex64)))
    complex_kernel = (
        kernel_dtype == np.complex64) or (
        kernel_dtype == np.complex128)
    tm = tm.astype(h1.real.dtype)
    if alg == 'ctypes':
        interplib = import_ctypes_lib()  # ctypes.cdll['interp_table.dll']

        array1d_float_c = np.ctypeslib.ndpointer(
            dtype=h1.real.dtype,
            flags='F_CONTIGUOUS')
        #array2d_float_c = np.ctypeslib.ndpointer(dtype=float64, ndim=2, flags='C_CONTIGUOUS')

        if order == 0:
            if kernel_dtype == np.complex64:
                interp_func = interplib.interp2f_table0_complex_per_adj
            elif kernel_dtype == np.complex128:
                interp_func = interplib.interp2_table0_complex_per_adj
            elif kernel_dtype == np.float64:
                interp_func = interplib.interp2_table0_real_per_adj
            elif kernel_dtype == np.float32:
                interp_func = interplib.interp2f_table0_real_per_adj
            else:
                raise ValueError("ck has unsupported dtype")
        else:
            if kernel_dtype == np.complex64:
                interp_func = interplib.interp2f_table1_complex_per_adj
            elif kernel_dtype == np.complex128:
                interp_func = interplib.interp2_table1_complex_per_adj
            elif kernel_dtype == np.float64:
                interp_func = interplib.interp2_table1_real_per_adj
            elif kernel_dtype == np.float32:
                interp_func = interplib.interp2f_table1_real_per_adj
            else:
                raise ValueError("ck has unsupported dtype")

        if complex_kernel:
            interp_func.argtypes = [array1d_float_c, array1d_float_c,
                                    ctypes.c_int, ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int,
                                    array1d_float_c,
                                    ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int]
        else:
            interp_func.argtypes = [array1d_float_c, array1d_float_c,
                                    ctypes.c_int, ctypes.c_int,
                                    array1d_float_c,
                                    array1d_float_c,
                                    ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int,
                                    array1d_float_c,
                                    ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int]

        J1 = np.int32(Jd[0])
        J2 = np.int32(Jd[1])
        K1 = np.int32(Kd[0])
        K2 = np.int32(Kd[1])
        L1 = np.int32(Ld[0])
        L2 = np.int32(Ld[1])
        N = np.int32(N)
        tm = np.asfortranarray(tm)
        #r_fm = np.asfortranarray(np.zeros((M,N)))
        #i_fm = np.asfortranarray(np.zeros((M,N)))
        r_fm = np.asfortranarray(fm.real)
        i_fm = np.asfortranarray(fm.imag)

        r_ck = np.asfortranarray(np.zeros_like(ck.real))
        i_ck = np.asfortranarray(np.zeros_like(ck.real))
        r_h1 = np.asfortranarray(h1.real)
        r_h2 = np.asfortranarray(h2.real)

        if complex_kernel:
            i_h1 = np.asfortranarray(h1.imag)
            i_h2 = np.asfortranarray(h2.imag)
        #array2d_float_c = np.ctypeslib.ndpointer(dtype=float64, ndim=2, flags='C_CONTIGUOUS')

        # for nn in range(0,N): #never tested for N>1!
        if (alg == 'ctypes'):

            if complex_kernel:
                interp_func(
                    r_ck,
                    i_ck,
                    K1,
                    K2,
                    r_h1,
                    i_h1,
                    r_h2,
                    i_h2,
                    J1,
                    J2,
                    L1,
                    L2,
                    tm,
                    M,
                    r_fm,
                    i_fm,
                    N)
            else:
                interp_func(
                    r_ck,
                    i_ck,
                    K1,
                    K2,
                    r_h1,
                    r_h2,
                    J1,
                    J2,
                    L1,
                    L2,
                    tm,
                    M,
                    r_fm,
                    i_fm,
                    N)
            # print "fm.shape = "
            # print fm.shape
            # print r_fm.shape
            ck.real = np.asfortranarray(r_ck)
            ck.imag = np.asfortranarray(i_ck)

        else:
            raise ValueError("invalid algorithm specified")

        return ck


# [False, False]):
def interp3_table_adj(fm, h1, h2, h3, Jd, Ld, tm, Kd, order=None, flips=None):

    M = fm.shape[0]
    N = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, None]
    if h2.ndim == 1:
        h2 = h2[:, None]
    if h3.ndim == 1:
        h3 = h3[:, None]

    Jd = np.asanyarray(Jd).astype(np.int32)
    Ld = np.asanyarray(Ld).astype(np.int32)
    flips = np.asanyarray(flips)
    if flips.any():
        if not (flips.size is 2):
            raise ValueError("flips must be length 2 array")
        flips = flips.astype(np.int32)

    if not ((len(Jd) == 3) & (len(Ld) == 3) & (len(Kd) == 3)):
        raise ValueError("Error:  J, K and L must all be length 3")

    if (h1.shape[0] != Jd[0] * Ld[0] + 1) | (h1.shape[1] != 1):
        print(("J = %d, L=%d, tablelength=%d" % Jd[0], Ld[0], h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != Jd[1] * Ld[1] + 1) | (h2.shape[1] != 1):
        print(("J = %d, L=%d, tablelength=%d" % Jd[1], Ld[1], h2.shape[0]))
        raise ValueError("h2 size problem")

    if (h3.shape[0] != Jd[2] * Ld[2] + 1) | (h3.shape[1] != 1):
        print(("J = %d, L=%d, tablelength=%d" % Jd[2], Ld[2], h3.shape[0]))
        raise ValueError("h3 size problem")

    if not tm.shape == (M, 3):  # (M != tm.shape[0]) | (2 != tm.shape[1]):
        raise ValueError("tm must be Mx2")

    ck = _interp3_table_adj(fm, Kd, h1, h2, h3, Jd, Ld, tm, M, N, order, flips)
    return ck


# 'weave'): #'python1'
def _interp3_table_adj(
        fm, Kd, h1, h2, h3, Jd, Ld, tm, M, N, order, flips, alg='ctypes'):
    #ncenter1 = math.floor(Jd[0]*Ld[0]/2)
    #ncenter2 = math.floor(Jd[1]*Ld[1]/2)

    fm = np.asarray(fm)  # from Matrix to array
    if fm.ndim == 1:
        fm = fm[..., np.newaxis]

    kernel_dtype = h1.dtype
    ck = np.asfortranarray(
        np.zeros(
            (np.prod(Kd), N), dtype=np.result_type(
                kernel_dtype, np.complex64)))
    complex_kernel = (
        kernel_dtype == np.complex64) or (
        kernel_dtype == np.complex128)
    tm = tm.astype(h1.real.dtype)
    if alg == 'ctypes':
        interplib = import_ctypes_lib()  # ctypes.cdll['interp_table.dll']

        array1d_float_c = np.ctypeslib.ndpointer(
            dtype=h1.real.dtype,
            flags='F_CONTIGUOUS')
        #array2d_float_c = np.ctypeslib.ndpointer(dtype=float64, ndim=2, flags='C_CONTIGUOUS')

        if order == 0:
            if kernel_dtype == np.complex64:
                interp_func = interplib.interp3f_table0_complex_per_adj
            elif kernel_dtype == np.complex128:
                interp_func = interplib.interp3_table0_complex_per_adj
            elif kernel_dtype == np.float64:
                interp_func = interplib.interp3_table0_real_per_adj
            elif kernel_dtype == np.float32:
                interp_func = interplib.interp3f_table0_real_per_adj
            else:
                raise ValueError("ck has unsupported dtype")
        else:
            if kernel_dtype == np.complex64:
                interp_func = interplib.interp3f_table1_complex_per_adj
            elif kernel_dtype == np.complex128:
                interp_func = interplib.interp3_table1_complex_per_adj
            elif kernel_dtype == np.float64:
                interp_func = interplib.interp3_table1_real_per_adj
            elif kernel_dtype == np.float32:
                interp_func = interplib.interp3f_table1_real_per_adj
            else:
                raise ValueError("ck has unsupported dtype")

        if complex_kernel:
            interp_func.argtypes = [array1d_float_c, array1d_float_c,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    array1d_float_c, array1d_float_c,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    array1d_float_c,
                                    ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int]
        else:
            interp_func.argtypes = [array1d_float_c, array1d_float_c,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    array1d_float_c,
                                    array1d_float_c,
                                    array1d_float_c,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    array1d_float_c,
                                    ctypes.c_int,
                                    array1d_float_c, array1d_float_c,
                                    ctypes.c_int]

        J1 = np.int32(Jd[0])
        J2 = np.int32(Jd[1])
        J3 = np.int32(Jd[2])
        K1 = np.int32(Kd[0])
        K2 = np.int32(Kd[1])
        K3 = np.int32(Kd[2])
        L1 = np.int32(Ld[0])
        L2 = np.int32(Ld[1])
        L3 = np.int32(Ld[2])
        N = np.int32(N)
        tm = np.asfortranarray(tm)
        #r_fm = np.asfortranarray(np.zeros((M,N)))
        #i_fm = np.asfortranarray(np.zeros((M,N)))
        r_fm = np.asfortranarray(fm.real)
        i_fm = np.asfortranarray(fm.imag)

        r_ck = np.asfortranarray(np.zeros_like(ck.real))
        i_ck = np.asfortranarray(np.zeros_like(ck.imag))
        r_h1 = np.asfortranarray(h1.real)
        r_h2 = np.asfortranarray(h2.real)
        r_h3 = np.asfortranarray(h3.real)

        if complex_kernel:
            i_h1 = np.asfortranarray(h1.imag)
            i_h2 = np.asfortranarray(h2.imag)
            i_h3 = np.asfortranarray(h3.imag)
        #array2d_float_c = np.ctypeslib.ndpointer(dtype=float64, ndim=2, flags='C_CONTIGUOUS')

        # for nn in range(0,N): #never tested for N>1!
        if (alg == 'ctypes'):
            if complex_kernel:
                interp_func(
                    r_ck,
                    i_ck,
                    K1,
                    K2,
                    K3,
                    r_h1,
                    i_h1,
                    r_h2,
                    i_h2,
                    r_h3,
                    i_h3,
                    J1,
                    J2,
                    J3,
                    L1,
                    L2,
                    L3,
                    tm,
                    M,
                    r_fm,
                    i_fm,
                    N)
            else:
                interp_func(
                    r_ck,
                    i_ck,
                    K1,
                    K2,
                    K3,
                    r_h1,
                    r_h2,
                    r_h3,
                    J1,
                    J2,
                    J3,
                    L1,
                    L2,
                    L3,
                    tm,
                    M,
                    r_fm,
                    i_fm,
                    N)
            ck.real = np.asfortranarray(r_ck)
            ck.imag = np.asfortranarray(i_ck)
        else:
            raise ValueError("invalid algorithm specified")

        return ck
