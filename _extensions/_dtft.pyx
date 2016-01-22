# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:59:29 2014

@author: lee8rx
"""

from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange

from _complexstuff cimport zexp, double_complex

from libc.math cimport cos, sin

#ctypedef fused data_t:
#    cnp.float32_t
#    cnp.float64_t
#    cnp.complex64_t
#    cnp.complex128_t

#cython:  Constructing complex number not allowed without gil, so use _dtft_adj_separate_real_complex instead for parallel processing of repetitions
#TODO: dtft
    
def dtft_adj(X, omega, Nd, n_shift=None, separate_real_complex=True):
    
    omega = np.asfortranarray(omega.astype(np.float64))
    
    Nd = np.asfortranarray(Nd)
    if n_shift is None:
        n_shift = np.zeros(Nd,dtype=np.intp)
    else:
        n_shift = np.asfortranarray(n_shift)
    
    dd = len(Nd)
    if omega.shape[1] != dd:
        raise ValueError("number of columns in omega doesn't match length of Nd")
    if dd != len(n_shift):
        raise ValueError("argument size mismatch")
    if dd > 3:
        raise ValueError("Nd cannot exceed 3D")
    if dd < 3:
        Nd = np.concatenate((Nd,[1,]*(3-dd)))
    
    if X.ndim==1:
        X = X[:,None]
    
    repetitions = X.shape[1]
                     
    if not separate_real_complex:
        X = np.asfortranarray(X.astype(np.complex128))
        x_out = np.zeros(tuple(Nd) + (repetitions,), 
                         dtype = X.dtype).ravel(order='F')
    
        _dtft_adj(X, omega, Nd, n_shift, repetitions, x_out)
    else:
        Xr = np.asfortranarray(X.real.astype(np.float64))
        Xi = np.asfortranarray(X.imag.astype(np.float64))
        xr_out = np.zeros(tuple(Nd) + (repetitions,), 
                     dtype = np.float64).ravel(order='F')
        xi_out = np.zeros(tuple(Nd) + (repetitions,), 
                     dtype = np.float64).ravel(order='F')

        _dtft_adj_separate_real_complex(Xr, Xi, omega, Nd, n_shift, repetitions, xr_out, xi_out)
        x_out = xr_out + 1j * xi_out
        
    return x_out.reshape(tuple(Nd) + (repetitions,),order='F')


@cython.wraparound(False)
@cython.boundscheck(False)
cdef _dtft_adj(double_complex [::1,:] X, double [::1,:] omega, 
               Py_ssize_t [::1] Nd, Py_ssize_t [::1] n_shift, 
               Py_ssize_t repetitions, double_complex [::1] x_out):
    """  Compute adjoint of d-dim DTFT for spectrum X at frequency locations omega
    
     In
    	X	[M,L]		dD DTFT values
    	omega	[M,d]		frequency locations (radians)
    	n_shift [d,1]		use [0:N-1]-n_shift (default [0 ... 0])
     repetitions : int
         repeat same omega for all repetitions in X
     Out
    	x	[(Nd),L]	signal values
    
     Requires enough memory to store M * (*Nd) size matrices. (For testing only)
    
    Matlab version: Copyright 2003-4-13, Jeff Fessler, The University of Michigan
    """

    cdef:
        Py_ssize_t i, j, k, l, m, M, idx
        double t
    M = omega.shape[0]
    for k in range(-n_shift[2], Nd[2] - n_shift[2]):
        for j in range(-n_shift[1], Nd[1] - n_shift[1]):
            for i in range(-n_shift[0], Nd[0]  - n_shift[0]):
                for l in range(repetitions):
                    idx = i+Nd[0]*(j+Nd[1]*(k + l*Nd[2]))
                    #with nogil, parallel():
                    for m in range(M):
                        Xm = X[m, l]
                        t = omega[m, 0]*i + omega[m, 1]*j + omega[m, 2]*k
                        x_out[idx] = x_out[idx] + zexp(1j*t) * Xm
    return 0


@cython.wraparound(False)
@cython.boundscheck(False)
cdef _dtft_adj_separate_real_complex(double [::1,:] Xr, double [::1,:] Xi, double [::1,:] omega, 
               Py_ssize_t [::1] Nd, Py_ssize_t [::1] n_shift, 
               Py_ssize_t repetitions, double [::1] xr_out, double [::1] xi_out):
    """  Compute adjoint of d-dim DTFT for spectrum X at frequency locations omega
    
     In
    	X	[M,L]		dD DTFT values
    	omega	[M,d]		frequency locations (radians)
    	n_shift [d,1]		use [0:N-1]-n_shift (default [0 ... 0])
     repetitions : int
         repeat same omega for all repetitions in X
     Out
    	x	[(Nd),L]	signal values
    
     Requires enough memory to store M * (*Nd) size matrices. (For testing only)
    
    Matlab version: Copyright 2003-4-13, Jeff Fessler, The University of Michigan
    """

    cdef:
        Py_ssize_t i, j, k, l, m, M, N, idx
        double kshift, jshift, ishift
        double t, st, ct, xi, xr
    M = omega.shape[0]
    N = Nd[0]*Nd[1]*Nd[2]
    
    with nogil, parallel():
        for l in prange(repetitions):
            for m in range(M):
                xi = Xi[m,l]
                xr = Xr[m,l]            
                for k in range(Nd[2]):
                    kshift = k-n_shift[2]
                    for j in range(Nd[1]):
                        jshift = j-n_shift[1]
                        for i in range(Nd[0]):
                            ishift = i-n_shift[0]
                            idx = i+Nd[0]*(j+Nd[1]*(k + l*Nd[2]))
                            t = omega[m,0]*ishift + omega[m,1]*jshift + omega[m,2]*kshift
                            st = sin(t)
                            ct = cos(t)
                            xr_out[idx] = xr_out[idx] + ct * xr - st*xi
                            xi_out[idx] = xi_out[idx] + st * xr + ct*xi                        
    return 0

