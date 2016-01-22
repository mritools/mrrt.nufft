from __future__ import division, print_function, absolute_import

import numpy as np

""" Slow, brute force DTFT routines.  These can be used to validate the NUFFT
on small problem sizes. """


def dtft(x, omega, Nd=None, n_shift= None, useloop=False):
    """  Compute d-dimensional DTFT of signal x at frequency locations omega
    In
    	x	[[Nd],L]	signal values
    	omega	[M,dd]		frequency locations (radians)
    	n_shift [dd,1]		use [0:N-1]-n_shift (default [0 0])
    	useloop			1 to reduce memory use (slower)
     Out
    	X	[M,L]		DTFT values
    
     Requires enough memory to store M * prod(Nd) size matrices (for testing only)
    
    Matlab version Copyright 2001-9-17, Jeff Fessler, The University of Michigan
    """

    dd = omega.shape[1]
    if Nd is None:
        if x.ndim == dd+1:
            Nd = x.shape[:-1]
        elif x.ndim == dd:
            Nd = x.shape
        else:
            raise ValueError("Nd must be specified")
    Nd = np.atleast_1d(Nd)
    
    if len(Nd) == dd:		# just one image
        x = x.ravel(order='F')
        x = x[:, np.newaxis]
    elif len(Nd) == dd+1:	# multiple images
        Nd = Nd[:-1]
        x = np.reshape(x, (np.prod(Nd), -1))	# [*Nd,L]
    else:
        print('bad input signal size')

    if n_shift is None:
        n_shift=np.zeros(dd)
    n_shift = np.atleast_1d(np.squeeze(n_shift))
    if len(n_shift) != dd:
        raise ValueError("must specify one shift per axis")        

    if np.any(n_shift != 0):
        nng = []
        for d in range(dd):
            nng.append(np.arange(0, Nd[d]) - n_shift[d])
        nng = np.meshgrid(*nng, indexing='ij')
    else:
        nng = np.indices(Nd)

    if useloop:
        #
        # loop way: slower but less memory
        #
        M = len(omega)
        X = np.zeros((x.size // np.prod(Nd), M),
                     dtype=np.result_type(x.dtype, omega.dtype,
                                          np.complex64))	# [L,M]
        if omega.shape[1] < 3:
            # trick: make '3d'
            omega = np.hstack((omega,
                               np.zeros(omega.shape[0])[:, np.newaxis]))
        for d in range(dd):
            nng[d] = nng[d].ravel(order='F')
        for mm in range(0, M):
            tmp = omega[mm, 0] * nng[0]
            for d in range(1, dd):
                tmp += omega[mm, d] * nng[d]
            X[:, mm] = np.dot(np.exp(-1j*tmp), x)
        X = X.T  # [M,L]
    else:
        X = np.outer(omega[:, 0], nng[0].ravel(order='F'))
        for d in range(1, dd):
            X += np.outer(omega[:, d], nng[d].ravel(order='F'))
        #X = np.asmatrix(np.exp(-1j*X)) * np.asmatrix(x).T
        X = np.dot(np.exp(-1j*X), x)
    
    return X


def dtft_adj(X, omega, Nd=None, n_shift=None, useloop=False):
    """  Compute adjoint of d-dim DTFT for spectrum X at frequency locations omega
    
     In
    	X	[M,L]		dD DTFT values
    	omega	[M,d]		frequency locations (radians)
    	n_shift [d,1]		use [0:N-1]-n_shift (default [0 ... 0])
    	useloop			1 to reduce memory use (slower)
     Out
    	x	[(Nd),L]	signal values
    
     Requires enough memory to store M * (*Nd) size matrices. (For testing only)
    
    Matlab version: Copyright 2003-4-13, Jeff Fessler, The University of Michigan
    """
    dd = omega.shape[1]
    if Nd is None:
        if X.ndim == dd+1:
            Nd = X.shape[:-1]
        elif X.ndim == dd:
            Nd = X.shape
        else:
            raise ValueError("Nd must be specified")
    Nd = np.atleast_1d(Nd)
    if len(Nd) == dd:		# just one image
        X = X.ravel(order='F')
        X = X[:, np.newaxis]
    elif len(Nd) == dd+1:	# multiple images
        Nd = Nd[:-1]
        X = np.reshape(X, (np.prod(Nd), -1))	# [*Nd,L]
    else:
        print('bad input signal size')

    if len(Nd) != dd:
        raise ValueError("length of Nd must match number of columns in omega")       
        
    if n_shift is None:
        n_shift = np.zeros(dd)
    n_shift = np.atleast_1d(np.squeeze(n_shift))
    if len(n_shift) != dd:
        raise ValueError("must specify one shift per axis")       

    if np.any(n_shift != 0):
        nn = []
        for id in range(dd):
            nn.append(np.arange(0, Nd[id]) - n_shift[id])
        nn = np.meshgrid(*nn, indexing='ij')
    else:
        nn = np.indices(Nd)

    if useloop:
        # slower, but low memory
        M = omega.shape[0];
        x = np.zeros(Nd)	# [(Nd),M]
        for mm in range(0, M):
            t = omega[mm, 0]*nn[0]
            for d in range(1, dd):
                t += omega[mm, d]*nn[d]
            x = x + np.exp(1j*t) * X[mm] #X[mm,:];
    else:
       x = np.outer(nn[0].ravel(order='F'), omega[:, 0])
       for d in range(1, dd):
           x += np.outer(nn[d].ravel(order='F'), omega[:, d])
       x = np.dot(np.exp(1j*x[:, np.newaxis]), X)  # [(*Nd),L]
       x = np.reshape(x, Nd, order='F')	# [(Nd),L]
    return x
