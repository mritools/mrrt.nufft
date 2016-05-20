# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
from pyir.nufft.dtft import dtft, dtft_adj
from pyir.utils import max_percent_diff

from numpy.testing import assert_allclose

# data_dir = '/media/Data1/src_repositories/my_git/pyrecon/pyir.nufft/test/data'

__all__ = ['test_dtft_3d',
           'test_dtft_2d',
           'test_dtft_1d',
           'test_dtft_adj_3d',
           'test_dtft_adj_2d',
           'test_dtft_adj_1d']


def test_dtft_3d(verbose=False):
    """ function dtft_test() """
    Nd = np.asarray([4, 6, 5]) * 2
    n_shift = np.asarray([1, 3, 2]).reshape(3, 1)
    rstate = np.random.RandomState(1234)
    x = rstate.standard_normal(tuple(Nd))	#test signal
    pi = np.pi
    # test with uniform frequency locations
    o1 = 2*pi*np.arange(0,Nd[0])/Nd[0]
    o2 = 2*pi*np.arange(0,Nd[1])/Nd[1]
    o3 = 2*pi*np.arange(0,Nd[2])/Nd[2]
    o1, o2, o3 =np.meshgrid(o1, o2, o3, indexing='ij')
    om = np.hstack((o1.reshape((-1, 1), order='F'),
                    o2.reshape((-1, 1), order='F'),
                    o3.reshape((-1, 1), order='F')))

    # DTFT result       
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)
    
    # compare to FFT-based result
    Xf = np.fft.fftn(x)
    # phase shift
    Xf = Xf.ravel(order='F') * np.exp(1j * (np.dot(om, n_shift)))[:, 0]
    
    assert_allclose(np.squeeze(Xd), Xf, atol=1e-7)
    if verbose:
        max_err = np.max(np.abs(np.squeeze(Xd)-np.squeeze(Xf)))
        print('max error = %g' % max_err)


def test_dtft_2d(verbose=False):
    """ function dtft_test() """
    Nd = np.asarray([4, 6]) * 2
    n_shift = np.asarray([1, 3]).reshape(2, 1)
    rstate = np.random.RandomState(1234)
    x = rstate.standard_normal(tuple(Nd))	#test signal
    pi = np.pi
    # test with uniform frequency locations
    o1 = 2*pi*np.arange(0,Nd[0])/Nd[0]
    o2 = 2*pi*np.arange(0,Nd[1])/Nd[1]
    o1, o2 =np.meshgrid(o1, o2, indexing='ij')
    om = np.hstack((o1.reshape((-1, 1), order='F'),
                    o2.reshape((-1, 1), order='F')))

    # DTFT result       
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)
    
    # compare to FFT-based result
    Xf = np.fft.fftn(x)
    # phase shift
    Xf = Xf.ravel(order='F') * np.exp(1j * (np.dot(om, n_shift)))[:, 0]
    
    assert_allclose(np.squeeze(Xd), Xf, atol=1e-7)
    if verbose:
        max_err = np.max(np.abs(np.squeeze(Xd)-np.squeeze(Xf)))
        print('max error = %g' % max_err)


def test_dtft_1d(verbose=False):
    """ function dtft_test() """
    Nd = np.asarray([16, ])
    n_shift = np.asarray([5, ]).reshape(1, 1)
    rstate = np.random.RandomState(1234)
    x = rstate.standard_normal(tuple(Nd))	#test signal
    pi = np.pi
    # test with uniform frequency locations
    o1 = 2*pi*np.arange(0,Nd[0])/Nd[0]
    om = o1.reshape((-1, 1), order='F')

    # DTFT result       
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)
    
    # compare to FFT-based result
    Xf = np.fft.fftn(x)
    # phase shift
    Xf = Xf.ravel(order='F') * np.exp(1j * (np.dot(om, n_shift)))[:, 0]
    
    assert_allclose(np.squeeze(Xd), Xf, atol=1e-7)
    if verbose:
        max_err = np.max(np.abs(np.squeeze(Xd)-np.squeeze(Xf)))
        print('max error = %g' % max_err)


def test_dtft_adj_3d(verbose=False, test_Cython=False):
    
    #Nd = [4, 6, 5];
    Nd = [32, 16, 2]
    n_shift = np.asarray([2, 1, 3]).reshape(3, 1)
    #n_shift = [0,0,0];
    # test with uniform frequency locations:
    o1 = 2*np.pi*np.arange(Nd[0])/Nd[0]
    o2 = 2*np.pi*np.arange(Nd[1])/Nd[1]
    o3 = 2*np.pi*np.arange(Nd[2])/Nd[2]
    o1, o2, o3 = np.meshgrid(o1, o2, o3, indexing='ij')
    X = o1 + o2 - o3; # test spectrum
    om=np.hstack((o1.reshape((-1, 1), order='F'),
                  o2.reshape((-1, 1), order='F'),
                  o3.reshape((-1, 1), order='F')))
                  
    xd = dtft_adj(X, om, Nd, n_shift)
    xl = dtft_adj(X, om, Nd, n_shift, True)
    assert_allclose(xd, xl, atol=1e-7)
    
    Xp = np.exp(-1j * np.dot(om, n_shift))
    Xp = X * Xp.reshape(X.shape, order='F')
    xf = np.fft.ifftn(Xp) * np.prod(Nd)
    assert_allclose(xd, xf, atol=1e-7)
    if verbose:
        print('loop max %% difference = %g' % max_percent_diff(xl, xd))
        print('ifft max %% difference = %g' % max_percent_diff(xf,xd))
    
    if test_Cython:
        import time
        from pyir.nufft.cy_dtft import dtft_adj as cy_dtft_adj
        t_start=time.time()
        xc = cy_dtft_adj(X.ravel(order='F'), om, Nd, n_shift)
        print("duration (1 rep) = {}".format(time.time()-t_start))
        print('ifft max %% difference = %g' % max_percent_diff(xf,xc))
        
        X_16rep = np.tile(X.ravel(order='F')[:,None],(1,16))
        t_start=time.time()
        xc16 = cy_dtft_adj(X_16rep, om, Nd, n_shift)
        print("duration (16 reps) = {}".format(time.time()-t_start))
        t_start=time.time()
        X_64rep = np.tile(X.ravel(order='F')[:,None],(1,64))
        xc64 = cy_dtft_adj(X_64rep, om, Nd, n_shift)
        max_percent_diff(xf,xc64[...,-1])
        print("duration (64 reps) = {}".format(time.time()-t_start))
#        %timeit xd = dtft_adj(X_16rep, om, Nd, n_shift);

    return


def test_dtft_adj_2d(verbose=False):
    N1 = 4
    N2 = 6
    n_shift = np.asarray([2, 1]).reshape(2, 1)
    
    # test with uniform frequency locations:
    o1 = 2*np.pi*np.arange(0, N1) / float(N1)
    o2 = 2*np.pi*np.arange(0, N2) / float(N2)
    o1, o2 = np.meshgrid(o1, o2, indexing='ij')
    X = o1 + o2        # test spectrum
    om = np.hstack((o1.reshape((-1, 1), order='F'),
                    o2.reshape((-1, 1), order='F')));
    xd = dtft_adj(X, om, [N1, N2], n_shift)
    xl = dtft_adj(X, om, [N1, N2], n_shift, True)
    assert_allclose(xd, xl, atol=1e-7)
    #print('loop max %% difference = %g' % max_percent_diff(xl,xd))
    #Xp = X * np.reshape(np.exp(-1j * np.dot(om, np.asarray(n_shift).ravel().T)), X.shape);
    Xp = np.exp(-1j * np.dot(om, n_shift))
    Xp = X * Xp.reshape(X.shape, order='F')
    xf = np.fft.ifftn(Xp) * N1 * N2
    assert_allclose(xd, xf, atol=1e-7)       
    if verbose:
        print('ifft max %% difference = %g' % max_percent_diff(xf,xd))
    return


def test_dtft_adj_1d(verbose=False):
    N1 = 16
    n_shift = np.asarray([2, ]).reshape(1, 1)
    
    # test with uniform frequency locations:
    o1 = 2*np.pi*np.arange(0, N1) / float(N1)
    X = o1        # test spectrum
    om = o1.reshape((-1, 1), order='F')
    xd = dtft_adj(X, om, N1, n_shift)
    xl = dtft_adj(X, om, N1, n_shift, True)
    assert_allclose(xd, xl, atol=1e-7)
    #print('loop max %% difference = %g' % max_percent_diff(xl,xd))
    #Xp = X * np.reshape(np.exp(-1j * np.dot(om, np.asarray(n_shift).ravel().T)), X.shape);
    Xp = np.exp(-1j * np.dot(om, n_shift))
    Xp = X * Xp.reshape(X.shape, order='F')
    xf = np.fft.ifftn(Xp) * N1
    assert_allclose(xd, xf, atol=1e-7)       
    if verbose:
        print('ifft max %% difference = %g' % max_percent_diff(xf,xd))
    return