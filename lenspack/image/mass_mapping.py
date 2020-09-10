#! /usr/bin/env Python
'''
Created on April 9 2020

@authors: Kostas Themelis & Jean-Luc Starck & Austin Peel
'''

import os
import numpy as np
from astropy.io import fits
from scipy import ndimage
import math
import matplotlib.pyplot as plt
import astropy
import pylab
from os import remove
from subprocess import check_call
from datetime import datetime
import matplotlib
from subprocess import check_call
import readline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pysap.extensions import sparse2d
import pyqtgraph
from pyqtgraph.Qt import QtGui
import numpy as np
from numpy import linalg as LA
from sys import getsizeof
from scipy.special import erf

import pycs as ps
import pycs
from pycs.sparsity.sparse2d.starlet import *
from pycs.misc.cosmostat_init import *
from pycs.misc.mr_prog import *
from pycs.misc.utilHSS import *
from pycs.misc.im1d_tend import *
from pycs.misc.stats import *
from pycs.sparsity.sparse2d.dct import dct2d, idct2d
from pycs.sparsity.sparse2d.dct_inpainting import dct_inpainting
import pycs.sparsity.sparse2d.starlet


def get_ima_spectrum_map(Px,nx,ny):
    """
    Create an isotropic image from a power spectrum
        Ima[i+nx/2, j+ny/2] = Px[ sqrt(i^2 + j^j) ]

    Parameters
    ----------
    Px : : np.ndarray
        1D powspec.
    nx,ny : int
        image size to be created.

    Returns
    -------
    power_map : np.ndarray
        2D image.
    """
    Np = Px.shape[0]
#    print("nx = ", nx, ", ny = ", ny, ", np = ", Np)
    k_map =  np.zeros((nx, ny))
    power_map = np.zeros((nx, ny) )
#    info(k_map)
    for (i,j), val in np.ndenumerate(power_map):
        k1 = i - nx/2.0
        k2 = j - ny/2.0
        k_map[i, j] = (np.sqrt(k1*k1 + k2*k2))
        if k_map[i,j]==0:
            power_map[i, j] = 0.
        else:
            ip = int(k_map[i, j])
            if ip < Np:
                   power_map[i, j] = Px[ip]                
    return power_map



class shear_data():
    '''
    Class for input data, containing the shear components g1,g2, the covariance matrix,
    the theoretical convergence power spectrum. 
    '''
    g1=0    # shear 1st component
    def __init__(self): # __init__ is the constructor
        self.g1=0
 
    g2=0    # shear 2nd component
    Ncov=0  # diagonal noise cov mat of g = g1 + 1j g2,  of same size as g1 and g2
            # the noise cov mat relative to g1 alone is Ncov /2.   (same for g2)
    mask=0  # mask 
    ktr=0   # true kappa (...for simulations)
    g1t=0   # true g1
    g2t=0   # true g2
    ps1d=0  # theoretical convergence power spectrum
    nx=0
    ny=0

    # file names
    DIR_Input=0 # dir input data
    g1_fn=0     # g1 file name
    g2_fn=0     # g2 file name
    ktr_fn=0    # for sumulation only, true convergence map
    ps1d_fn=0   # Convergence map 1D theoretical power spectrum used for Wiener filtering
    ncov_fn=0   # covariance filename
    

    def get_shear_noise(self,FillMask=False):
        """
        Return a noise realisation using the covariance matrix.
        If FillMask is True, the non observed area where the covariance is infinitate, 
        will be filled with randon values with a variance value equals to the maximum 
        variance in the observed area, i.e. where the mask is 1.

        Parameters
        ----------
        FillMask : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        n1 : np.ndarray
            noise realisation for g1.
        n2 : np.ndarray
            noise realisation for g2.

        """
        Mat = np.sqrt(self.Ncov / 2.)
        if FillMask == True:
            ind = np.where(self.mask == 1)
            MaxCov = np.max(Mat[ind])
            ind = np.where(self.mask == 0)
            Mat[ind] = MaxCov
            #info(Mat, name="cov")
            #info(Mat*self.mask, name="cov")
            #print("MaxCov = ", MaxCov)
            #tvima(self.mask)
        n1 = np.random.normal(loc=0.0, scale=Mat)
        n2 = np.random.normal(loc=0.0, scale=Mat)
        return n1,n2

# class shear_simu():
#     def __init__(self):
#         a=0
#     a = np.random.normal(loc=0.0, scale=10.0, size=[200])


class massmap2d():
    """ Mass Mapping class
    This class contains the tools to reconstruct mass maps from shear measurements
    """
    
    kernel1 = 0   # internal variable for wiener filtering
    kernel2 = 0   # internal variable for wiener filtering
    nx=0          # image size  (number of lines)
    ny=0          # image size  (number of column)
    # WT=0          # Starlet wavelet Class defined in starlet.py
    Verbose = False # Parameter to switch on/off print
    DEF_niter=12     # Default number if iterations in the iterative methods.
    DEF_Nrea=10     # Default number of realizations.
    DEF_Nsigma=3.   # Default detection level in wavelet space
    niter_debias =0   # For space recovery using soft thresholding, a final 
                      # debiasing step could be useful. By default we don't any
                        # debiasing.
    DEF_FirstDetectScale=1 # default first detection scale in wavelet space.
                           # very often, the noise is highly dominating the 
                            # the signal, and the first or the first scales 
                            # can be removed.
                            # DEF_FirstDetectScale=2 => the  two finest scales
                            # are removed
    WT_Sigma = 0            # Noise standard deviation in the wavelet space
    WT_ActiveCoef=0         # Active wavelet coefficients
    SigmaNoise = 0          # Noise standard deviation in case of Gaussian white noise
    def __init__(self, name='mass'): # __init__ is the constructor
        self.WT = starlet2d()  # Starlet wavelet Class defined in starlet.py

    def init_massmap(self,nx,ny,ns=0):
        """
        Initialize the class for a given image size and a number of scales ns 
        to be used in the wavelet decomposition.
        If ns ==0, the number of scales is automatically calculated in the
        starlet initialization (see init_starlet, field self.WT.ns).

        Parameters
        ----------
        nx, ny : int
            Image size
        ns : int, optional
            Number of scales. The default is 0.

        Returns
        -------
        None.

        """
        self.nx=nx
        self.ny=ny
        self.WT = starlet2d(gen2=True,l2norm=True, bord=1, verb=False)
        self.WT.init_starlet(nx, ny, nscale=ns)
        self.WT.name="WT-MassMap"
        k1, k2 = np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny))
        denom = k1*k1 + k2*k2
        denom[0, 0] = 1  # avoid division by 0
        self.kernel1 = (k1**2 - k2**2)/denom
        self.kernel2 = (2*k1*k2)/denom
        if self.Verbose:
            print("Init Mass Mapping: Nx = ", nx, ", Ny = ", ny, ", Nscales = ", self.WT.ns)

    def inpaint(self, kappa, mask, niter=DEF_niter):
        """
        Apply the sparse inpainting recovery technique to an image using the 
        Discrete Cosine Transform. 

        Parameters
        ----------
        kappa : np.ndarray
                Input data array
        mask : TYPE
            DESCRIPTION.
        niter : TYPE, optional
            DESCRIPTION. The default is self.DEF_niter.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return dct_inpainting(kappa, mask, niter=niter)

    def get_theo_kappa_power_spectum(self, d, niter=None, PowSpecNoise=None, FirstFreqNoNoise=1):
        """
        Estimate the theoretical convergence power spectrum from the data themselfve. 
        Two methods are available:
            Method 1: Estimate inpainted ke and kb using iterative Kaiser-Squire method.
                    if PowSpecNoise==0, assume that there is no B-mode,
                    and the B-mode is used a noise power spectrum estimation.
                    powspec_Theo_E = powspec(ke) - powspec(kb)
                    powspec_Theo_B = 0 
                    powspec_Theo_Noise = powspec(kb)
            Method 2: Use the input noise power spectrum
                    Then:
                        powspec_Theo_E = powspec(ke) - PowSpecNoise
                        powspec_Theo_B = powspec(kb) - PowSpecNoise

        Parameters
        ----------
        d :  Class  shear_data
            Input Class describing the obervations.
        niter : int, optional
            Number of iterations in the iKS method. The default is None.
        PowSpecNoise : np.ndarray, optional
            Noise power spectrum. The default is 0.
        FirstFreqNoNoise : int, optional
            At very low frequencies, signal is dominating and we generally prefer to not 
            apply any denoising correction. So we will have :
                    powspec_Theo_Noise[0:FirstFreqNoNoise] = 0
            The default is 1.

        Returns
        -------
        pke : TYPE
            DESCRIPTION.
        pkb : TYPE
            DESCRIPTION.
        pn : TYPE
            DESCRIPTION.

        """
        k = self.iks(d.g1, d.g2, d.mask, niter=niter)
        ke = k.real
        kb = k.imag
        pe = im_isospec(ke)
        pb = im_isospec(kb)
        #fsky = mask.sum()/mask.size
        if PowSpecNoise is None:
            pn = pb
        else:
            pn = PowSpecNoise
        pn[0:FirstFreqNoNoise] = 0.
        pke = pe - pn
        pkb = pb - pn 
        
        # Denoise the estimated powsepc
        UseTendancyFiltering=False
        if UseTendancyFiltering is True:
            e1 = reverse(pke)
            fe1 = im1d_tend(e1)
            pke = reverse(fe1)
            b1 = reverse(pkb)
            fb1 = im1d_tend(b1)  # , opt='-T50'))
            pkb = reverse(fb1)
            pke[pke < 0] = 0 
            # the smoothing does not work very well above nx/2, 
            # because of the increase of the variance (frequencies at the corner)
            # we apodize the corner
            npix = pke.shape
            npix=npix[0]
            fp  = int(npix / np.sqrt(2.))
            min_end = pke[fp]
            pke[fp::]= pke[fp]  
            pke[fp::] = min_end

        # pke[pkb < 0] = 0 
        pkb[pkb < 0] = 0
        #pe = pe - pn/fsky
        #pb = pb - pn/fsky
        #pef=mr_prog(pe, prog="mr1d_filter -m5 -P ")
        #pbf=mr_prog(pb, prog="mr1d_filter -m5 -P ")
        tv=0
        if tv:
            plot(pke)
            plot(pkb)
            plot(pn)
            plot(d.ps1d)
        return pke, pkb, pn

    def get_tps(self, d, niter=None, Nrea=None):
        return self.get_theo_kappa_power_spectum(d,niter=niter)


    def kappa_to_gamma(self, kappa):
        """  
        This routine performs computes the shear field from the  convergence
        map (no B-mode).

        Parameters
        ----------
        kappa: np.ndarray
                Input convergence data array

        Returns
        -------
        g1,g2: np.ndarray
                complext output shear field
        Notes
        -----
        """
        (Nx,Ny) = np.shape(kappa)
        if self.nx != Nx or self.ny != Ny:
            self.init_massmap(Nx,Ny)
        k = np.fft.fft2(kappa)
        g1 = np.fft.ifft2(self.kernel1 * k) 
        g2 = np.fft.ifft2(self.kernel2 * k)
        return g1.real - g2.imag, g2.real + g1.imag 
 
    # Fast call
    def k2g(self, kappa):
        return self.kappa_to_gamma(kappa)

    def gamma_to_cf_kappa (self, g1, g2):
        """  
        This routine performs a direct inversion from shear to convergence,
        it return a comlex field, with the real part being the convergence (E mode), 
        the imaginary part being the B mode.

        Parameters
        ----------
        g1, g2: np.ndarray
                Input shear field

        Returns
        -------
        kappa: np.ndarray
                output complex convergence field
        Notes
        -----
        """
        if self.WT.nx == 0 or self.WT.ny == 0:
            (nx,ny) = np.shape(g1)
            self.WT.init_starlet(nx,ny,gen2=1,l2norm=1, name="WT-MassMap")
        g = g1 + 1j*g2
        return np.fft.ifft2((self.kernel1 - 1j*self.kernel2)* np.fft.fft2(g))

    def gamma_to_kappa (self, g1, g2):
        """  
        Same as gamma_to_cf_kappa, but returns only the E mode (convergence)

        Parameters
        ----------
        g1, g2: np.ndarray
                Input shear field

        Returns
        -------
        kappa: np.ndarray
                output convergence field
        Notes
        -----
        """
        k = self.gamma_to_cf_kappa (g1, g2)
        return k.real
    
    # Fast  interactive call to gamma_to_kappa
    def g2k(self, gam1, gam2):
        return self.gamma_to_kappa(gam1, gam2)

    def smooth(self, map, sigma=2.):
        """
        Gaussian smoothing of an image.

        Parameters
        ----------
        map : 2D np.ndarray
        input image.
        sigma : float, optional
            Standard deviation of the used Gaussian kernel. The default is 2..

        Returns
        -------
         np.ndarray
            Smoother array.

        """
        return ndimage.filters.gaussian_filter(map,sigma=sigma)

    def kaiser_squires(self, gam1, gam2, sigma=2.):
        """  
        This routine performs a direct inversion from shear to convergence,
        followed by a Gaussian filtering. 
        This is the standard Kaiser-Squires method. 

        Parameters
        ----------
        gam1, gam2: np.ndarray
                Input shear field
        sigma: float, optional
                Default is 2.
        Returns
        -------
        kappa: np.ndarray
                output convergence field
        Notes
        -----
        """
        ks = self.gamma_to_cf_kappa(gam1, gam2)
        ksg = ndimage.filters.gaussian_filter(ks.real,sigma=sigma)
        return ksg
    
    # Fast interactive call to kaiser_squires
    def ks(self, gam1,gam2,sigma=2.):
        return self.kaiser_squires(gam1, gam2, sigma=2.)
    
    def eb_kaiser_squires(self, gam1, gam2, sigma=2.):
        """  
        Same as kaiser_squires, but return also the B-mnode. 

        Parameters
        ----------
        gam1, gam2: np.ndarray
                Input shear field

        Returns
        -------
        E_kappa: np.ndarray
                output convergence field (E mode)
        B_kappa: np.ndarray
                output convergence field (B mode)
        Notes
        -----
        """
        ks = self.gamma_to_cf_kappa(gam1, gam2)
        ksg = ndimage.filters.gaussian_filter(ks.real,sigma=sigma)
        ksbg = ndimage.filters.gaussian_filter(ks.imag,sigma=sigma)
        return ksg, ksbg


    def H_operator_eb2g(self, ka_map, kb_map):
        """
        This routine converts (E,B) modes to shear

        Parameters
        ----------
        ka_map, kb_map : np.ndarray
            (E,B) mode
 
        Returns
        -------
        (g1,g2): np.ndarray
        output shear field
        None.
        """
        # ka_map and kb_map should be of the same size
        [nx,ny] = ka_map.shape
        g1_map = np.zeros((nx,ny))
        g2_map = np.zeros((nx,ny))
        ka_map_fft = np.fft.fft2(ka_map)
        kb_map_fft = np.fft.fft2(kb_map)
    
        f1, f2 = np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
        p1 = f1 * f1 - f2 * f2
        p2 = 2 * f1 * f2
        f2 = f1 * f1 + f2 * f2
    
        f2[0,0] = 1 # avoid division with zero
        kafc =  (p1 * ka_map_fft - p2 * kb_map_fft) / f2
        kbfc =  (p1 * kb_map_fft + p2 * ka_map_fft) / f2
        g1_map[:,:] = np.fft.ifft2(kafc).real
        g2_map[:,:] = np.fft.ifft2(kbfc).real
        return g1_map, g2_map
    # Fast interactice call to H_operator_eb2g
    def eb2g(self, ka_map, kb_map):
         return self.H_operator_eb2g(ka_map, kb_map)

    def H_adjoint_g2eb(self, g1_map, g2_map):
        """
        This routine reconstruct the (E,B) modes from the shear field

        Parameters
        ----------
        g1_map, g2_map : 2D np.ndarray
            shear field.
 
        Returns
        -------
        (E,B) modes : np.ndarray
            output convergence field
        None.
        """
        [nx,ny] = g1_map.shape
        kappa1 = np.zeros((nx,ny))
        kappa2 = np.zeros((nx,ny))
        g1_map_ifft = np.fft.ifft2(g1_map)
        g2_map_ifft = np.fft.ifft2(g2_map)
        f1, f2 = np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
        p1 = f1 * f1 - f2 * f2
        p2 = 2 * f1 * f2
        f2 = f1 * f1 + f2 * f2
        f2[0,0] = 1
        g1fc =  (p1 * g1_map_ifft + p2 * g2_map_ifft) / f2
        g2fc =  (p1 * g2_map_ifft - p2 * g1_map_ifft) / f2
        kappa1[:,:] = np.fft.fft2(g1fc).real
        kappa2[:,:] = np.fft.fft2(g2fc).real
        return kappa1, kappa2
    # Fast interactice call to H_adjoint_g2eb 
    def g2eb(self, g1_map, g2_map):
        return self.H_adjoint_g2eb(g1_map, g2_map)

    def get_wt_noise_level(self, InshearData,Nrea=DEF_Nrea):
        """
        Computes the noise standard deviation for each wavelet coefficients of 
        the convergence map, using Nrea noise realisations of the shear field

        Parameters
        ----------
        InshearData : Class  shear_data
            Input Class describing the obervations.
        Nrea : int, optional
            Number of noise realisations. The default is 20.

        Returns
        -------
        WT_Sigma : 3D np.ndarray
            WT_Sigma[s,i,j] is the noise standard deviation at scale s and position
            (i,j) of the convergence.

        """
        mask = InshearData.mask
        Ncov = InshearData.Ncov
        for i in np.arange(Nrea):
            n1,n2 = InshearData.get_shear_noise(FillMask=True)
            ke, kb = self.g2eb(n1,n2)
            self.WT.transform(ke)
            if i == 0:
                WT_Sigma = np.zeros((self.WT.ns,self.WT.nx,self.WT.ny))
            WT_Sigma += (self.WT.coef)** 2.   # by definition the mean of wt
                                             # is zero.
        WT_Sigma = np.sqrt(WT_Sigma / Nrea)
        # info(WT_Sigma)
        return WT_Sigma 

    def get_active_wt_coef(self, InshearData, UseRea=False, SigmaNoise=1., Nsigma=None, Nrea=None, WT_Sigma=None, FirstDetectScale=DEF_FirstDetectScale, OnlyPos=False, ComputeWTCoef=True):
        """
        Estimate the active set of coefficents, i.e. the coefficients of the 
        convergence map with an absolute value large than Nsigma * NoiseStandardDeviation.
        It returns a cube  A[s,i,j] containing 0 or 1. 
        If A[s,i,j] == 1 then we consider we have a detection at scale s and position (i,j).

        Parameters
        ----------
        InshearData : Class  shear_data
            Input Class describing the obervations.
        UseRea : bool, optional
            If true, make noise realisation to estimate the detection level in
            wavelet space. The default is False.
        Nrea : int, optional
            Number of noise realisations. The default is None.
        WT_Sigma : 3D np.ndarray, optional
            WT_Sigma[s,i,j] is the noise standard deviation at scale s and position
            (i,j) of the convergence. If it not given, the function get_wt_noise_level
            is used to calculate it.
        SigmaNoise: int, optional
            When UseRea==False, assume Gaussian nosie with standard deviation equals to SigmaNoise.
            Default is 1
        Nsigma : int, optional
            level of detection (Nsigma * noise_std). The default is None.
        FirstDetectScale: int, optional
            detect coefficients at scale < FirstDetectScale
        OnlyPos: Bool, optional
            Detect only positive wavelet coefficients. Default is no.
        ComputeWTCoef: bool, optional
            if true, recompute the wavelet coefficient  from the shear data.
            Default is true.
        Returns
        -------
        WT_Active : 3D np.ndarray
            WT_Active[s,i,j] = 1 if an coeff of the convergence map is larger 
            than Nsigma * Noise std
        """

        if ComputeWTCoef:
            e,b = self.g2eb(InshearData.g1, InshearData.g2)
            self.WT.transform(e)
        
        WT_Support = self.WT.coef * 0.
        Last = self.WT.ns - 1
        if UseRea and WT_Sigma is None:
            WT_Sigma = self.get_wt_noise_level(InshearData,Nrea=Nrea)

        if Nsigma is None:
            Nsigma = self.DEF_Nsigma
        if Nrea is None:
            Nrea=self.DEF_Nrea
        if FirstDetectScale is None:
            FirstDetectScale=DEF_FirstDetectScale
        for j in range(Last):
            wtscale=self.WT.get_scale(j)
            #if j == 2:
            #    tvilut(wtscale,title='scale2')
            if j == 0:
                Nsig= Nsigma + 1
            else:
                Nsig = Nsigma
            #vThres = WT_Sigma * Nsigma * self.WT.TabNorm[j]
            if OnlyPos is False:
                if UseRea :
                    wsigma=WT_Sigma[j,:,:]
                    ind= np.where( np.abs(wtscale) > wsigma * Nsig * self.WT.TabNorm[j])
                    # WT_Support[j,:,:] = np.where( np.abs(self.WT.coef[j,:,:]) > WT_Sigma[j,:,:] * Nsig * self.WT.TabNorm[j], 1, 0)
                else:
                    ind= np.where( np.abs(wtscale) > SigmaNoise * Nsig * self.WT.TabNorm[j])
                    #WT_Support[j,:,:] = np.where( np.abs(self.WT.coef[j,:,:]) > SigmaNoise * Nsig * self.WT.TabNorm[j], 1, 0)
            else:
                if UseRea :
                    wsigma=WT_Sigma[j,:,:]
                    # WT_Support[j,:,:] = np.where( self.WT.coef[j,:,:] > WT_Sigma[j,:,:] * Nsig * self.WT.TabNorm[j], 1, 0)
                    ind = np.where( wtscale > wsigma * Nsig * self.WT.TabNorm[j])
                else:
                    T = SigmaNoise * Nsig * self.WT.TabNorm[j]
                    ind = np.where( wtscale > T)
                    # WT_Support[j,:,:] = np.where( self.WT.coef[j,:,:] > SigmaNoise * Nsig * self.WT.TabNorm[j], 1, 0)
            wtscale[:,:]=0
            wtscale[ind]=1  
            #if j == 2:
            #    tvilut(wtscale,title='sup2')
            WT_Support[j,:,:]= wtscale 
        if FirstDetectScale > 0:
            WT_Support[0:FirstDetectScale,:,:] = 0
        WT_Support[Last,:,:]=1
        self.WT_ActiveCoef=WT_Support
        return WT_Support

    def get_noise_powspec(self, CovMat,mask=None,nsimu=100, inpaint=False):
        """
        Build the noise powerspectum from the covariance map of the gamma field.
        Parameters
        ----------
        CovMat : : 2D np.ndarray
            covariance matrix of the shier field.
        mask : 2D np.ndarray, optional
            Apply a mask to the simulated noise realisation. The default is None.
        nsimu : int, optional
            Number of realisation to estimate the noise power spectrum. The default is 100.
        inpaint: Bool, optional
            Compute the power spectrum on inpainted Kaiser-Squires maps rather than on the masked
            maps. If inpaint==False, the estimated noise power spectrum is biased and 
            should be corrected from .the fraction of sky not observed (i.e. fsky).
            Default is No
        Returns
        -------
        px : 1D np.ndarray
            Estimated Power spectrum from noise realizations.

        """
        if mask is None:
            m = 1.
        else:
            m = mask
        for i in np.arange(nsimu):
            n1 = np.random.normal(loc=0.0, scale=np.sqrt(CovMat/2.))*m
            n2 = np.random.normal(loc=0.0, scale=np.sqrt(CovMat/2.))*m
            if mask is not None and inpaint is True:
                k = self.iks(n1,n2,mask)
            else:
                k = self.gamma_to_cf_kappa(n1,n2)
            p = im_isospec(k.real)
            
            if i==0:
                Np= p.shape[0]
                TabP = np.zeros([nsimu,Np], dtype = float)
            TabP[i,:] = p      
        px = np.mean(TabP, axis=0)
        return px


    def mult_wiener(self, map, WienerFilterMap):
        """" apply one wiener step in the iterative wiener filtering """
        return np.fft.ifft2(np.fft.fftshift(WienerFilterMap * np.fft.fftshift(np.fft.fft2(map))))
        
    def wiener(self, gamma1, gamma2, PowSpecSignal, PowSpecNoise):
        """
        Compute the standard wiener mass map.
        Parameters
        ----------
        gamma1,  gamma2: 2D np.ndarray
            shear fied.
        PowSpecSignal : 1D np.ndarray
            Signal theorical power spectrum.
        PowSpecNoise: 1D np.ndarray, optional
            noise theorical power spectrum.
        Returns
        -------
        TYPE  2D np.ndarray
              (E,B) reconstructed modes. Convergence = E
        """
        (nx,ny) = gamma1.shape
        
        if self.Verbose:
            print("Wiener filtering: ", nx, ny)
        #if mask is None:
        #    print("Wiener NO MASK")
        # info(gamma1, name="Wiener g1: ")
        # info(gamma2, name="Wiener g2: ")
        # info(PowSpecSignal, name="Wiener PowSpecSignal: ")
        # info(Ncv, name="Wiener Ncv: ")
        # if isinstance(PowSpecNoise, int):

        Ps_map = get_ima_spectrum_map(PowSpecSignal,nx,ny)
        Pn_map = get_ima_spectrum_map(PowSpecNoise,nx,ny)
        Den = (Ps_map + Pn_map)
        ind = np.where(Den !=0)
        Wfc = np.zeros((nx,ny))
        Wfc[ind] = Ps_map[ind] / Den[ind]
        t= self.gamma_to_cf_kappa (gamma1, gamma2)                            # xg + H^T(eta / Sn * (y- H * xg))
        kw = self.mult_wiener(t,Wfc)
        retr = np.zeros((nx,ny))
        reti = np.zeros((nx,ny))
        retr[:,:] = kw.real
        reti[:,:] = kw.imag

        # info(kw.real, name="WIENER OUTPUT: ")
        return retr, reti

    def get_lmax_dct_inpaint(self, gamma1, gamma2):
        """ return the maximum of the DCT absolute value of the convergence map """ 
        #image, icf = self.H_adjoint_g2eb(gamma1, gamma2)
        eb=self.gamma_to_cf_kappa(gamma1, gamma2)
        lmax = np.max(np.abs(dct2d(eb.real, norm='ortho')))
        return lmax

    def step_dct_inpaint(self, xg, xn, mask, n, niter, lmin, lmax, InpaintAlsoImag=True):
        """
        Apply one step of a iterative DCT inpainting method.
        First, we replace in xg value in mask[] == 0, obtained from the previous 
        inpainting at iter n-1, and a hard thresholding in the DCT domain 
        is applied to xg, on both real and imaginary parts. 

        Parameters
        ----------
        xg :  2D np.cfarray
            convergence field.
        xn : 2D np.cfarray
            convergence field at the previous iteration.
        mask : 2D np.ndarray
            mask related to the observed data.
            mask[i,j] = 1 if the observed shear field has information 
            at pixel (i,j) 
        n : int
            iteration number in the interative method. n must be in [0,niter-1]
        niter : int
            number of iteration in the iterative method.
        lmin,max : int
            minimum and maximum absolute values of  the DCT transform 
            of the shear E-mode map
        InpaintAlsoImag : bool, optional
            If true, both real and imaginary part are inpainted
            Otherwise on the real part is inpainted. The default is True.

        Returns
        -------
        xg : 2D np.cfarray
            inpainted convergence field.
        """
        (nx,ny) = xg.shape
        ret = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))

        lval = lmin + (lmax - lmin) * (1 - erf(2.8 * n / niter))  # exp decay
        # real part 
        ima = mask * xg.real + (1-mask) * xn.real
        alpha = dct2d(ima, norm='ortho')
        new_alpha = np.copy(alpha)  # Can we do this without copying ?
        new_alpha[np.abs(new_alpha) <= lval] = 0
        rec = idct2d(new_alpha, norm='ortho')
        # Enforce std. dev. constraint inside the mask
        std_out = rec[mask.astype(bool)].std()
        std_in = rec[~mask.astype(bool)].std()
        
        MultiScaleConstraint=False
        if std_in != 0:
            if not MultiScaleConstraint:
                rec[~mask.astype(bool)] *= std_out / std_in
            else:
                self.WT.transform(rec)
                for j in range(self.WT.ns):
                    scale = self.WT.get_scale(j)
                    std_out = scale[mask.astype(bool)].std()
                    std_in = scale[~mask.astype(bool)].std() 
                    scale[~mask.astype(bool)] *= std_out / std_in
                    self.WT.put_scale(scale, j)
                    rec = self.WT.recons()

        # imaginary part 
        if InpaintAlsoImag:
            ima = mask * xg.imag + (1-mask) * xn.imag
            alpha = dct2d(ima, norm='ortho')
            new_alpha = np.copy(alpha)  # Can we do this without copying ?
            new_alpha[np.abs(new_alpha) <= lval] = 0
            reci = idct2d(new_alpha, norm='ortho')
            # Enforce std. dev. constraint inside the mask
            std_out = reci[mask.astype(bool)].std()
            std_in = reci[~mask.astype(bool)].std()
            if std_in != 0:
                reci[~mask.astype(bool)] *= std_out / std_in
                
        else:
            reci = 0.*rec
        ret[:,:] = rec + 1j * reci
        return ret

    def iks(self, g1, g2, mask, niter=None, dctmax=None):
        """
        Iterative Kaiser-Squires with DCT inpainting.

        Parameters
        ----------
        g1,g2 : np.ndarray
                Input convergence data array
         mask : np.ndarray
            mask  of missing data.
        niter : int, optional
            Number of iteration. The default is None.

        Returns
        -------
            2D complex np.ndarray
               output complex EB field
        """
        if niter is None:
            niter = self.DEF_niter
        lmin = 0
        nx,ny=g1.shape
        xg = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        r1=np.zeros((nx,ny))
        r2=np.zeros((nx,ny))
        if dctmax is None:
            ks =  self.gamma_to_cf_kappa(g1*mask, g2*mask) 
            lmax = self.get_lmax_dct_inpaint(ks.real, ks.imag)
        else:
            lmax = dctmax

        for n in range(niter):
            t1,t2 = self.H_operator_eb2g(xg.real, xg.imag)  
            r1[:,:] = mask*(g1 - t1)
            r2[:,:] = mask*(g2 - t2)
            t1,t2 = self.H_adjoint_g2eb(r1, r2) 
            xg[:,:] = xg + (t1 + 1j*t2)                                             # xg + H^T(eta / Sn * (y- H * xg))
            xg[:,:] = self.step_dct_inpaint(xg, xg, mask, n, niter, lmin, lmax)
        return xg


    def get_resi(self, xg, gamma1, gamma2, ResiWeight, mask=None, niter=None, dctmax=None):
        """
        Compute the residual from an estimation of the convergence map.
        The return residual is on KappaE et KappaB.
        If a mask is given, then an inpainting iterative Kaiser-Squire
        method is used to backproject the residual from shear space to 
        kappa space.

        Parameters
        ----------
        xg :  2D np.cfarray
            Estimated convergence field
        gamma1, gamma2 : 2D np.ndarray
            data: shear measurements.
        ResiWeight : 2D np.ndarray
            Weights to apply on shear residual.
        Mask:  2D np.ndarray, optional
            Weight to apply to the shear compoenents
            Default is none.
        niter: int, optional
            Number of iteration in the inpainting. Default is self.NITER
        dctmax: float, optional
            first threshold used in the inpainting. Default is automotically
            estimated.
        Returns
        -------
        resi_kappa_e : 2D np.ndarray
            residual E mode.
        resi_kappa_b : 2D np.ndarray
            residual B mode.

        """
        (nx,ny) = xg.shape
        r1=np.zeros((nx,ny))
        r2=np.zeros((nx,ny))
        t1,t2 = self.H_operator_eb2g(xg.real, xg.imag)  
        r1[:,:] = ResiWeight*(gamma1 - t1)
        r2[:,:] = ResiWeight*(gamma2 - t2)
        
        if mask is None :
            # H * xg
            t1,t2 = self.H_adjoint_g2eb(r1, r2) 
            r1[:,:]=t1
            r2[:,:]=t2    
        else:
            if niter is None:
                niter = self.DEF_niter
            # iterative Kaiser Squires with inpainting
            xi = self.iks(r1, r2, mask, niter=niter,dctmax=dctmax)
            r1[:,:]=xi.real
            r2[:,:]=xi.imag
        #resi = (gamma1 + 1j * gamma2 -  self.gamma_to_cf_kappa(gamma1,gamma2)) * ResiWeight
        return r1,r2

    def prox_wiener_filtering(self, gamma1, gamma2, PowSpecSignal, NcvIn, Pn=None, niter=None, Inpaint=False, ktr=None, PropagateNoise=None):
        """
        Compute the wiener mass map considering not stationary noise
        Proximal wiener method published in:
            J. Bobin, J.-L. Starck, F. Sureau, and J. Fadili, 
            "CMB map restoration", Advances in Astronomy , 2012, Id703217, 2012.
        Parameters
        ----------
        gamma1,  gamma2: 2D np.ndarray
            shear fied.
        PowSpecSignal : 1D np.ndarray
            Signal theorical power spectrum.
        Ncv : 2D np.ndarray
            Diagonal covariance matrix (same size as gamma1 and gamm2), i.e. variance per pixel
        Pn: 1D np.ndarray, optional
            noise theorical power spectrum.
        niter: int
            number of iterations. Default is DEF_niter
        Inpaint: bool, optional
            if true, inpainting the missing data. Default is false.
        ktr: 2D np.ndarray, optional
            true convergence map, known in case simulated data are used.
            if given, errors are calculated at each iteration.
        PropagateNoise: Bool, optional
            if True, run the routine on a noise realization instead of the input shear field.
        Returns
        -------
        TYPE  2D np.ndarray
              (E,B) reconstructed modes. Convergence = E
        """
        if niter is None:
            niter = self.DEF_niter
        (nx,ny) = gamma1.shape     
        if self.Verbose:
            print("Iterative Wiener filtering: ", nx, ny, ", Niter = ", niter)
        Ncv = NcvIn / 2.
        Ncv[Ncv==0] = 1e9  # infinite value for no measurement
        index = np.where(Ncv<1e2)
        mask = np.zeros((nx,ny))
        mask[index] = 1
    
        xg = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
    
        # find the minimum noise variance
        ind = np.where(Ncv != 0)
        tau = np.min(Ncv[ind])
    
        # set the step size
        # eta = 1.83 * tau
        eta = tau
        # compute signal coefficient
        Esn = eta / Ncv
        Esn[Esn == np.inf] = 0

        # calculate the wiener filter coefficients
        Px_map = get_ima_spectrum_map(PowSpecSignal,nx,ny)
        # info((Px_map + eta))
        Wfc = np.zeros((nx,ny))
        if Pn is not None:
            Pn_map = get_ima_spectrum_map(Pn,nx,ny)
            Den = (Px_map + Pn_map)
            ind = np.where(Den == 0)
            Den[ind] = eta
            Wfc = Px_map / Den
        else:
            Wfc = Px_map / (Px_map + eta)
        #info(Esn,name='Esn')
        #info(mask,name='mask')

        #    writefits("xx_fft_wfc.fits", Wfc)
        #    t = gamma1 + 1j*gamma2
        #    z = np.fft.fftshift(np.fft.fft2(t))
        #    z1 = z*conj(z)
        #    writefits("xx_fft_k.fits", real(z1))
        if Inpaint:
            lmin = 0
            lmax = self.get_lmax_dct_inpaint(gamma1, gamma2)
        #print("lmax = ", lmax)
        
        
        if PropagateNoise is not None:
            n1,n2 = PropagateNoise.get_shear_noise()
            n1= n1 * mask
            n2 =n2 * mask
            gamma1 = n1
            gamma2 = n2
            
    
        for n in range(niter):
            xn = np.copy(xg)
            t1,t2 = self.get_resi(xg, gamma1, gamma2, Esn)
            # print("T1     Sigma = ", np.std(t1), ", Max = ", np.max(t1))

            t = xg + (t1 + 1j*t2)                                             # xg + H^T(eta / Sn * (y- H * xg))
            xg = self.mult_wiener(t,Wfc) # wiener filtering in fourier space
            if Inpaint:
                xg = self.step_dct_inpaint(xg, xn, mask, n, niter, lmin, lmax) 

                # print("     Sigma = ", np.std(xg), ", Max = ", np.max(xg))
            # info(xg.real,name="XGR=>")
               
            if self.Verbose:
                if  ktr is not None:
                    print("   it. Wiener Iter ", n+1, ", Err = ", LA.norm((xg.real - ktr)*mask) / LA.norm(ktr*mask) * 100.)
                else:
                    print("   Wiener rec Iter: ", n+1, ", std ke =  %5.4f" %  (np.std(xg[ind]/ tau)))

 #       xg.real = M.inpaint(xg.real, mask, 50)
 #          if ktr is not None:
 #              print("Iter ", n+1, ", Err = ", LA.norm(xg.real - ktr) / LA.norm(ktr) * 100.)
        return xg.real, xg.imag


    def sparse_wiener_filtering(self, InshearData, PowSpecSignal, niter=None, Nsigma=None, Inpaint=False, InpNiter=20, OnlyPos=True, FirstDetectScale=DEF_FirstDetectScale, Bmode=True, ktr=None, PropagateNoise=False):
        """
        MCAlens algorithm; Estimate the complex EB mode. The solution is assumed to have 
        a Gaussian component and a sparse Component.
        The algorithm estimate these two parts, for both the E and B mode.
        It returns 4 maps, estimated E and B mode and sparse E and B mode.
        The Gaussian E and B modes are obtained taking the difference between
        the estimated mode and sparse mode.
        Parameters
        ----------
        InshearData : Shear Class
            Class contains the shear information.
        PowSpecSignal : 1D np.ndarray
             Theorical Signal power spectrum.
        niter : int, optional
            number of iterations. Default is DEF_niter
        Nsigma : float, optional
            Detection level on wavelet coefficients. The default is self.DEF_Nsigma.
        Inpaint : Bool, optional
            if true, inpainting the missing data. Default is false.
        InpNiter : int, optional
            Number of iterationd in inpainting algorithm. The default is 20.
        OnlyPos : Bool, optional
            Only positive wavelet coefficients are detected. The default is True.
        FirstDetectScale : TYPE, optional
             No wavelet coefficient are detected in the finest wavelet scales. 
        Bmode : Bool, optional
            Calculate also the B-mode. The default is True.
        ktr : 2D np.ndarray, optional
            true convergence map, known in case simulated data are used.
            if given, errors are calculated at each iteration.
        PropagateNoise : Bool, optional
            if True, run the routine on a noise realization instead of the input shear field.
        Returns
        -------
        2D np.ndarray
              E reconstructed mode. Convergence = E
        2D np.ndarray
              B reconstructed mode.  
        2D np.ndarray
              E reconstructed mode of the sparse component. Convergence = E
        2D np.ndarray
              B reconstructed mode  of the sparse component. 
        """
        
        gamma1 = InshearData.g1
        gamma2 = InshearData.g2        
        nx = self.nx
        ny = self.ny
        xg = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        mask = InshearData.mask
 
        if niter is None:
            niter = self.DEF_niter
        if Nsigma is None:
            Nsigma = self.DEF_Nsigma

        RMS_ShearMap =  np.sqrt(InshearData.Ncov / 2.)
        Ncv = InshearData.Ncov / 2.
        xg = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        xs = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        xw = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        xt = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        rec = np.zeros((nx,ny))
        reci = np.zeros((nx,ny))
        SigmaNoise = np.min(RMS_ShearMap)
        Esn_Sparse = SigmaNoise / RMS_ShearMap
        Esn_Sparse[Esn_Sparse == np.inf] = 0

    
        # find the minimum noise variance
        tau = np.min(Ncv)
        # set the step size
        # eta = 1.83 * tau
        eta = tau
        # compute signal coefficient
        Esn = eta / Ncv
    
        # calculate the wiener filter coefficients
        Px_map = get_ima_spectrum_map(PowSpecSignal,nx,ny)
        Wfc = Px_map / (Px_map + eta)
        Wfc[Wfc == np.inf] = 0

        #    writefits("xx_fft_wfc.fits", Wfc)
        #    t = gamma1 + 1j*gamma2
        #    z = np.fft.fftshift(np.fft.fft2(t))
        #    z1 = z*conj(z)
        #    writefits("xx_fft_k.fits", real(z1))

            
        ind_maskOK = np.where(mask == 1)
        ind_maskZero = np.where(mask == 0)
        
        #xg1,xg2 = self.prox_wiener_filtering(gamma1, gamma2, PowSpecSignal, InshearData.Ncov, Pn=None, niter=10, Inpaint=True, ktr=None)              

        
        # Detection of the significant wavelet coefficents.
        # to avoid border artefacts, we first make a rough very smooth estimate
        # using a smoothed KS, compute the residual and the residul to the
        # rough estimate. This avoid the detection of many wavelet coeff along
        # the border.

        if Inpaint:
            lmin = 0
            resi1,resi2 =  self.get_resi(xg, gamma1, gamma2, Esn) 
            lmax = self.get_lmax_dct_inpaint(resi1,resi2)

        ks =  self.gamma_to_cf_kappa(gamma1, gamma2) 
        rec[:,:] = ks.real
        ks = self.smooth(rec, sigma=15)
        resi1,resi2 =  self.get_resi(ks, gamma1, gamma2, Esn_Sparse) 

        resi1 += ks
        self.WT.transform(resi1)
        self.WT_ActiveCoef = self.get_active_wt_coef(InshearData, OnlyPos=OnlyPos, UseRea=False, SigmaNoise=SigmaNoise, Nsigma=Nsigma,ComputeWTCoef=False)                
        self.WT_ActiveCoef[self.WT.ns-1,:,:] = 0 
        
        # Replace the shear measurements by noise realisations
        if PropagateNoise is True:
            n1,n2 = InshearData.get_shear_noise()
            n1= n1 * mask
            n2 =n2 * mask
            gamma1 = n1
            gamma2 = n2

        for n in range(niter):
            resi1,resi2 =  self.get_resi(xg, gamma1, gamma2, Esn_Sparse) # , mask=mask, niter=20)
            
            # sparse component
            xt[:,:] = (resi1 + 1j*resi2)                                             # xg + H^T(eta / Sn * (y- H * xg))
            self.WT.transform(xt.real)
            self.WT.coef *= self.WT_ActiveCoef 
            #self.WT.threshold(SigmaNoise=SigmaNoise, Nsigma=Nsigma, ThresCoarse=True, hard=True, FirstDetectScale=FirstDetectScale,Verbose=False)
            #if OnlyPos:
            #    ind = np.where(self.WT.coef < 0)
            #    self.WT.coef[ind]=0
            signif_resi = self.WT.recons() 
            #if n % 10 == 0:
            #    info(signif_resi)
            rec[:,:] = xs.real + signif_resi
            
            #if n % 10 == 0:
            #    print(n, "max = ", np.max(rec))
            
            #if n % 100 == 0:
            #    tvilut(rec,title='S_f'+str(n))
            if PropagateNoise is False:
                if OnlyPos:
                    ind = np.where(rec < 0)
                    rec[ind] = 0
            if Bmode:
                self.WT.transform(xs.imag)
                self.WT.threshold(SigmaNoise=SigmaNoise, Nsigma=Nsigma, ThresCoarse=True, hard=True, FirstDetectScale=FirstDetectScale,Verbose=False)
                reci[:,:] = self.WT.recons()
            else:
                reci[:,:] = 0
            xs = rec + 1j * reci
            xg[:,:] = xw + xs

                # ind_maskOK
                # xw[:,:] = inp_x[:,:]
            # Wiener component
            # calculate the residual
            # xs =0
            xn = np.copy(xw)

            InpMethod1=1
            if InpMethod1:
                nw = 1
            else:
                nw = 1
            for i in range(nw): 
                if Inpaint and InpMethod1:
                        t1,t2 =  self.get_resi(xg, gamma1, gamma2, Esn, mask=mask, niter=InpNiter, dctmax=lmax)
                else:
                    t1,t2 =  self.get_resi(xg, gamma1, gamma2, Esn)
                xt[:,:] = (t1 + 1j*t2)       
                xt += xw                                            # xg + H^T(eta / Sn * (y- H * xg))
                xw[:,:] = self.mult_wiener(xt,Wfc) # wiener filtering in fourier space
                xg[:,:] = xw + xs

            xg[:,:] = xw + xs
            
            if self.Verbose:
                ind = np.where(mask == 1)
                if ktr is not None: 
                    print("   Sparse rec Iter: ", n+1, ", Err = %5.4f" % (np.std((xg.real[ind] - ktr[ind])) / np.std(ktr[ind]) * 100.), ", Resi ke (x100) =  %5.4f" % (np.std(resi1[ind])*100.), ", Resi kb (x100) =  %5.4f" %  (np.std(resi2[ind])*100.))
                else:  
                    print("   Sparse rec Iter: ", n+1, ", Resi ke =  %5.4f" %  (np.std(resi1[ind]/ tau)), ", Resi kb = %5.4f" %  (np.std(resi2[ind])/tau))
        #endfor

        return xg.real, xg.imag, xs.real, xs.imag

    def step_wt_recons(self, xg):
        self.WT.transform(xg.real)
        self.WT.coef *= self.WT_ActiveCoef
        xg.real = self.WT.recons()
        return xg

    def sparse_recons(self, InshearData, UseNoiseRea=False, WT_Support=None, WT_Sigma=None, niter=None, Nsigma=None, ThresCoarse=False, Inpaint=False, ktr=None, FirstDetectScale=None, Nrea=None, hard=True, FirstGuess=None):
        """
        Reconstruction of the convergence field using sparsity. The detection levels 
         
        convergence map with an absolute value large than Nsigma * NoiseStandardDeviation.
        It returns a cube  A[s,i,j] containing 0 or 1. 
        If A[s,i,j] == 1 then we consider we have a detection at scale s and position (i,j).

        Parameters
        ----------
        InshearData : Class  shear_data
            Input Class describing the obervations.

        Parameters
        ----------
        InshearData : TYPE
            DESCRIPTION.
        WT_Support : 3D np.ndarray, optional
            This variable is not used anymore.
            We keep it for back compatibility,
            The default is None.
        WT_Sigma : 3D np.ndarray, optional
            Noise standard deviation on each wavelet coefficient. The default is None
            and it is calculated in the routine.
        niter : int, optional
            number of iterations. Default is DEF_niter
        Nsigma : float, optional
            Detection level on wavelet coefficients. The default is self.DEF_Nsigma.
        ThresCoarse : Bool, optional
            If True the coarsest wavelet scale is removed. The default is False.
        Inpaint : Bool, optional
            if true, inpainting the missing data. Default is False.
        ktr : 2D np.ndarray, optional
            true convergence map, known in case simulated data are used.
            Errors are calculated at each iteration.  Default is None.
        FirstDetectScale : int, optional
             No wavelet coefficient are detected in the finest wavelet scales. 
             The default is DEF_FirstDetectScale.
        Nrea : int, optional
            Generate Nrea noise realizations to estimate the noise level on
            each wavelet coefficient. The default is DEF_Nrea.

        Returns
        -------
        2D np.ndarray
              E reconstructed mode. Convergence = E
        2D np.ndarray
              B reconstructed mode.  
        """
        if niter is None:
            niter = self.DEF_niter
        if Nsigma is None:
            Nsig = self.DEF_Nsigma
        else:
            Nsig=Nsigma
        if not hard:
            Nsig = Nsig / 2.  # for soft thresholding, the thresholding must
                                # be small than for hard thresholding
        if FirstDetectScale is None:
            FirstDetectScale=self.DEF_FirstDetectScale
        if Nrea is None:
            Nrea=self.DEF_Nrea
        (nx,ny) = InshearData.g1.shape
  
        if UseNoiseRea:
            if WT_Sigma is None:
                self.WT_Sigma = self.get_wt_noise_level(InshearData,Nrea=Nrea)
            else:
                self.WT_Sigma = WT_Sigma
#            info(self.WT_Sigma, name='WT_Sigma2')
#            if WT_Support is None:
#                self.WT_ActiveCoef = self.get_active_wt_coef(InshearData,  UseRea=True, SigmaNoise=1., Nsigma=Nsigma, Nrea=Nrea, WT_Sigma=WT_Sigma)
#            else:
#                self.WT_ActiveCoef = WT_Support
            WeightResi = InshearData.mask
        else:
            RMS_ShearMap =  np.sqrt(InshearData.Ncov / 2.)
            tau = np.min(RMS_ShearMap)
            self.SigmaNoise = tau
            
            ind = np.where(RMS_ShearMap > 0)
            WeightResi = np.zeros((nx,ny))
            WeightResi[ind] = tau / RMS_ShearMap[ind]

            self.WT_Sigma = tau
        gamma1 = InshearData.g1
        gamma2 = InshearData.g2
        mask = InshearData.mask
        self.nx=nx
        self.ny=ny
        
        Iz = 1j * np.zeros((nx,ny))
        xg = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        if FirstGuess is not None:
            if FirstGuess.dtype == 'float64':
                xg = FirstGuess + Iz
            elif FirstGuess.dtype == 'complex128':
                xg = FirstGuess
            else:
                print("Warning: first guess must be float64 ou complex128. It is not used.")
 
        if self.Verbose:
            if UseNoiseRea:
                print("Sparse rec. with noise rea: ", nx, ny, ", Niter = ", niter, ", Nsigma = ", Nsigma, ", Nrea = ", Nrea, "Inpaint = ", Inpaint, "Hard = ", hard)
 #           for j in range(self.WT.ns):
 #               print("   Scale ", j+1, ": Numner of active coeffs = ", (self.WT_ActiveCoef[j,:,:]).sum(), ", Nbr (%) = ", (self.WT_ActiveCoef[j,:,:]).sum() / (nx*ny)*100.)

        # Initialisation for  inpainting
        WT_Inpaint = 0
        DCT_Inpaint= 0
        if Inpaint:
            DCT_Inpaint = 1  # we will use on WT inpainting, 
                            # DCT inpating is availabble if we 
                            # replace WT_Inpaint = 1  by DCT_Inpaint = 1
        if DCT_Inpaint:
            lmin = 0
            lmax = self.get_lmax_dct_inpaint(gamma1, gamma2)
        if WT_Inpaint:
            lmin = 0
            resi1,resi2 =  self.get_resi(xg, gamma1, gamma2, WeightResi)
            self.WT.transform(resi1)
            lmax = np.max(np.abs(self.WT.coef[0:self.WT.ns-1, :, :] ))
    
        # Main iteration
        Verbose=self.Verbose
        # info(WeightResi, name='Esn')
        rec = np.zeros((nx,ny))
        reci = np.zeros((nx,ny))
        for n in range(niter):
            xn = np.copy(xg)
            # print("XG=", xg.shape)
            resi1,resi2 =  self.get_resi(xg, gamma1, gamma2, WeightResi)
            xg += (resi1 + 1j*resi2)
            # print("   BEF  0.87?Sparse rec Iter: ", n+1, ", Sol =  %5.4f" %  LA.norm(xg.real))
            
            self.WT.transform(xg.real)
            self.WT.threshold(SigmaNoise=self.WT_Sigma, Nsigma=Nsig, ThresCoarse=False, hard=hard, FirstDetectScale=FirstDetectScale,Verbose=False)
            rec[:,:] = self.WT.recons()
            
            if WT_Inpaint:
                lval = lmin + (lmax - lmin) * (1 - erf(2.8 * n / niter))  # exp decay
                self.WT.threshold(SigmaNoise=1., Nsigma=lval, hard=False, KillCoarse=False)
                inp = self.WT.recons()
                rec[:,:] = (1-mask)*inp + mask * xg.real
            xg[:,:] = rec + Iz
            # xg = self.step_wt_recons(t)

            if DCT_Inpaint:
                rec[:,:] = self.step_dct_inpaint(xg, xn, mask, n, niter, lmin, lmax) 
                rec[:,:] = (1-mask)*rec + mask * xg.real
                xg[:,:] = rec + Iz
            
            if Verbose:
                if ktr is not None: 
                    print("   Sparse rec Iter: ", n+1, ", Err = %5.4f" % (LA.norm((xg.real - ktr)*mask) / LA.norm(ktr*mask) * 100.), ", Resi ke (x100) =  %5.4f" % (LA.norm(resi1*mask)*100.), ", Resi kb (x100) =  %5.4f" %  (LA.norm(resi2*mask)*100.))
                else:
                    print("   Sparse rec Iter: ", n+1, ", Sol =  %5.4f" %  LA.norm(xg.real))
 
        # debiaising sterp
        if self.niter_debias > 0:
            xn = np.copy(xg)
            resi1,resi2 =  self.get_resi(xg, gamma1, gamma2, WeightResi)
            xn += (resi1 + 1j*resi2)
            self.WT.transform(xn.real)
            # determine the set of active coefficients
            if UseNoiseRea:
                self.WT_ActiveCoef = self.get_active_wt_coef(InshearData, WT_Sigma=self.WT_Sigma, UseRea=UseNoiseRea, SigmaNoise=1., Nsigma=Nsigma,ComputeWTCoef=False)
            else:
                self.WT_ActiveCoef = self.get_active_wt_coef(InshearData, UseRea=False, SigmaNoise=tau, Nsigma=Nsigma,ComputeWTCoef=False)                
            for n in range(self.niter_debias):
                resi1,resi2 =  self.get_resi(xg, gamma1, gamma2, mask)
                xg += (resi1 + 1j*resi2)
                self.WT.coef *= self.WT_ActiveCoef
                rec[:,:] = self.WT.recons()  
                xg[:,:] = rec + Iz
        return xg.real, xg.imag


    def sparse_recons_covmat(self, gamma1, gamma2, NcvIn, niter=None, Nsigma=None, ThresCoarse=False, Inpaint=False, ktr=None, FirstDetectScale=DEF_FirstDetectScale, Bmode=True, FirstGuess=None):
        """
        This routine should not be used anymore.
        It is equivalent to self.sparse_recons using the parameter UseNoiseRea=False
        """
        
        if niter is None:
            niter = self.DEF_niter
        if Nsigma is None:
            Nsigma = self.DEF_Nsigma

        RMS_ShearMap =  np.sqrt(NcvIn / 2.)
        (nx,ny) = gamma1.shape
        self.nx=nx
        self.ny=ny
        
        xg = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        Iz = 1j * np.zeros((nx,ny))
        if FirstGuess is not None:
            if FirstGuess.dtype == 'float64':
                xg = FirstGuess + Iz
            elif FirstGuess.dtype == 'complex128':
                xg = FirstGuess
            else:
                print("Warning: first guess must be float64 ou complex128. It is not used.")
 

        index = np.where(NcvIn<1e2)
        mask = np.zeros((nx,ny))
        mask[index] = 1

        # find the minimum noise variance
        tau = np.min(RMS_ShearMap)
        # tau =1.
        SigmaNoise = tau
        # compute signal coefficient
        Esn = tau / RMS_ShearMap
        # print("size ESN ", vsize(Esn))

        if self.Verbose:
            print("Sparse (l_0) Rec. with covmat: ", nx, ny, ", Niter = ", niter, ", Nsigma = ", Nsigma, "Inpaint = ", Inpaint)
 
        Verbose=self.Verbose
        if Inpaint:
            lmin = 0
            lmax = self.get_lmax_dct_inpaint(gamma1, gamma2)
        rec = np.zeros((nx,ny))
        reci = np.zeros((nx,ny)) 
        for n in range(niter):
            xn = np.copy(xg)
            resi1,resi2 =  self.get_resi(xg, gamma1, gamma2, Esn)
            xg += (resi1 + 1j*resi2)
            self.WT.transform(resi1)
            self.WT.threshold(SigmaNoise=SigmaNoise, Nsigma=Nsigma, ThresCoarse=ThresCoarse, hard=True, FirstDetectScale=FirstDetectScale,Verbose=False)
            rec[:,:] = self.WT.recons()
            if Bmode:
                self.WT.transform(xg.imag)
                self.WT.threshold(SigmaNoise=SigmaNoise, Nsigma=Nsigma, ThresCoarse=ThresCoarse, hard=True, FirstDetectScale=FirstDetectScale,Verbose=False)
                reci[:,:] = self.WT.recons()
            else:
                reci[:,:] = 0
            xg[:,:] = rec + 1j * reci

            if Inpaint:
                xg[:,:] = self.step_dct_inpaint(xg, xn, mask, n, niter, lmin, lmax, InpaintAlsoImag=Bmode)                 

            if Verbose:
                ind = np.where(mask == 1)
                if ktr is not None: 
                    print("   Sparse rec Iter: ", n+1, ", Err = %5.4f" % (np.std((xg.real[ind] - ktr[ind])) / np.std(ktr[ind]) * 100.), ", Resi ke (x100) =  %5.4f" % (np.std(resi1[ind])*100.), ", Resi kb (x100) =  %5.4f" %  (np.std(resi2[ind])*100.))
                else:  
                    print("   Sparse rec Iter: ", n+1, ", Resi ke =  %5.4f" %  (np.std(resi1[ind]/ tau)), ", Resi kb = %5.4f" %  (np.std(resi2[ind])/tau))
        
        return xg.real, xg.imag


    def rea_sparse_wiener_filtering(self, InshearData, PowSpecSignal, WT_Support=None, niter=None, WT_Sigma=None, Nsigma=DEF_Nsigma, Inpaint=False, FirstDetectScale=DEF_FirstDetectScale, Nrea=DEF_Nrea, ktr=None):
        """
        This routine should not be used. It is a test routine 
        """
            
        if niter is None:
            niter = self.DEF_niter
        if Nsigma is None:
            Nsigma = self.DEF_Nsigma

        if WT_Support is None:
            WT_Support = self.get_active_wt_coef(InshearData, UseRea=True, SigmaNoise=1., Nsigma=Nsigma, Nrea=Nrea, WT_Sigma=WT_Sigma)
        if FirstDetectScale > 0:
            WT_Support[0:FirstDetectScale,:,:] = 0        
        WT_Support[0:FirstDetectScale,:,:] = 0        
        WT_Support[self.WT.ns-1,:,:] = 0    
        gamma1 = InshearData.g1
        gamma2 = InshearData.g2      
        (nx,ny) = gamma1.shape
        mask = InshearData.mask
        self.nx=nx
        self.ny=ny       
        xg = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        mask = InshearData.mask
        if FirstDetectScale > 0:
            WT_Support[0:FirstDetectScale] = 0

        Ncv = InshearData.Ncov / 2.
        xg = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        xs = np.zeros((nx,ny)) + 1j * np.zeros((nx,ny))
        # find the minimum noise variance
        tau = np.min(Ncv)
    
        # set the step size
        # eta = 1.83 * tau
        eta = tau

        # compute signal coefficient
        Esn = eta / Ncv
    
        # calculate the wiener filter coefficients
        Px_map = get_ima_spectrum_map(PowSpecSignal,nx,ny)

        Wfc = Px_map / (Px_map + eta)
        #    writefits("xx_fft_wfc.fits", Wfc)
        #    t = gamma1 + 1j*gamma2
        #    z = np.fft.fftshift(np.fft.fft2(t))
        #    z1 = z*conj(z)
        #    writefits("xx_fft_k.fits", real(z1))
        if Inpaint:
            lmin = 0
            lmax = self.get_lmax_dct_inpaint(gamma1, gamma2)
            
        xs = xg
        nw = 1
        for n in range(niter):
            xn = np.copy(xg)
            resi1,resi2 =  self.get_resi(xg, gamma1, gamma2, mask)
            # sparse component
            xs = (resi1 + 1j*resi2)                                             # xg + H^T(eta / Sn * (y- H * xg))
            self.WT.transform(xs.real)
            self.WT.coef *= WT_Support
            xs.real = self.WT.recons() 
            xs.imag[:,:] = 0
            
            # Winer component
            # calculate the residual
            xg = xg - xs
            for i in 0, range(nw):
                t1,t2 = self.H_operator_eb2g(xg.real+xs.real, xg.imag+xs.imag)        # H * xg
                t1,t2 = self.H_adjoint_g2eb(Esn*(gamma1 - t1), Esn*(gamma2 - t2))           # H^T(eta / Sn * (y- H * xg))
                t = xg + (t1 + 1j*t2)                                             # xg + H^T(eta / Sn * (y- H * xg))
                xg = self.mult_wiener(t,Wfc) # wiener filtering in fourier space                

            xg = xg + xs
            if Inpaint:
                xg = self.step_dct_inpaint(xg, xn, mask, n, niter, lmin, lmax)                 
    
                # xg.real = (1.-mask)*rec + mask * xg.real
            if self.Verbose:
                ind = np.where(mask == 1)
                if ktr is not None: 
                    print("   Sparse rec Iter: ", n+1, ", Err = %5.4f" % (np.std((xg.real[ind] - ktr[ind])) / np.std(ktr[ind]) * 100.), ", Resi ke (x100) =  %5.4f" % (np.std(resi1[ind])*100.), ", Resi kb (x100) =  %5.4f" %  (np.std(resi2[ind])*100.))
                else:  
                    print("   Sparse rec Iter: ", n+1, ", Resi ke =  %5.4f" %  (np.std(resi1[ind]/ tau)), ", Resi kb = %5.4f" %  (np.std(resi2[ind])/tau))

        # k = M.inpaint(Res.ikw, d.mask, 50)
 #          if ktr is not None:
 #              print("Iter ", n+1, ", Err = ", LA.norm(xg.real - ktr) / LA.norm(ktr) * 100.)
        return xg.real, xs.real

     

############ END CLASS #######################

if __name__ == '__main__':
    print ( "Main :)")
 
 

