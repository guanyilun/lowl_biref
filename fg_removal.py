#%%
import os, os.path as op, argparse
import numpy as np
import healpy as hp
from scipy.optimize import minimize
from orphics import cosmology as cosmo
from cosmoslib.utils import cwignerd

import files, lib
from gen_sims import generate_cls

NSIDE = 16
arcmin = np.deg2rad(1/60)

class SignalCov:
    def __init__(self, nside=16, lmax=None, fwhm=0, cl=None, Tcmb=2.72548*1e6):
        """
        Parameters
        ----------
        cl: dict with keys like "EE", "TT", etc.
        lmax: maximum ell to sum
        nside: nside of the covariance
        fwhm: optionally apply a beam with given fwhm in arcmin 

        Returns
        -------
        covmat with shape(2, 2, npix x npix)

        """
        if lmax is None: lmax = 3*nside-1
        # get cl if not given
        if cl is None:
            print("Warning: no cl provided, using default theory in orphics.cosmology")
            ell = np.arange(lmax+1)
            cl = cosmo.default_theory()
            EE = cl.lCl('ee', ell)
            BB = cl.lCl('bb', ell)
        else:
            if Tcmb != 1: print(f"Warning: assumed input cl is dimensionless, applying Tcmb^2={Tcmb}^2 to it")
            lmax = len(cl['ee'])-1
            ell = np.arange(lmax+1)
            EE = cl['ee']*Tcmb**2
            BB = cl['bb']*Tcmb**2

        # apply beam if needed
        if fwhm > 0:
            print("Applying beam with fwhm={} arcmin".format(fwhm))
            bl = hp.gauss_beam(fwhm=fwhm*arcmin, lmax=lmax)
            EE *= bl**2 
            BB *= bl**2 

        # get cosine for gauss legende quadrature
        # using pairwise dot products
        npix = hp.nside2npix(nside)
        x,y,z = hp.pix2vec(nside, np.arange(npix))
        cos = np.outer(x,x) + np.outer(y,y) + np.outer(z,z)
        # make sure cosine values are well behaved
        cos = np.clip(cos, -1, 1)
        # turn into 1d vector
        flat_cos = np.ravel(cos)

        # A = \sum (2l+1)/4\pi*(EE+BB)/2 d_{ 22}^l(\mu)
        # B = \sum (2l+1)/4\pi*(EE-BB)/2 d_{-22}^l(\mu)
        print("Computing covariance matrix elements...")
        A = cwignerd.wignerd_cf_from_cl( 2,2,1,len(flat_cos),lmax,flat_cos,(2*ell+1)/(4*np.pi)*(EE+BB)/2)
        B = cwignerd.wignerd_cf_from_cl(-2,2,1,len(flat_cos),lmax,flat_cos,(2*ell+1)/(4*np.pi)*(EE-BB)/2)
        A = A.reshape(cos.shape)  # npix x npix
        B = B.reshape(cos.shape)

        # covariance assuming no rotation; rotation is done later in .calc
        self.cov = np.array([[A+B, 0*A],[0*A, A-B]])  # allocate some memory here to avoid reallocation if needed
        self.A = A
        self.B = B

    def calc(self, alpha=0, inplace=True):
        """Calculate signal covariance matrix (with off-diagonal)

        Parameters
        ----------
        alpha: rotation angle in arcmin

        Returns
        -------
        cov: noise covariance matrix (2, 2, npix, npix)

        """
        if alpha==0:  return self.cov
        if not inplace: cov = np.zeros_like(self.cov)
        else: cov = self.cov
        cov[0,0,...] = self.A + self.B*np.cos(4*alpha)
        cov[1,1,...] = self.A - self.B*np.cos(4*alpha)
        cov[0,1,...] = self.B*np.sin(4*alpha)
        cov[1,0,...] = self.B*np.sin(4*alpha)
        return cov


class NoiseCov:
    def __init__(self, nlev=1, fwhm=0, nside=16, lmax=None, nl=None, regularizer=0.2):
        """Noise covariance matrix: (N_1 + N_2) / (1-c_1-c_2)^2

        Parameters
        ----------
        nlev: tt noise level in uK arcmin
        fwhm: in arcmin
        lmax: maximum ell to sum
        nside: nside of the map
        nl: noise power spectrum (optional)
        regularizer: a small noise added to the diagonal to avoid singularity (in uK arcmin)


        """
        if lmax is None: lmax = 3*nside-1
        if nl is None:
            if fwhm > 0:  bl = hp.gauss_beam(fwhm=fwhm*arcmin, lmax=lmax)
            else: bl = np.ones(lmax+1)
            nl = 2*(nlev*arcmin)**2*bl**2  # factor 2 from T to P
        else:
            lmax = len(nl)-1

        # compute cosine for gauss legendre quadrature
        npix = hp.nside2npix(nside)
        # pairwise dot products (n \cdot n')
        x,y,z = hp.pix2vec(nside, np.arange(npix))
        cos = np.outer(x,x) + np.outer(y,y) + np.outer(z,z)
        # make sure cosine values are well behaved
        cos = np.clip(cos, -1, 1)
        flat_cos = np.ravel(cos)

        # with smoothing, the covariance matrix is not diagonal anymore,
        # we calculate it similar to covariance matrix of the signal. Now
        # NlEE = NlBB -> A \neq 0, B=0.
        print("Computing covariance matrix elements...")
        ell = np.arange(0, lmax+1)
        A = cwignerd.wignerd_cf_from_cl(2,2,1,len(flat_cos),lmax,flat_cos,(2*ell+1)/(4*np.pi)*nl)
        A = A.reshape(cos.shape)

        # add a small regularizer to avoid singularity
        if regularizer>0:  # TODO: check this is correct
            print(f"Add a regularizer with {regularizer} uK arcmin to the diagonal to avoid singularity")
            # add to the diagonal only: (the fast way)
            diag = np.einsum('ii->i', A)  # in-place memory slicing
            diag += (regularizer*arcmin)**2  # in-place addition to the diagonal part of A
        self.A = A
        self.cov = np.array([[A, 0*A],[0*A, A]])

    def calc(self, c=[0,0], inplace=True):
        """Calculate noise covariance

        Parameters
        ----------
        c: list of coefficients for each template

        Returns
        -------
        cov: noise covariance matrix (2, 2, npix, npix)

        """
        if inplace: cov = self.cov
        else: cov = np.zeros_like(self.cov)
        cov[0,0,...] = self.A/(1-c[0]-c[1])**2
        cov[1,1,...] = cov[0,0,...]
        return cov


class TotalCov:
    def __init__(self, nside=16, lmax=200, nlev=1, fwhm_s=0, fwhm=9.16*60, nl=None, regularizer=0.2):
        """Total covariance matrix with both signal and noise covariance

        Parameters
        ----------
        nside: nside of the map
        lmax: maximum ell to sum
        nlev: tt noise level in uK arcmin
        fwhm: in arcmin
        nl: noise power spectrum (optional)
        regularizer: a small noise added to the diagonal to avoid singularity (in uK arcmin)

        Returns
        -------
        covmat with shape(2, 2, npix, npix)

        """
        # get power spectra
        cls = generate_cls(lmax)
        # get covariance for both reionization bump (low ell) and recombination (high ell)
        # for beam see the note below
        fwhm_signal = (fwhm_s**2 + fwhm**2)**0.5
        print("Initializing signal covariance matrix for reionization bump...")
        self.cov_rei = SignalCov(nside=nside, cl=cls['rb'],    fwhm=fwhm_signal)
        print("Initializing signal covariance matrix for recombination...")
        self.cov_rec = SignalCov(nside=nside, cl=cls['no_rb'], fwhm=fwhm_signal)
        # note: signal has an additional beam that's channel-dependent, 
        # while noise only has the smoothing beam applied during preprocessing. 
        # In practice fwhm_s is much smaller than fwhm (effect on 
        # a few parts in 10^3.), so it's fine to assume fwhm_s is 0. 
        print("Initializing noise covariance matrix...")
        self.noise = NoiseCov(nside=nside, nlev=nlev, fwhm=fwhm, lmax=lmax, nl=nl, 
                              regularizer=regularizer)

        # allocate some memory here to avoid reallocation if needed
        self.cov = np.zeros_like(self.cov_rei.cov)

    def calc(self, thetas, c, inplace=True, small_angle=True):
        """Calculate total covariance matrix
        
        Parameters
        ----------
        thetas: [theta_rei, theta_rec]
        c: list of coefficients for each template
        inplace: if True, return the covariance matrix inplace
        small_angle: if True, use small angle approximation

        Returns
        -------
        cov: total covariance matrix (2, 2, npix, npix)

        """
        theta_rei, theta_rec = thetas
        if inplace: cov = self.cov
        else: cov = np.zeros_like(self.cov)
        # use small angle approximation if wanted
        if small_angle:
            diag = self.cov_rei.A + self.cov_rec.A
            off_diag = 4*(self.cov_rei.B*theta_rei + self.cov_rec.B*theta_rec)
            cov[0,0,...] = diag
            cov[1,1,...] = diag
            cov[0,1,...] = off_diag
            cov[1,0,...] = off_diag
        else:
            raise NotImplementedError
        # add noise covariance
        cov += self.noise.calc(c)
        return cov


def inv_cov(covmat):
    """Invert covariance matrix
    
    Parameters
    ----------
    covmat: covariance matrix with shape (2, 2, npix, npix)
    
    Returns
    -------
    icovmat: inverse covariance matrix with shape (2, 2, npix, npix)

    """
    # block matrix inversion following https://en.wikipedia.org/wiki/Block_matrix
    A = covmat[0,0]
    B = covmat[0,1]
    C = covmat[1,0]
    D = covmat[1,1]

    iA = np.linalg.inv(A)
    iE = np.linalg.inv(D - C@iA@B)
    icov = np.array([[iA + iA@B@iE@C@iA, -iA@B@iE],
            [-iE@C@iA, iE]])
    return icov

def slogdet_cov(covmat):
    """Calculate the determinant of covariance matrix which is a block matrix
    
    Parameters
    ---------- 
    covmat: covariance matrix with shape (2,2,npix,npix)
    
    Following https://en.wikipedia.org/wiki/Block_matrix
    
    Returns
    -------
    sign: sign of the determinant
    logdet: log of the determinant

    """
    A = covmat[0,0]
    B = covmat[0,1]
    # special case with A=D, B=C
    # return np.linalg.det(A-B)*np.linalg.det(A+B)
    sign1, logdet1 = np.linalg.slogdet(A-B)
    sign2, logdet2 = np.linalg.slogdet(A+B)
    return sign1*sign2, logdet1+logdet2

def preprocess(imaps):
    omaps = []
    for i in range(len(imaps)):
        # smooth target maps by 9.16 deg
        imap = hp.smoothing(imap[i], fwhm=np.deg2rad(9.16))
        # downgrade to nside=16
        imap = hp.ud_grade(imap, nside_out=NSIDE)
        omaps.append(imap)
    return omaps

# covmat = TotalCov(nside=NSIDE, nlev=nlevs[i], fwhm=fwhms[i])
def build_likelihood(imaps, templates, covmats):
    """Build likelihood function based on the given templates and covariance matrices. 
    Assume two templates, but easily generalizable to more."""
    t1, t2 = templates
    def loglike(theta1, theta2, c1, c2):
        chi2 = 0
        for i in range(len(imaps)):
            imap = imaps[i]
            covmat = covmats[i]
            # compute x; x has shape (2, npix)
            x  = imap - t1*c1 - t2*c2
            x /= (1-c1-c2)
            # convert to 1d array
            cov = covmat.calc([theta1, theta2], [c1, c2])
            # compute inverse; icov has shape (2, 2, npix, npix)
            icov = inv_cov(cov)
            # compute chi-square: x' cov^-1 x
            chi2 += np.einsum('ik, ijkl, jl', x, icov, x)
            # add determinant part
            chi2 += slogdet_cov(cov)[1]
        return -0.5*chi2
    return loglike

def fg_removal(imaps, templates, covmats, **kwargs):
    """Remove foregrounds in a series of maps. This assumes that input maps
    and templates are preprocessed properly. The processing assumed here includes
    a smoothing of 9.16 deg and downgrade to nside=16.
    
    Parameters
    ----------
    imaps: list CMB maps to be fg-cleaned, with shape (nfreq, {Q,U}, npix)
    templates: list of template maps with shape (ntemplates, {Q,U}, npix)
    covmats: list of covariance matrice objects
    
    Returns
    -------
    imaps: dict
        maps with foregrounds removed

    """
    loglike = build_likelihood(imaps, templates, covmats)
    res = minimize(loglike, **kwargs)
    return res

#%%
noise = NoiseCov(nside=16, lmax=200, regularizer=0.2, fwhm=9.16*60)
cov = noise.calc([0.1, 0.1])

# %%
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['figure.dpi'] = 140

plt.imshow(cov[0,0], cmap='gray')
# %%
np.diag(cov[0,0])

# %%
np.linalg.det(cov[0,0])
