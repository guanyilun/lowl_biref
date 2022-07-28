import numpy as np
import healpy as hp
import pysm3.units as u
from cosmoslib.utils import cwignerd
from orphics import cosmology as cosmo
from pixell import utils as u

NSIDE = 16

class SignalCov:
    def __init__(self, lmax=200, nside=16, fwhm=0, cl=None):
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
        # get cl if not given
        if cl is None:
            ell = np.arange(lmax+1)
            cl = cosmo.default_theory()
            EE = cl.lCl('EE', ell)
            BB = cl.lCl('BB', ell)
        else:
            ell = np.arange(lmax+1)
            EE = cl['EE']
            BB = cl['BB']

        # apply beam if needed
        if fwhm > 0:
            bl = hp.gauss_beam(fwhm=fwhm*u.arcmin, lmax=lmax)
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
        A = cwignerd.wignerd_cf_from_cl( 2,2,1,len(flat_cos),lmax,flat_cos,(EE+BB)/2)
        B = cwignerd.wignerd_cf_from_cl(-2,2,1,len(flat_cos),lmax,flat_cos,(EE-BB)/2)
        A = A.reshape(cos.shape)  # npix x npix
        B = B.reshape(cos.shape)

        # covariance assuming no rotation; rotation is done later in .calc
        self.cov = np.array([[A+B, 0*A],[0*A, A-B]])
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
    def __init__(self, nlev=1, fwhm=0, lmax=200, nside=16, nl=None, regularizer=0.2):
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
        if nl is None:
            if fwhm > 0:  bl = hp.gauss_beam(fwhm=fwhm*u.arcmin, lmax=lmax)
            else: bl = np.ones(lmax+1)
            nl = 2*(nlev*u.arcmin)**2*bl**2  # factor 2 from T to P

        # compute cosine for gauss legendre quadrature
        npix = hp.nside2npix(nside)
        # pairwise dot products
        x,y,z = hp.pix2vec(nside, np.arange(npix))
        cos = np.outer(x,x) + np.outer(y,y) + np.outer(z,z)
        # make sure cosine values are well behaved
        cos = np.clip(cos, -1, 1)
        flat_cos = np.ravel(cos)
        # with smoothing, the covariance matrix is not diagonal anymore,
        # we calculate it similar to covariance matrix of the signal. Now
        # NlEE = NlBB -> A \neq 0, B=0.
        A = cwignerd.wignerd_cf_from_cl(2,2,1,len(flat_cos),lmax,flat_cos,nl)
        A = A.reshape(cos.shape)
        self.cov = np.array([[A, 0*A],[0*A, A]])
        # add a small regularizer to avoid singularity
        if regularizer:
            self.cov += np.diag(np.ones(npix)*(regularizer*u.arcmin)**2)  # TODO: check

    def calc(self, c=[0]):
        """Calculate noise covariance

        Parameters
        ----------
        c: list of coefficients for each template

        Returns
        -------
        cov: noise covariance matrix (2, 2, npix, npix)

        """
        return self.cov / (1-np.sum(c))**2


class TotalCov:
    def __init__(self, nside=16, lmax=200, nlev=1, fwhm=9.16*60, nl=None, regularizer=0.2):
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
        self.signal = SignalCov(lmax=lmax, nside=nside, fwhm=fwhm)
        self.noise = NoiseCov(nlev=nlev, fwhm=fwhm, lmax=lmax, nside=nside, nl=nl, regularizer=regularizer)

    def calc(self, alpha, c):
        return self.signal.calc(alpha) + self.noise.calc(c)

def icov(covmat):
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
    def loglike(alpha, cs):
        chi2 = 0
        for i in range(len(imaps)):
            imap = imaps[i]
            covmat = covmats[i]
            c = cs[i]  # template coefficients
            # compute x; x has shape (2, npix)
            x  = imap - np.sum(templates*c[:,None,None], axis=0)
            x /= (1-np.sum(c))
            # convert to 1d array
            cov = covmat.calc(alpha, c)
            # compute inverse; icov has shape (2, 2, npix, npix)
            icov = icov(cov)
            # compute chi-square: x' cov^-1 x
            chi2 += np.einsum('ik, ijkl, jl', x, icov, x)
        return -0.5*chi2
    return loglike

def fg_removal(imaps, templates, covmats):
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
