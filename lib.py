import files
from enlib.bunch import Bunch
import healpy as hp, numpy as np
import nawrapper as nw
from pixell import utils as u

from orphics import cosmology as cosmo
from cosmoslib.utils import cwignerd

def compute_spectra(imap1, imap2=None):
    """basic calculation of power spectrum based on healpy"""
    if imap2 is None:
        alm = hp.map2alm(imap1)
        cls = hp.alm2cl(alm)
    else:
        alm  = hp.map2alm(imap1)
        alm2 = hp.map2alm(imap2)
        cls  = hp.alm2cl(alm, alm2)
    return cls

def compute_spectra_namaster(imap1, imap2, lmin, lmax, lbin_widths=1,
                             masks=None, beams=None, wfunc=None,
                             mcm_dir=".mcm", overwrite=False,
                             purify_e=False, purify_b=False):
    """Calculate power spectrum using nawrapper (namaster algorithm)

    Parameters
    ----------
    imap1       : tqu maps of len 3
    imap2       : tqu maps of len 3
    masks       : tuple of len 2, e.g., (Tmask, Bmask)
    beams       : tuple of len 2, e.g., (Tbeam, Bbeam)
    lbin_widths : with of ell bin, can be a list or a number
    mcm_dir     : path to store mode-coupling matrix
    overwrite   : whether to overwrite mcm

    """
    # create namaster map objects
    # assume healpix pixelization

    namap_1 = nw.namap_hp(maps=imap1, masks=masks, beams=beams, purify_e=purify_e, purify_b=purify_b)
    namap_2 = nw.namap_hp(maps=imap2, masks=masks, beams=beams, purify_e=purify_e, purify_b=purify_b)

    # create binning
    bins = nw.create_binning(lmin=lmin, lmax=lmax, widths=lbin_widths,
                             weight_function=wfunc)
    # compute mode coupling matrix
    mc = nw.mode_coupling(namap_1, namap_2, bins, mcm_dir=mcm_dir,
                          overwrite=overwrite)
    # compute spectra
    cls = nw.compute_spectra(namap_1, namap_2, mc=mc)
    return cls

def get_chan(chan):
    chans, bcs, bws, brs, sens = files.load_bandpass()
    i = list(chans).index(chan)
    if i == -1: raise ValueError("Channel not found!")
    else:
        return Bunch(chan=chans[i],bc=bcs[i],bw=bws[i],
                     fwhm=brs[i],sens=sens[i])
def get_bl(chan, lmax=512):
    chdata = get_chan(chan)
    return hp.gauss_beam(np.deg2rad(chdata.fwhm/60), lmax)

def est_alpha(cleb, clee, clbb=None, lmin=None, lmax=None):
    """Estimate isotropic rotation angle from cleb, clee, and clbb.
    If clbb is None, it will be ignored.
    """
    ls = np.arange(len(cleb))
    if lmin is None: lmin = ls.min()
    if lmax is None: lmax = ls.max()
    m  = (ls >= lmin) * (ls <= lmax)
    # cleb_l = clee_l * A + noise_l
    # for simplicity we assume diagonal noise:
    # <n_i, n_j> = delta_ij c_j = delta_ij c
    # (this assumes c_j is const)
    ls   = ls[m]
    clee = clee[m]
    cleb = cleb[m]
    if clbb is not None: clbb = clbb[m]
    else: clbb = np.zeros_like(clee)
    # nl   = np.ones_like(ls)  # dummy nl
    nl  = 1/(2*ls+1)*(clee*clbb)
    cov  = np.diag(nl)
    P    = clee - clbb
    N    = np.linalg.inv(cov)
    div  = np.einsum("i,ij,j", P, N, P)
    A    = np.einsum("i,ij,j", P, N, cleb) / div
    return np.arcsin(2*A)/4

def est_alpha_sim(ocleb, oclee, oclbb, scleb, sclee, sclbb,
                  lmin=None, lmax=None, binner=None):
    """Estimate isotropic rotation angle from cleb, clee, and clbb.
    If clbb is None, it will be ignored. Input power spectra that
    start with o are observed, those that start with s are sims.
    We shall assume sims have shape (nsim, nell)
    """
    ls = np.arange(len(ocleb))
    if lmin is None: lmin = ls.min()
    if lmax is None: lmax = ls.max()
    m  = (ls >= lmin) * (ls <= lmax)
    # cleb_l = clee_l * A + noise_l
    # for simplicity we assume diagonal noise:
    # <n_i, n_j> = delta_ij c_j = delta_ij c
    # (this assumes c_j is const)
    ls    = ls[m]
    oclee = oclee[m]
    ocleb = ocleb[m]
    oclbb = oclbb[m]
    # simulation, assume (nsim, nell)
    sclee = sclee[:,m]
    scleb = scleb[:,m]
    sclbb = sclbb[:,m]
    # binning if needed
    if binner is not None:
        assert np.allclose(binner.ells, ls)
        oclee = binner.bindata(oclee)
        ocleb = binner.bindata(ocleb)
        oclbb = binner.bindata(oclbb)
        sclee = binner.bindata(sclee)
        scleb = binner.bindata(scleb)
        sclbb = binner.bindata(sclbb)
    # compute fiducial cosmology based on sims
    fclee = np.mean(sclee, axis=0)
    fcleb = np.mean(scleb, axis=0)
    fclbb = np.mean(sclbb, axis=0)
    # compute covariance from sims
    cov   = np.cov(scleb, rowvar=False)
    # if binner is None:
    #     nl  = 1/(2*ls+1)*(fcleb**2+(oclee*oclbb))
    #     cov  = np.diag(nl)
    P    = fclee - fclbb
    N    = np.linalg.inv(cov)
    div  = np.einsum("i,ij,j", P, N, P)
    A    = np.einsum("i,ij,j", P, N, ocleb) / div
    # repeat for sims
    Asim = np.einsum("i,ij,kj", P, N, scleb) / div
    return np.arcsin(2*A)/4, np.arcsin(2*Asim)/4

def rotate_pol(imap, angle):
    """imap is assumed to have a shape of (3, npix) angle is
    assumed to be in radian.
    """
    c, s = np.cos(2*angle), np.sin(2*angle)
    omap = imap.copy()
    omap[1] = c * imap[1] - s * imap[2]
    omap[2] = s * imap[1] + c * imap[2]
    return omap

def get_smoothing_fwhms(fwhms, target):
    """Get smoothing fwhms to get a list of fwhms into the target fwhm"""
    # if no target is specified, use the maximal fwhm given
    if target is None: target = np.max(fwhms)
    diff = target**2 - fwhms**2
    diff[diff < 0] = 0  # edge case: do nothing when target beam is smaller
    return np.sqrt(diff)

def beam_match(imap, src, tgt):
    """smooth imap from a beam fwhm of src to tgt, both specified in fwhm in arcmin.
    We assume the tgt has a larger beam radius.
    """
    # get beam radius to smooth
    if tgt <= src: return imap
    fwhm = (tgt**2 - src**2)**0.5
    omap = hp.smoothing(imap, fwhm=fwhm*u.arcmin)
    return omap

class SignalCov:
    def __init__(self, cl=None, lmax=200, nside=16):
        """
        Parameters
        ----------
        aps: dict with keys like "EE", "TT", etc.
        lmax: maximum ell to sum
        # alpha: rotation angle, if necessary because it induces QU covariance which is 0 with alpha=0

        Returns
        -------
        covmat with shape(2, 2, npix x npix)

        """
        npix = hp.nside2npix(nside)
        x,y,z = hp.pix2vec(nside, np.arange(npix))
        # pairwise dot products
        cos = np.outer(x,x) + np.outer(y,y) + np.outer(z,z)
        # make sure cosine values are well behaved
        cos = np.clip(cos, -1, 1)
        flat_cos = np.ravel(cos)

        self.ell = np.arange(lmax+1)
        if cl is None:
            ell = np.arange(lmax+1)
            cl = cosmo.default_theory()
            EE = cl.lCl('EE', ell)
            BB = cl.lCl('BB', ell)
            cl = {}
            cl['EE'] = EE
            cl['BB'] = EE
        else:
            EE = cl['EE']
            BB = cl['BB']
        A = cwignerd.wignerd_cf_from_cl( 2,2,1,len(flat_cos),lmax,flat_cos,(EE+BB)/2)
        B = cwignerd.wignerd_cf_from_cl(-2,2,1,len(flat_cos),lmax,flat_cos,(EE-BB)/2)
        A = A.reshape(cos.shape)
        B = B.reshape(cos.shape)

        # covariance assuming no rotation
        self.cov = np.array([[A+B, 0*A],[0*A, A-B]])
        self.A = A
        self.B = B

    def calc(self, alpha=0):
        if alpha==0:  return self.cov
        # assumption: change in B from rotation is small
        B_unrot = self.B
        off_diag = np.sin(4*alpha)*B_unrot
        self.cov[0,1,...] = off_diag
        self.cov[1,0,...] = off_diag
        return self.cov

class NoiseCov:
    def __init__(self, nl=None, nlev=1, fwhm=0, lmax=200, alpha=0, nside=16):
        """
        Parameters
        ----------
        nlev: tt noise level in uK arcmin
        fwhm: in arcmin
        lmax: maximum ell to sum

        Returns
        -------
        covmat with shape(2, 2, npix x npix)

        """
        npix = hp.nside2npix(nside)
        x,y,z = hp.pix2vec(nside, np.arange(npix))
        # pairwise dot products
        cos = np.outer(x,x) + np.outer(y,y) + np.outer(z,z)
        # make sure cosine values are well behaved
        cos = np.clip(cos, -1, 1)
        flat_cos = np.ravel(cos)
        self.ell = np.arange(lmax+1)
        if nl is None:
            if fwhm > 0:  bl = hp.gauss_beam(np.deg2rad(fwhm/60), lmax)
            else: bl = np.ones(lmax+1)
            nl = nlev**2*2 * bl**2
        A = cwignerd.wignerd_cf_from_cl( 2,2,1,len(flat_cos),lmax,flat_cos,nl)
        B = cwignerd.wignerd_cf_from_cl(-2,2,1,len(flat_cos),lmax,flat_cos,nl)
        A = A.reshape(cos.shape)
        B = B.reshape(cos.shape)
        # covariance assuming no rotation
        self.cov = np.array([[A+B, 0*A],[0*A, A-B]])
        self.A = A
        self.B = B

    def calc(self, alpha=0):
        """alpha = [alpha_i, ...] where i are each component"""
        return self.cov/(1-np.sum(alpha))**2

def icov(covmat):
    """input covariance matrix has shape, 2, 2, npix, npix"""
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
