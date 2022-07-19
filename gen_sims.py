import argparse, os, os.path as op
import numpy as np, healpy as hp
import pysm3, pysm3.units as u
from enlib import bunch, bench, utils
from pathlib import Path
import files, lib

# alias
Bunch = bunch.Bunch
Tcmb  = 2.725*1e6

def generate_cls(lmax, r=0):
    """generate power spectrum with an isotropic rotation that
    applies to recombination only.

    Parameters:
    -----------
      lmax: maximum multipole to generate cls

    """
    import classy

    params = {
        'output': 'tCl,pCl,lCl',
        'l_max_scalars': lmax,
        'modes':'s,t',
        'r': r,
        'lensing': 'yes',
        'accurate_lensing': 1
    }
    # compute cosmology without reionization bump
    params['reio_parametrization'] = 'reio_none'
    cosmo = classy.Class()
    cosmo.set(params)
    with bench.show("compute cls without reio"):
        cosmo.compute()
    # obtain power spectrum without reio bump
    cls_no_rb = cosmo.lensed_cl()
    # compute cosmology with reionization bump
    params['reio_parametrization'] = 'reio_camb'
    cosmo = classy.Class()
    cosmo.set(params)
    with bench.show("compute cls with reio"):
        cosmo.compute()
    # obtain power spectrum with reio bump
    cls_with_rb = cosmo.lensed_cl()
    # obtain the power spectrum of reio bump as the
    # difference of the two at low ells
    cls_rb = {}
    reio_lmax = 19
    ls = cls_with_rb['ell']
    for sp in ['tt','ee','bb','te']:
        cls_rb[sp] = (ls*0).astype(float)
        if sp in ['te', 'ee']:
            sel = slice(0, reio_lmax+1)
            cls_rb[sp][sel] = cls_with_rb[sp][sel] - cls_no_rb[sp][sel]
    # FIXME: be more careful when substracting so that I don't get
    # negative cl here as a simple solution, I will just replace the
    # negative part with 0
    return Bunch(
        with_rb=cls_with_rb,
        no_rb=cls_no_rb,
        rb=cls_rb
    )

def generate_cmb_rot(cls, alpha, nside=512, seed=None, add_rb=True, return_rb=False):
    if seed: np.random.seed(seed)
    lmax = 4 * nside
    with bench.show("gen tqu maps"):
        # generate sims
        tqu_map = hp.synfast([cls.no_rb[sp]*Tcmb**2 for sp in ['tt','ee','bb','te']],
                             new=True, nside=nside)
    if alpha:
        with bench.show("rotate tqu maps"):
            # rotate maps
            tqu_map_rot = lib.rotate_pol(tqu_map, alpha)
    else:  # if no rotation is needed, skip that
        tqu_map_rot = tqu_map
    if add_rb:
        if seed: np.random.seed(seed+100)
        with bench.show("gen reio tqu maps"):
            # generate maps from reionization bump
            tqu_map_rb = hp.synfast([cls.rb[sp]*Tcmb**2 for sp in ['tt','ee','bb','te']],
                                    new=True, nside=nside)
            # add the rb tqu maps to the cmb tqu maps
            tqu_map_rot += tqu_map_rb
    if return_rb: return tqu_map_rot, tqu_map_rb
    else: return tqu_map_rot

def get_emission(sky, fc, bw=None, bandpass_int=False, unit=u.K_CMB):
    if bandpass_int:
        fmin = fc - bw/2
        fmax = fc + bw/2
        fsteps = int(np.ceil(fmax - fmin) + 1)
        bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
        weights = np.ones(len(bandpass_frequencies))
        omap = sky.get_emission(bandpass_frequencies, weights)
        omap *= pysm3.bandpass_unit_conversion(bandpass_frequencies, weights, unit)
    else:
        omap = sky.get_emission(fc * u.GHz)
        omap = omap.to(unit, equivalencies=u.cmb_equivalencies(fc * u.GHz))
    return omap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--odir', default='out')
    parser.add_argument('--nside', type=int, default=512)
    parser.add_argument('--lmax', type=int, default=3000)
    parser.add_argument('--alpha', type=float, default=0.35, help='biref in deg')
    parser.add_argument('--sigma-beta', type=float, default=0.2, help='sigma of det miscal in deg')
    parser.add_argument('--sid', type=int, default=0, help='sim id')
    parser.add_argument('--gen-cmb', action='store_true')
    parser.add_argument('--gen-fg', action='store_true')
    parser.add_argument('--gen-noise', action='store_true')
    parser.add_argument('--comps', default="")
    parser.add_argument('--cmb-seed', type=int, default=0, help='seed for cmb realization')
    parser.add_argument('--noise-seed', type=int, default=0, help='seed for noise realization')
    parser.add_argument('--det-seed', type=int, default=0, help='seed for det rotation realization')
    parser.add_argument('--bandpass-int', action='store_true')
    parser.add_argument('--unit', default='K_CMB')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not op.exists(args.odir): os.makedirs(args.odir)
    os.environ["PYSM_LOCAL_DATA"] = op.abspath(args.odir)
    unit  = u.Unit(args.unit)
    alpha = np.deg2rad(args.alpha)
    nside = args.nside
    npix  = hp.nside2npix(nside)

    # load bandpass information
    names, fcs, bws, beams, sens = files.load_bandpass()

    # dictonary to store all maps by keys
    maps  = {}
    # load or generate cmb sky if needed
    oname = files.cmb_sim(args.odir, sid=args.sid)
    if args.gen_cmb:
        # generate cls
        cls = generate_cls(lmax=args.lmax)
        # generate tqu maps
        tqu_map = generate_cmb_rot(cls, alpha, seed=args.cmb_seed)
        # write maps
        hp.write_map(oname, tqu_map, dtype=float, overwrite=args.overwrite)
    # make a sky model
    with utils.nowarn():
        sky = pysm3.Sky(
            nside=args.nside,
            component_objects=[
                pysm3.CMBMap(args.nside, map_IQU=Path(op.basename(oname)))
            ])
    # generate a map for each channel
    maps['cmb'] = np.zeros((len(fcs), 3, npix))
    with bench.show("get cmb emissions"):
        for i, (fc, bw, beam) in enumerate(zip(fcs, bws, beams)):
            cmb_map = get_emission(sky, fc, bw, bandpass_int=args.bandpass_int, unit=unit)
            if args.smooth:
                with utils.nowarn():
                    cmb_map = hp.smoothing(cmb_map, fwhm=np.deg2rad(beam/60))
            # store cmb maps for each channel
            maps['cmb'][i] = cmb_map

    # generate foreground components
    # see for model description:
    # https://pysm3.readthedocs.io/en/latest/models.html
    if args.comps: components = args.comps.split(",")
    else: components = []
    for comp in components:
        maps[comp] = maps['cmb']*0
        with bench.show(f"get fg emission for comp: {comp}"):
            if args.gen_fg:
                with utils.nowarn():
                    sky = pysm3.Sky(nside=args.nside, preset_strings=[comp])
                for i, (fc, bw, beam) in enumerate(zip(fcs, bws, beams)):
                    fg_map = get_emission(sky, fc, bw, bandpass_int=args.bandpass_int, unit=unit)
                    # smooth the map if necessary
                    if args.smooth:
                        with utils.nowarn():
                            fg_map = hp.smoothing(fg_map, fwhm=np.deg2rad(beam/60))
                    maps[comp][i] = fg_map
            else:
                for i, name in enumerate(names):
                    with utils.nowarn():
                        maps[comp][i] = files.read_map(args.odir, comp, name)
    # save each component at each channel if newly generated
    to_save = []
    if args.gen_cmb: to_save += ['cmb']  # note that this takes too much space
    if args.gen_fg:  to_save += components
    for comp in to_save:
        for i, name in enumerate(names):
            files.write_map(args.odir, maps[comp][i], comp, name, sid=args.sid, verbose=True)
    # calculate coadded map of all components
    # with bench.show("coadd maps"):
    #     coadd_maps = np.sum([m for _, m in maps.items()], axis=0)
    # # assume the instrument is off by an angle of beta where
    # # we use a random variable on sub degree scale to
    # # simulate that for each channel
    # np.random.seed(args.det_seed)  # specify a seed
    # betas = np.random.randn(len(names)) * args.sigma_beta
    # print("betas:", betas)
    # np.savetxt(op.join(args.odir, 'betas.txt'), betas)  # in deg
    # betas = np.deg2rad(betas)
    # with bench.show("rotate total maps"):
    #     rotated_coadd_maps = []
    #     for map_chan, beta in zip(coadd_maps, betas):
    #         rotated_coadd_maps.append(lib.rotate_pol(map_chan, beta))
    #     coadd_maps = np.array(rotated_coadd_maps)
    #     del rotated_coadd_maps
    # # add noise
    # if args.gen_noise:
    #     nmaps = np.zeros((len(fcs), 3, npix))
    #     with bench.show("gen noise sim"):
    #         for i, (fc, bw, sen) in enumerate(zip(fcs, bws, sens)):
    #             np.random.seed(args.noise_seed+i)
    #             P_sen = sen * u.uK_CMB
    #             P_sen = P_sen.to_value(unit, equivalencies=u.cmb_equivalencies(fc*u.GHz))
    #             P_rms = P_sen / hp.nside2resol(nside, arcmin=True)
    #             T_rms = P_rms / np.sqrt(2)
    #             tot_rms  = np.array([T_rms, P_rms, P_rms]).reshape(3, 1)
    #             nmaps[i] = np.random.randn(3, npix) * tot_rms
    #     coadd_maps += nmaps
    # for i, name in enumerate(names):
    #     files.write_map(args.odir, coadd_maps[i], 'tot', name, sid=args.sid, verbose=True)