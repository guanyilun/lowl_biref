import os, os.path as op
import healpy as hp
import numpy as np
from pixell import utils

# common directories (on tigercpu)
simdir = "/projects/ACT/yilung/work/lowl_biref/data/"

def get_fname(comp, chan, sid=None, ext="fits"):
    oname = comp
    if chan is not None: oname += f"_{chan}"
    if sid  is not None: oname += f"_{sid:03d}"
    oname += f".{ext}"
    return oname

def ps(tag, comp):
    return op.join(simdir, tag, f"ps_{comp}.hdf")

def load_sim(tag, comp, chan=None, sid=None):
    """load simulation based on a tag, sid, and given channel. If chan
    is None, it will load all channels."""
    if chan is None:
        maps = {}
        chans, _, _, _, _ = load_bandpass()
        for chan in chans: maps[chan] = load_sim(tag, comp, chan, sid)
        return maps
    elif isinstance(chan, list):
        maps = {}
        for c in chan: maps[c] = load_sim(tag, comp, c, sid)
        return maps
    # for dust and synchrotron model, no need to apply sid
    if comp[0] in ['d','s']: sid=None
    oname = op.join(simdir, tag, get_fname(comp, chan, sid))
    if not op.exists(oname): raise ValueError("sim not found")
    with utils.nowarn():
        imap = hp.read_map(oname, field=(0,1,2), dtype=float)
    return imap
def cmb_sim(odir, sid=0):
    return op.join(odir, f"cmb_orig_{sid:03d}.fits")
def load_bandpass(rows=None):
    if rows is None:
        # load bandpass information
        name, band_c, band_w, beam_r, sens = np.genfromtxt("data/bandpass_toshiya.txt",
                                                           dtype=None, encoding=None, unpack=True)
    else:
        # unpack to beg and end (inclusive)
        r_beg, r_end = rows
        # convert to nskip and nrows
        r_skip = r_beg - 1
        nrows = r_end - r_beg + 1
        # load bandpass information
        name, band_c, band_w, beam_r, sens = np.genfromtxt("data/bandpass_toshiya.txt",
                                                           dtype=None, encoding=None, unpack=True,
                                                           skip_header=r_skip, max_rows=nrows)        

    return (name, band_c, band_w, beam_r, sens)
def write_map(odir, m, comp, chan, sid=0, overwrite=True, verbose=False):
    if comp in ['cmb','tot']:
        oname = op.join(odir, f"{comp}_{chan}_{sid:03d}.fits")
    else:
        oname = op.join(odir, f"{comp}_{chan}.fits")
    if verbose: print("Writing:", oname)
    if not overwrite and op.exists(oname): return
    else: hp.write_map(oname, m, dtype=float, overwrite=overwrite)
def read_map(odir, comp, chan, sid=None):
    if comp in ['cmb','tot']:
        oname = op.join(odir, f"{comp}_{chan}_{sid:03d}.fits")
    else:
        oname = op.join(odir, f"{comp}_{chan}.fits")
    return hp.read_map(oname, field=(0,1,2), dtype=float)

# chans, _, _, _, _ = load_bandpass() 