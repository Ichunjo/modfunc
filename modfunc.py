"""
    A collection of modded VapourSynth functions and utilities.

    The main difference between this and lvsfunc is that this *func contains functions that were purely changed to add a specific additional functionality.
    Functions in lvsfunc are either new ones, ones that are not readily available elsewhere, or refurbished functions with new functionalities and uses.

    If you have a request, feel free to send in an issue.
    If you have a function to add, please send in a Pull Request.
"""

import vapoursynth as vs
from vsutil import *
import fvsfunc as fvf
from functools import partial
core = vs.core


def adaptive_grain_mod(clip: vs.VideoNode,
                       strength=0.25,
                       cstrength=0,
                       hcorr=0,
                       vcorr=None,
                       static=True,
                       luma_scaling=12,
                       show_mask=False) -> vs.VideoNode:
    """
    Original *Func: kagefunc

    Added functionalities:
        - Allow adding grain to chroma
        - Allow changing hcorr and vcorr
    """
    import numpy as np

    def fill_lut(y):
        x = np.arange(0, 1, 1 / (1 << 8))
        z = (1 - (x * (1.124 + x * (-9.466 + x * (36.624 + x * (-45.47 + x * 18.188)))))) ** ((y ** 2) * luma_scaling)
        if clip.format.sample_type == vs.INTEGER:
            z = z * 255
            z = np.rint(z).astype(int)
        return z.tolist()

    def generate_mask(n, f, clip):
        frameluma = round(f.props.PlaneStatsAverage * 999)
        table = lut[int(frameluma)]
        return core.std.Lut(clip, lut=table)

    lut = [None] * 1000
    for y in np.arange(0, 1, 0.001):
        lut[int(round(y * 1000))] = fill_lut(y)

    if vcorr is None:
        vcorr = hcorr

    luma = get_y(fvf.Depth(clip, 8)).std.PlaneStats()
    grained = core.grain.Add(clip, var=strength, uvar=cstrength, constant=static, hcorr=hcorr, vcorr=vcorr)

    mask = core.std.FrameEval(luma, partial(generate_mask, clip=luma), prop_src=luma)
    mask = core.resize.Spline36(mask, clip.width, clip.height)

    if get_depth(clip) != 8:
        mask = fvf.Depth(mask, bits=get_depth(clip))
    if show_mask:
        return mask
    return core.std.MaskedMerge(clip, grained, mask)

