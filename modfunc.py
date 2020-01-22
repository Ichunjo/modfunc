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
import mvsfunc as mvf
from functools import partial
core = vs.core


def adaptive_grain_mod(clip: vs.VideoNode,
                       strength=0.25, cstrength=0,
                       hcorr=0, vcorr=None,
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


def hybriddenoise_mod(clip: vs.VideoNode, knl=0.5, sigma=2, radius1=1, depth=16) -> vs.VideoNode:
    """
    Original *Func: kagefunc

    Added functionalities:
        - Work in float 32 bits
        - Allow the depth to be changed at the output
    """
    if get_depth(clip) != 32:
        clip = clip.resize.Point(format=clip.format.replace(bits_per_sample=32, sample_type=vs.FLOAT))
    y = get_y(clip)
    y = mvf.BM3D(y, radius1=radius1, sigma=sigma)
    denoised = core.knlm.KNLMeansCL(clip, a=2, h=knl, d=3, device_type='gpu', device_id=0, channels='UV')
    return fvf.Depth(core.std.ShufflePlanes([y, denoised], planes=[0, 1, 2], colorfamily=vs.YUV), depth)


def DescaleAA_mod(src: vs.VideoNode,
                  w=None, h=720,
                  kernel='bicubic', b=1/3, c=1/3, taps=3,
                  thr=10, expand=3, inflate=3, showmask=False,
                  opencl=False, device=0) -> vs.VideoNode:
    """
    Original *Func: fvsfunc

    Added functionalities:
        - Automatic calculation of the width
        - Allow using opencl for the upscale
    """
    try:
        import nnedi3_rpow2 as nnp2  # https://gist.github.com/4re/342624c9e1a144a696c6
    except ImportError:
        import edi_rpow2 as nnp2  # https://gist.github.com/YamashitaRen/020c497524e794779d9c
    import nnedi3_rpow2CL as nnp2CL # https://github.com/Ichunjo/nnedi3_rpow2CL/blob/master/nnedi3_rpow2CL.py

    if kernel.lower().startswith('de'):
        kernel = kernel[2:]

    w = get_w(h)

    ow = src.width
    oh = src.height

    bits = src.format.bits_per_sample
    sample_type = src.format.sample_type

    if sample_type == vs.INTEGER:
        maxvalue = (1 << bits) - 1
        thr = thr * maxvalue // 0xFF
    else:
        maxvalue = 1
        thr /= (235 - 16)

    # Fix lineart
    src_y = core.std.ShufflePlanes(src, planes=0, colorfamily=vs.GRAY)
    deb = fvf.Resize(src_y, w, h, kernel=kernel, a1=b, a2=c, taps=taps, invks=True)
    if opencl:
        sharp = nnp2CL.nnedi3_rpow2CL(deb, 2, ow, oh, device=device)
    else:
        sharp = nnp2.nnedi3_rpow2(deb, 2, ow, oh)
    thrlow = 4 * maxvalue // 0xFF if sample_type == vs.INTEGER else 4 / 0xFF
    thrhigh = 24 * maxvalue // 0xFF if sample_type == vs.INTEGER else 24 / 0xFF
    edgemask = core.std.Prewitt(sharp, planes=0)
    edgemask = core.std.Expr(edgemask, "x {thrhigh} >= {maxvalue} x {thrlow} <= 0 x ? ?"
                                       .format(thrhigh=thrhigh, maxvalue=maxvalue, thrlow=thrlow))
    if kernel == "bicubic" and c >= 0.7:
        edgemask = core.std.Maximum(edgemask, planes=0)
    sharp = core.resize.Point(sharp, format=src.format.id)

    # Restore true 1080p
    deb_upscale = fvf.Resize(deb, ow, oh, kernel=kernel, a1=b, a2=c, taps=taps)
    diffmask = core.std.Expr([src_y, deb_upscale], 'x y - abs')
    for _ in range(expand):
        diffmask = core.std.Maximum(diffmask, planes=0)
    for _ in range(inflate):
        diffmask = core.std.Inflate(diffmask, planes=0)

    mask = core.std.Expr([diffmask,edgemask], 'x {thr} >= 0 y ?'.format(thr=thr))
    mask = mask.std.Inflate().std.Deflate()
    out = core.std.MaskedMerge(src, sharp, mask, planes=0)

    if showmask:
        out = mask
    return out

def to_444(clip, w=None, h=None, join=True):
    """
    Original location : Zastin's pastebin https://pastebin.com/u/Zastin
    """
    uv = [nnedi3x2(c) for c in split(clip)[1:]]
    
    if w in (None, clip.width) and h in (None, clip.height):
        uv = [core.fmtc.resample(c, sy=0.5, flt=0) for c in uv]
    else:
        uv = [core.resize.Spline36(c, w, h, src_top=0.5) for c in uv]
    
    return core.std.ShufflePlanes([clip] + uv, [0]*3, vs.YUV) if join else uv

def nnedi3x2(clip):
    if hasattr(core, 'znedi3'):
        return clip.std.Transpose().znedi3.nnedi3(1, 1, 0, 0, 4, 2).std.Transpose().znedi3.nnedi3(0, 1, 0, 0, 4, 2)
    else:
        return clip.std.Transpose().nnedi3.nnedi3(1, 1, 0, 0, 3, 1).std.Transpose().nnedi3.nnedi3(0, 1, 0, 0, 3, 1)
