"""Forensic feature vector for the ML classifier (shared by train and predict).

All features are deterministic functions of the pixel data (numpy + Pillow
only). They target physical differences between camera captures and
generative-model output: sensor-noise statistics and cross-channel noise
correlation (demosaicing), radial FFT spectrum shape, gradient statistics,
local-variance uniformity, and color-saturation statistics.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

_SINGLE_SCALE_NAMES = [
    *[f"fft_radial_{i}" for i in range(12)],
    "fft_high_low_ratio",
    "fft_peakiness",
    "fft_slope",
    "noise_mean",
    "noise_std",
    "noise_dead_frac",
    "noise_chan_corr_rg",
    "noise_chan_corr_gb",
    "noise_autocorr_h",
    "noise_autocorr_v",
    "noise_autocorr_h4",
    "noise_autocorr_v4",
    "noise_autocorr_h8",
    "noise_autocorr_v8",
    "grid_freq8_energy",
    "grid_freq4_energy",
    "blur_residual_mean",
    "blur_residual_ratio",
    "lsb_entropy",
    "grad_mean",
    "grad_std",
    "grad_kurtosis",
    "blockvar_uniformity",
    "sat_mean",
    "sat_std",
    "blockiness_8",
]

# Full vector = features at native tile scale plus at 2x downsample —
# generative artifacts live at specific scales, cameras' noise at others.
FEATURE_NAMES = _SINGLE_SCALE_NAMES + [f"{n}_s2" for n in _SINGLE_SCALE_NAMES]

TILE = 256


def _radial_profile(mag: np.ndarray, nbins: int = 8) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    r = r / (r.max() + 1e-9)
    prof = np.empty(nbins)
    for i in range(nbins):
        mask = (r >= i / nbins) & (r < (i + 1) / nbins)
        prof[i] = np.log1p(mag[mask]).mean() if mask.any() else 0.0
    return prof - prof[0]  # normalize against overall brightness/energy


def _kurtosis(x: np.ndarray) -> float:
    x = x.ravel()
    m = x.mean()
    s = x.std() + 1e-9
    return float(((x - m) ** 4).mean() / s**4)


def tile_features(rgb: np.ndarray) -> np.ndarray:
    """Two-scale feature vector for one RGB uint8 array (TILE x TILE x 3)."""
    half = np.asarray(
        Image.fromarray(rgb).resize((rgb.shape[1] // 2, rgb.shape[0] // 2)),
        dtype=np.uint8,
    )
    return np.concatenate([_single_scale(rgb), _single_scale(half)])


def _single_scale(rgb: np.ndarray) -> np.ndarray:
    rgbf = rgb.astype(np.float64)
    gray = rgbf.mean(axis=2)

    # --- FFT spectrum shape
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    prof = _radial_profile(f, nbins=12)
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h // 2, xx - w // 2)
    r = r / (r.max() + 1e-9)
    low = f[r < 0.25].mean() + 1e-9
    high = f[r > 0.60].mean() + 1e-9
    high_low = np.log(low / high)
    peakiness = np.log1p(f[r > 0.60].max() / high)
    # spectral decay exponent: slope of log-power vs log-frequency
    bins = np.linspace(0.05, 0.95, 10)
    logp = []
    for i in range(len(bins) - 1):
        m = (r >= bins[i]) & (r < bins[i + 1])
        logp.append(np.log(f[m].mean() + 1e-9))
    slope = float(np.polyfit(np.log((bins[:-1] + bins[1:]) / 2), logp, 1)[0])

    # --- noise residual (median filter) and cross-channel correlation
    img = Image.fromarray(rgb)
    med = np.asarray(img.filter(ImageFilter.MedianFilter(3)), dtype=np.float64)
    res = rgbf - med
    res_gray = np.abs(res).mean(axis=2)
    b = 16
    hb, wb = (res_gray.shape[0] // b) * b, (res_gray.shape[1] // b) * b
    blocks = res_gray[:hb, :wb].reshape(hb // b, b, wb // b, b).mean(axis=(1, 3))
    noise_mean = res_gray.mean()
    noise_std = blocks.std()
    dead_frac = float((blocks < 0.3).mean())

    def corr(a, bch):
        a, bch = a.ravel(), bch.ravel()
        sa, sb = a.std() + 1e-9, bch.std() + 1e-9
        return float(((a - a.mean()) * (bch - bch.mean())).mean() / (sa * sb))

    corr_rg = corr(res[:, :, 0], res[:, :, 1])
    corr_gb = corr(res[:, :, 1], res[:, :, 2])

    # noise spatial autocorrelation (camera noise ~white; synthetic residuals
    # are spatially structured)
    rg = res_gray - res_gray.mean()
    auto_h = corr(rg[:, :-1], rg[:, 1:])
    auto_v = corr(rg[:-1, :], rg[1:, :])
    auto_h4 = corr(rg[:, :-4], rg[:, 4:])
    auto_v4 = corr(rg[:-4, :], rg[4:, :])
    auto_h8 = corr(rg[:, :-8], rg[:, 8:])
    auto_v8 = corr(rg[:-8, :], rg[8:, :])

    # latent-grid frequency energy: diffusion VAE decoders leave periodic
    # structure at image_size/8 and /4; measure spectral energy at those
    # frequencies relative to their neighborhoods (uses the shifted FFT `f`)
    def band_energy(period: int) -> float:
        k = min(h, w) // period  # frequency index for that spatial period
        ring = (np.abs(r * (min(h, w) / 2) - k) < 1.5)
        neigh = (np.abs(r * (min(h, w) / 2) - k) < 6) & ~ring
        return float(np.log1p(f[ring].mean() / (f[neigh].mean() + 1e-9)))

    grid8 = band_energy(8)
    grid4 = band_energy(4)

    # second-scale residual: what survives a stronger blur
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(2)), dtype=np.float64)
    bres = np.abs(rgbf - blur).mean()
    blur_ratio = float(noise_mean / (bres + 1e-9))

    # LSB entropy: cameras have near-random low bits, renders often do not
    lsb = (rgb[:, :, 1] & 1).astype(np.float64)
    p1 = lsb.mean()
    p1 = min(max(p1, 1e-9), 1 - 1e-9)
    lsb_entropy = float(-(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1)))

    # --- gradients
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gmag = np.hypot(gx[:-1, :], gy[:, :-1])
    grad_mean = gmag.mean()
    grad_std = gmag.std()
    grad_kurt = min(_kurtosis(gmag), 200.0)

    # --- local variance uniformity
    gvar = gray[:hb, :wb].reshape(hb // b, b, wb // b, b).std(axis=(1, 3))
    blockvar_unif = float(gvar.std() / (gvar.mean() + 1e-9))

    # --- saturation
    mx = rgbf.max(axis=2)
    mn = rgbf.min(axis=2)
    sat = (mx - mn) / (mx + 1e-9)
    sat_mean = sat.mean()
    sat_std = sat.std()

    # --- 8px blockiness (JPEG grid energy)
    col_d = np.abs(np.diff(gray, axis=1))
    row_d = np.abs(np.diff(gray, axis=0))
    on_grid = col_d[:, 7::8].mean() + row_d[7::8, :].mean()
    off_grid = col_d.mean() + row_d.mean() + 1e-9
    blockiness = float(on_grid / off_grid)

    return np.array([
        *prof, high_low, peakiness, slope,
        noise_mean, noise_std, dead_frac, corr_rg, corr_gb,
        auto_h, auto_v, auto_h4, auto_v4, auto_h8, auto_v8, grid8, grid4,
        bres, blur_ratio, lsb_entropy,
        grad_mean, grad_std, grad_kurt,
        blockvar_unif, sat_mean, sat_std, blockiness,
    ])


def image_tiles(path: str, max_tiles: int = 9) -> list[np.ndarray]:
    """Up to max_tiles TILE-sized RGB crops sampled on a grid (center-ish)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if min(w, h) < TILE:
        img = img.resize((max(TILE, w), max(TILE, h)))
        w, h = img.size
    cols = max(1, w // TILE)
    rows = max(1, h // TILE)
    coords = [(r, c) for r in range(rows) for c in range(cols)]
    step = max(1, len(coords) // max_tiles)
    tiles = []
    for r, c in coords[::step][:max_tiles]:
        x, y = c * TILE, r * TILE
        if x + TILE <= w and y + TILE <= h:
            tiles.append(np.asarray(img.crop((x, y, x + TILE, y + TILE)), dtype=np.uint8))
    if not tiles:  # small image: center crop after resize
        img2 = img.resize((TILE, TILE))
        tiles.append(np.asarray(img2, dtype=np.uint8))
    return tiles


def image_features(path: str, max_tiles: int = 9) -> np.ndarray:
    """Per-tile feature matrix (n_tiles x n_features) for an image file."""
    return np.stack([tile_features(t) for t in image_tiles(path, max_tiles)])
