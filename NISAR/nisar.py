# ----------------------------------------------------------------------------
# This file is part of the InSARdev project: https://github.com/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
# ----------------------------------------------------------------------------
"""
NISAR Interferogram Processor — Compact Version (scipy + cv2)
=============================================================
Same result as nisar_numpy.py in a flat linear script using scipy/cv2.
Every step is visible top-to-bottom with no function jumps.

Dependencies: numpy, h5py, scipy, cv2 (opencv-python), matplotlib
Lines: 76 code + 5 imports + 6 prints + 13 plot + 17 comments + 9 docstrings + 18 empty = 144
"""
import numpy as np
import h5py
import cv2
from scipy.ndimage import gaussian_filter

f_ref = h5py.File('nisarB_data/172_008/NSR_172_008_20251204T024618_HH.h5', 'r')['science/LSAR/RSLC']
f_rep = h5py.File('nisarB_data/172_008/NSR_172_008_20251122T024618_HH.h5', 'r')['science/LSAR/RSLC']

# --- Read metadata ---
prf = 1.0 / float(f_ref['swaths/zeroDopplerTimeSpacing'][()])
range_spacing = float(f_ref['swaths/frequencyB/slantRangeSpacing'][()])
range_start = float(f_ref['swaths/frequencyB/slantRange'][0])
time_start_ref = float(f_ref['swaths/zeroDopplerTime'][0])
ny_ref, nx_ref = f_ref['swaths/frequencyB/HH'].shape
print(f'SLC: ({ny_ref},{nx_ref}), PRF={prf:.0f}Hz, dr={range_spacing:.2f}m')

# --- Coregistration: match geolocation grids to find azimuth/range offset ---
def read_geoloc(f):
    geoloc = f['metadata/geolocationGrid']
    return (geoloc['coordinateY'][0].astype(np.float64),
            geoloc['slantRange'][:].astype(np.float64),
            geoloc['zeroDopplerTime'][:].astype(np.float64))

lat_ref, sr_ref, time_ref = read_geoloc(f_ref)
lat_rep, sr_rep, time_rep = read_geoloc(f_rep)
time_start_rep = float(f_rep['swaths/zeroDopplerTime'][0])
range_start_rep = float(f_rep['swaths/frequencyB/slantRange'][0])
ny_rep = f_rep['swaths/frequencyB/HH'].shape[0]

azi_pixels_ref = (time_ref - time_start_ref) * prf
rng_pixels_ref = (sr_ref - range_start) / range_spacing

# find azimuth offset: where does the ref center latitude fall in the rep grid?
mid_row = lat_ref.shape[0] // 2
# interpolate latitude to find mean azimuth offset (average across range columns)
rep_indices = np.arange(lat_rep.shape[0], dtype=np.float64)
offsets = []
for col in range(0, lat_ref.shape[1], 10):  # every 10th column
    lat_col = lat_rep[:, col]
    sort = -1 if lat_col[0] > lat_col[-1] else 1
    idx = np.interp(lat_ref[mid_row, col], lat_col[::sort], rep_indices[::sort])
    offsets.append((np.interp(idx, rep_indices, time_rep) - time_start_rep) * prf - azi_pixels_ref[mid_row])
mean_offset_azi = float(np.nanmean(offsets))
mean_offset_rng = float(np.mean((sr_rep - range_start_rep) / range_spacing - rng_pixels_ref))
print(f'  geoloc offset azi={mean_offset_azi:.1f}  rng={mean_offset_rng:.1f}')

# --- Refine offset with cv2.phaseCorrelate on SLC amplitude ---
half = 512
cy, cx = ny_ref // 2, nx_ref // 2
int_da, int_dr = int(round(mean_offset_azi)), int(round(mean_offset_rng))
hann = np.outer(np.hanning(half*2), np.hanning(half*2)).astype(np.float32)
amp_ref = np.abs(f_ref['swaths/frequencyB/HH'][cy-half:cy+half, cx-half:cx+half]
                 .astype(np.complex64)).astype(np.float32)
amp_rep = np.abs(f_rep['swaths/frequencyB/HH'][cy+int_da-half:cy+int_da+half, cx+int_dr-half:cx+int_dr+half]
                 .astype(np.complex64)).astype(np.float32)
amp_ref = (amp_ref - amp_ref.mean()) / (amp_ref.std() + 1e-10) * hann
amp_rep = (amp_rep - amp_rep.mean()) / (amp_rep.std() + 1e-10) * hann
(dx, dy), _ = cv2.phaseCorrelate(amp_ref, amp_rep)
# compensate for int() truncation: phaseCorrelate "finds" the fractional part we lost
frac_a, frac_r = mean_offset_azi - int_da, mean_offset_rng - int_dr
offset_azi = int_da + dy - frac_a
offset_rng = int_dr + dx - frac_r
print(f'  refined offset azi={offset_azi:.1f}  rng={offset_rng:.1f}')

# --- Read reference SLC and resample secondary via cv2.remap (Lanczos) ---
print('Reading and resampling SLCs...')
ref_slc = f_ref['swaths/frequencyB/HH'][:].astype(np.complex64)
rep_raw = f_rep['swaths/frequencyB/HH'][:].astype(np.complex64)
# cv2.remap requires dims < 32767 — remap each half with its own source slab
rep_slc = np.empty_like(ref_slc)
for a0, a1 in [(0, ny_ref // 2), (ny_ref // 2, ny_ref)]:
    # source slab covering the mapped range + 8 pixels for Lanczos kernel
    s0 = max(0, int(a0 + offset_azi) - 8)
    s1 = min(rep_raw.shape[0], int(a1 + offset_azi) + 8)
    slab = rep_raw[s0:s1]
    map_rng = np.tile(np.arange(nx_ref, dtype=np.float32) + np.float32(offset_rng), (a1-a0, 1))
    map_azi = np.tile((np.arange(a0, a1, dtype=np.float32) + np.float32(offset_azi - s0))[:, None], (1, nx_ref))
    rep_slc[a0:a1] = cv2.remap(slab.real, map_rng, map_azi, cv2.INTER_LANCZOS4) + \
                   1j*cv2.remap(slab.imag, map_rng, map_azi, cv2.INTER_LANCZOS4)

# --- Crop to azimuth overlap and compute interferogram + coherence ---
row_start = max(0, int(np.ceil(-offset_azi)))
row_end = min(ny_ref, int(np.floor(ny_rep - offset_azi)))
ref_slc, rep_slc = ref_slc[row_start:row_end], rep_slc[row_start:row_end]

def multilook(data, looks = [10, 2]):
	# ~50m × 50m multilook at 5m azi / 25m rng spacing
    ny = (data.shape[0] // looks[0]) * looks[0]
    nx = (data.shape[1] // looks[1]) * looks[1]
    return data[:ny, :nx].reshape(ny//looks[0], looks[0], nx//looks[1], looks[1]).mean(axis=(1, 3)) + 1e-30

intf = multilook(ref_slc * np.conj(rep_slc))
corr = np.abs(intf) / np.sqrt(multilook(np.abs(ref_slc)**2) * multilook(np.abs(rep_slc)**2))
print(f'Coherence: mean={corr.mean():.3f}  median={np.median(corr):.3f}  shape={intf.shape}')

# --- Remove flat-earth range fringes (phase gradient + polyfit) ---
grad = np.angle(intf[:, 1:] * np.conj(intf[:, :-1])).mean(axis=0)
coeffs = np.polyfit(np.arange(len(grad)), grad, 1)
range_full = np.arange(intf.shape[1], dtype=np.float64)
intf_detrended = intf * np.exp(-1j * (coeffs[1]*range_full + coeffs[0]*range_full**2/2)[None, :])
print(f'  detrend range: coeffs={coeffs}')

# --- Unwrap: Gaussian smooth + multilook + horizontal-split np.unwrap ---
def gaussian(data, sigma = [40, 8]):
	# ~200m Gaussian at 5m azi / 25m rng spacing
	return gaussian_filter(data, sigma)

intf_smooth = multilook(gaussian(intf_detrended.real) + 1j*gaussian(intf_detrended.imag))
corr_smooth = multilook(gaussian(corr)) > 0.25

# unwrap each half from center outward (avoids near-range edge artifacts)
mid = intf_smooth.shape[0] // 2
bottom = np.unwrap(np.unwrap(np.angle(intf_smooth[mid:]), axis=1), axis=0)
top = np.unwrap(np.unwrap(np.angle(intf_smooth[:mid][::-1]), axis=1), axis=0)[::-1]
bottom[~corr_smooth[mid:]] = top[~corr_smooth[:mid]] = np.nan
unwrapped = np.concatenate([top + np.nanmedian(bottom[0]) - np.nanmedian(top[-1]), bottom])

# remove residual quadratic range trend
coeffs = np.polyfit(np.where(np.isfinite(unwrapped))[1], unwrapped[np.isfinite(unwrapped)], 2)
unwrapped -= np.polyval(coeffs, np.arange(unwrapped.shape[1]))[None, :]

# --- Plot: coherence, wrapped phase, unwrapped phase (north-up) ---
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
kw = dict(aspect='auto', interpolation='nearest')
axes[0].imshow(corr[::-1, ::-1], cmap='gray', vmin=0, vmax=1, **kw)
axes[0].set_title(f'Coherence (mean={corr.mean():.2f})', fontsize=16)
axes[1].imshow(np.angle(intf[::-1, ::-1]), cmap='gist_rainbow_r', vmin=-np.pi, vmax=np.pi, **kw)
axes[1].set_title('Phase (before detrend)', fontsize=16)
axes[2].imshow(unwrapped[::-1, ::-1], cmap='twilight', vmin=-10, vmax=15, **kw)
axes[2].set_title('Unwrapped phase', fontsize=16)
for ax in axes: ax.axis('off')
fig.tight_layout()
plt.savefig('nisar.png', dpi=150)
plt.show()
