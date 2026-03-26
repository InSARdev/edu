# ----------------------------------------------------------------------------
# This file is part of the InSARdev project: https://github.com/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
# ----------------------------------------------------------------------------
"""
Minimal NISAR Interferogram Processor — Educational Version
============================================================
Forms a radar-coordinate interferogram from two NISAR RSLC HDF5 files.

Pipeline:
  1. Read SLC metadata and geolocation grids from HDF5
  2. Coregister via geolocation grid matching + coherence-based refinement
  3. Resample secondary SLC to reference grid (bilinear interpolation)
  4. Compute multilooked interferogram and coherence
  5. Remove flat-earth range fringes from complex interferogram (phase gradients)
  6. Unwrap: Gaussian smooth, multilook, horizontal-split scan-line unwrap
  7. Remove residual range trend from unwrapped phase (quadratic fit)

NISAR embeds a geolocation grid (lon/lat per radar pixel) in each HDF5 file.
Matching these grids between scenes gives coregistration offsets without orbit
mechanics or DEM.

Dependencies: numpy, h5py, matplotlib
Lines: 200 code + 15 plot + 15 comments + 34 docstrings + 35 empty = 299
"""
import numpy as np
import h5py
import os

SLC_PATH = 'science/LSAR/RSLC/swaths'
GEOLOC_PATH = 'science/LSAR/RSLC/metadata/geolocationGrid'

def read_meta(h5_path, freq='B', pol='HH'):
    """Read SLC metadata from NISAR RSLC HDF5."""
    with h5py.File(h5_path, 'r') as f:
        swath = f[f'{SLC_PATH}/frequency{freq}']
        return dict(
            shape = swath[pol].shape,
            prf   = 1.0 / float(f[f'{SLC_PATH}/zeroDopplerTimeSpacing'][()]),
            range_spacing = float(swath['slantRangeSpacing'][()]),
            wavelength    = 299_792_458.0 / float(swath['processedCenterFrequency'][()]),
            range_start   = float(swath['slantRange'][0]),
            time_start    = float(f[f'{SLC_PATH}/zeroDopplerTime'][0]),
        )

def read_geoloc(h5_path):
    """Read coarse geolocation grid (lat, slant range, zero-Doppler time)."""
    with h5py.File(h5_path, 'r') as f:
        geoloc = f[GEOLOC_PATH]
        return (geoloc['coordinateY'][0].astype(np.float64),     # lat (azi_grid, rng_grid)
                geoloc['slantRange'][:].astype(np.float64),       # range coords of grid
                geoloc['zeroDopplerTime'][:].astype(np.float64))  # azimuth coords of grid

def compute_offsets(h5_ref, h5_rep, freq='B', pol='HH'):
    """Compute constant coregistration offset from geolocation grids + coherence refinement."""
    meta_ref, meta_rep = read_meta(h5_ref, freq, pol), read_meta(h5_rep, freq, pol)
    lat_ref, slant_range_ref, time_ref = read_geoloc(h5_ref)
    lat_rep, slant_range_rep, time_rep = read_geoloc(h5_rep)

    # geolocation grid → SLC pixel positions
    azi_pixels_ref = (time_ref - meta_ref['time_start']) * meta_ref['prf']
    rng_pixels_ref = (slant_range_ref - meta_ref['range_start']) / meta_ref['range_spacing']
    n_azi_ref, n_rng = lat_ref.shape
    n_azi_rep = lat_rep.shape[0]

    # for each range column, interpolate along azimuth to find matching latitude
    offset_azi = np.empty((n_azi_ref, n_rng))
    offset_rng = np.empty((n_azi_ref, n_rng))
    rep_indices = np.arange(n_azi_rep, dtype=np.float64)
    for col in range(n_rng):
        lat_col = lat_rep[:, col]
        if lat_col[0] > lat_col[-1]:  # descending latitude
            matched = np.interp(lat_ref[:, col], lat_col[::-1], rep_indices[::-1])
        else:
            matched = np.interp(lat_ref[:, col], lat_col, rep_indices)
        time_matched = np.interp(matched, rep_indices, time_rep)
        offset_azi[:, col] = (time_matched - meta_rep['time_start']) * meta_rep['prf'] - azi_pixels_ref
        offset_rng[:, col] = (slant_range_rep[col] - meta_rep['range_start']) / meta_rep['range_spacing'] - rng_pixels_ref[col]

    # use mean offset (geoloc spatial variation is noisier than true offset)
    ny, nx = meta_ref['shape']
    mean_offset_azi = float(np.nanmean(offset_azi))
    mean_offset_rng = float(np.nanmean(offset_rng))
    print(f'  geoloc offset azi={mean_offset_azi:.1f}  rng={mean_offset_rng:.1f}')

    # refine with coherence-maximizing sub-pixel sweep
    offset_azi_map = np.full((ny, nx), mean_offset_azi, dtype=np.float32)
    offset_rng_map = np.full((ny, nx), mean_offset_rng, dtype=np.float32)
    corr_azi, corr_rng = _coherence_refine(
        h5_ref, h5_rep, freq, pol, offset_azi_map, offset_rng_map, meta_ref, meta_rep)
    print(f'  correction: azi={corr_azi:+.1f} rng={corr_rng:+.1f}')
    return offset_azi_map + corr_azi, offset_rng_map + corr_rng, meta_ref

def _coherence_refine(h5_ref, h5_rep, freq, pol, offset_azi, offset_rng,
                      meta_ref, meta_rep, patch_size=512, n_test=6):
    """Find constant (azi, rng) correction that maximizes coherence on test patches."""
    ny_ref, nx_ref = meta_ref['shape']
    ny_rep, nx_rep = meta_rep['shape']
    half = patch_size // 2
    looks_azi, looks_rng = 32, 8

    # test points within overlap region
    mean_offset = float(np.nanmean(offset_azi))
    azi_lo = max(half, int(-mean_offset) + half + 100)
    azi_hi = min(ny_ref - half, int(ny_rep - mean_offset) - half - 100)
    test_points = [(int(cy), nx_ref // 2)
                   for cy in np.linspace(azi_lo, azi_hi, n_test)]
    slc_path = f'{SLC_PATH}/frequency{freq}/{pol}'

    freq_azi = np.fft.fftfreq(patch_size).reshape(-1, 1)
    freq_rng = np.fft.fftfreq(patch_size).reshape(1, -1)
    crop_azi = (patch_size // looks_azi) * looks_azi
    crop_rng = (patch_size // looks_rng) * looks_rng

    with h5py.File(h5_ref, 'r') as file_ref, h5py.File(h5_rep, 'r') as file_rep:
        slc_ref, slc_rep = file_ref[slc_path], file_rep[slc_path]
        ref_patches = [slc_ref[cy-half:cy+half, cx-half:cx+half].astype(np.complex64)
                       for cy, cx in test_points]

        def sweep(azi_range, rng_range):
            best_coh, best_azi, best_rng = 0, 0.0, 0.0
            for try_azi in azi_range:
                for try_rng in rng_range:
                    coh_values = []
                    for k, (cy, cx) in enumerate(test_points):
                        total_azi = offset_azi[cy, cx] + try_azi
                        total_rng = offset_rng[cy, cx] + try_rng
                        int_azi, int_rng = int(round(total_azi)), int(round(total_rng))
                        cy2, cx2 = cy + int_azi, cx + int_rng
                        if cy2-half < 0 or cy2+half > ny_rep or cx2-half < 0 or cx2+half > nx_rep:
                            continue
                        patch_rep = slc_rep[cy2-half:cy2+half, cx2-half:cx2+half].astype(np.complex64)
                        frac_azi = total_azi - int_azi
                        frac_rng = total_rng - int_rng
                        # sub-pixel shift via FFT phase ramp (more accurate than bilinear for coherence)
                        if abs(frac_azi) > 0.01 or abs(frac_rng) > 0.01:
                            patch_rep = np.fft.ifft2(np.fft.fft2(patch_rep) *
                                np.exp(-2j*np.pi*(frac_azi*freq_azi + frac_rng*freq_rng)))
                        intf = ref_patches[k] * np.conj(patch_rep)
                        intf_ml = intf[:crop_azi,:crop_rng].reshape(
                            crop_azi//looks_azi, looks_azi, crop_rng//looks_rng, looks_rng).mean(axis=(1,3))
                        power_ref = (np.abs(ref_patches[k][:crop_azi,:crop_rng])**2).reshape(
                            crop_azi//looks_azi, looks_azi, crop_rng//looks_rng, looks_rng).mean(axis=(1,3))
                        power_rep = (np.abs(patch_rep[:crop_azi,:crop_rng])**2).reshape(
                            crop_azi//looks_azi, looks_azi, crop_rng//looks_rng, looks_rng).mean(axis=(1,3))
                        coh = np.abs(intf_ml) / np.sqrt(power_ref * power_rep + 1e-30)
                        coh_values.append(coh[(power_ref > 0) & (power_rep > 0)].mean())
                    if coh_values and np.mean(coh_values) > best_coh:
                        best_coh = np.mean(coh_values)
                        best_azi, best_rng = try_azi, try_rng
            return best_azi, best_rng

        # coarse sweep, then fine sweep around best
        best_azi, best_rng = sweep(np.arange(-2, 2.5, 0.5), np.arange(-2, 2.5, 0.5))
        best_azi, best_rng = sweep(np.arange(best_azi-0.5, best_azi+0.6, 0.1),
                                   np.arange(best_rng-0.5, best_rng+0.6, 0.1))
    return best_azi, best_rng

def resample_slc(h5_rep, offset_azi, offset_rng, freq, pol):
    """Resample secondary SLC to reference grid using constant offset (bilinear interpolation)."""
    ny, nx = offset_azi.shape
    with h5py.File(h5_rep, 'r') as f:
        slc = f[f'{SLC_PATH}/frequency{freq}/{pol}'][:].astype(np.complex64)
    ny_slc, nx_slc = slc.shape
    target_azi = np.arange(ny, dtype=np.float32).reshape(-1, 1) + offset_azi
    target_rng = np.arange(nx, dtype=np.float32).reshape(1, -1) + offset_rng
    azi_idx = np.floor(target_azi).astype(np.intp)
    rng_idx = np.floor(target_rng).astype(np.intp)
    frac_azi = (target_azi - azi_idx).astype(np.float32)
    frac_rng = (target_rng - rng_idx).astype(np.float32)
    valid = (azi_idx >= 0) & (azi_idx < ny_slc - 1) & (rng_idx >= 0) & (rng_idx < nx_slc - 1)
    out = np.zeros((ny, nx), dtype=np.complex64)
    # bilinear: weighted sum of 4 nearest SLC pixels
    out[valid] = ((1-frac_azi[valid])*(1-frac_rng[valid]) * slc[azi_idx[valid], rng_idx[valid]] +
                  (1-frac_azi[valid])*frac_rng[valid]     * slc[azi_idx[valid], rng_idx[valid]+1] +
                  frac_azi[valid]*(1-frac_rng[valid])     * slc[azi_idx[valid]+1, rng_idx[valid]] +
                  frac_azi[valid]*frac_rng[valid]         * slc[azi_idx[valid]+1, rng_idx[valid]+1])
    return out

def multilook(data, looks_azi, looks_rng):
    """Boxcar multilook: average looks_azi × looks_rng blocks."""
    ny = (data.shape[0] // looks_azi) * looks_azi
    nx = (data.shape[1] // looks_rng) * looks_rng
    return data[:ny, :nx].reshape(ny//looks_azi, looks_azi, nx//looks_rng, looks_rng).mean(axis=(1, 3))

def coherence(ref_slc, rep_slc, looks_azi, looks_rng):
    """Compute coherence and multilooked interferogram."""
    intf = multilook(ref_slc * np.conj(rep_slc), looks_azi, looks_rng)
    power_ref = multilook(np.abs(ref_slc)**2, looks_azi, looks_rng)
    power_rep = multilook(np.abs(rep_slc)**2, looks_azi, looks_rng)
    return np.abs(intf) / np.sqrt(power_ref * power_rep + 1e-30), intf

def detrend_phase(intf, degree=2):
    """Remove flat-earth range fringes via phase gradients (no unwrapping needed)."""
    result = intf.copy()
    # phase gradient between adjacent range pixels — always small, no wrapping
    grad = np.angle(result[:, 1:] * np.conj(result[:, :-1]))
    grad_profile = grad.mean(axis=0)  # average along azimuth → 1D range profile
    # fit polynomial to gradient, then integrate to get phase model
    range_pixels = np.arange(len(grad_profile), dtype=np.float64)
    design = np.column_stack([range_pixels**d for d in range(degree)])
    coeffs, _, _, _ = np.linalg.lstsq(design, grad_profile, rcond=None)
    range_full = np.arange(result.shape[1], dtype=np.float64)
    phase_model = sum(coeffs[d] * range_full**(d+1) / (d+1) for d in range(degree))
    result *= np.exp(-1j * phase_model[None, :]).astype(np.complex64)
    print(f'  detrend range: coeffs={np.array2string(coeffs, precision=4)}')
    return result

def unwrap2d(intf_detrended, corr, sigma_azi=40, sigma_rng=8, coh_thr=0.25):
    """Phase unwrapping: Gaussian smooth, multilook, horizontal-split np.unwrap,
    then remove residual quadratic range trend.
    Splits azimuth at center and unwraps outward to avoid near-range edge artifacts.
    sigma in pixels: ~200m at 5m azi / 25m rng spacing."""
    from numpy.fft import fft2, ifft2, fftfreq
    pad_azi, pad_rng = sigma_azi * 3, sigma_rng * 3
    def gauss_smooth(data):
        padded = np.pad(data, ((pad_azi, pad_azi), (pad_rng, pad_rng)), mode='edge')
        ky = fftfreq(padded.shape[0]).reshape(-1, 1)
        kx = fftfreq(padded.shape[1]).reshape(1, -1)
        kernel = np.exp(-2*np.pi**2 * (sigma_azi**2*ky**2 + sigma_rng**2*kx**2))
        return ifft2(fft2(padded) * kernel)[pad_azi:-pad_azi, pad_rng:-pad_rng]
    # Gaussian smooth then multilook (factor = sigma in each direction)
    mlook_azi, mlook_rng = sigma_azi, sigma_rng
    intf_smooth = multilook(gauss_smooth(intf_detrended), mlook_azi, mlook_rng)
    corr_smooth = multilook(gauss_smooth(corr), mlook_azi, mlook_rng)
    low_coh = corr_smooth < coh_thr
    mid = intf_smooth.shape[0] // 2
    # unwrap each half from center outward (avoids near-range edge artifacts)
    bottom = np.unwrap(np.unwrap(np.angle(intf_smooth[mid:]), axis=1), axis=0)
    top = np.unwrap(np.unwrap(np.angle(intf_smooth[:mid][::-1]), axis=1), axis=0)[::-1]
    bottom[low_coh[mid:]] = np.nan
    top[low_coh[:mid]] = np.nan
    top += np.nanmedian(bottom[0]) - np.nanmedian(top[-1])
    unwrapped = np.concatenate([top, bottom], axis=0)
    # remove residual quadratic range trend from unwrapped phase
    range_pixels = np.arange(unwrapped.shape[1], dtype=np.float64)
    valid = np.isfinite(unwrapped)
    valid_rng = range_pixels[np.where(valid)[1]]
    design = np.column_stack([np.ones(valid.sum()), valid_rng, valid_rng**2])
    coeffs, _, _, _ = np.linalg.lstsq(design, unwrapped[valid], rcond=None)
    unwrapped -= np.polyval(coeffs[::-1], range_pixels)[None, :]
    return unwrapped

def plot(corr, intf, unwrapped):
    """Plot coherence, wrapped phase, and unwrapped phase (north-up)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    kw = dict(aspect='auto', interpolation='nearest')
    axes[0].imshow(corr[::-1, ::-1], cmap='gray', vmin=0, vmax=1, **kw)
    axes[0].set_title(f'Coherence (mean={corr.mean():.2f})', fontsize=16)
    axes[1].imshow(np.angle(intf[::-1, ::-1]), cmap='gist_rainbow_r', vmin=-np.pi, vmax=np.pi, **kw)
    axes[1].set_title('Phase (before detrend)', fontsize=16)
    axes[2].imshow(unwrapped[::-1, ::-1], cmap='twilight', vmin=-10, vmax=15, **kw)
    axes[2].set_title('Unwrapped phase', fontsize=16)
    for ax in axes: ax.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nisar_numpy.png'), dpi=150)
    plt.show()

def main():
    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nisarB_data', '172_008')
    freq, pol = 'B', 'HH'
    looks_azi, looks_rng = 10, 2  # ~50m × 50m

    h5_files = sorted(f for f in os.listdir(datadir) if f.endswith('.h5') and pol in f)
    assert len(h5_files) >= 2, f'Need ≥2 {pol} scenes in {datadir}'
    # later date as reference (matches notebook's pairs([1,0]) convention)
    h5_ref, h5_rep = [os.path.join(datadir, f) for f in [h5_files[1], h5_files[0]]]
    print(f'Ref: {h5_files[1]}\nRep: {h5_files[0]}')

    meta = read_meta(h5_ref, freq, pol)
    print(f'SLC: {meta["shape"]}, λ={meta["wavelength"]:.4f}m, PRF={meta["prf"]:.0f}Hz, dr={meta["range_spacing"]:.2f}m')

    print('Computing offsets...')
    offset_azi, offset_rng, _ = compute_offsets(h5_ref, h5_rep, freq, pol)

    print('Reading reference SLC...')
    with h5py.File(h5_ref, 'r') as f:
        ref_slc = f[f'{SLC_PATH}/frequency{freq}/{pol}'][:].astype(np.complex64)

    print('Resampling secondary SLC...')
    rep_slc = resample_slc(h5_rep, offset_azi, offset_rng, freq, pol)

    # crop to azimuth overlap (scenes differ by ~1500 lines)
    ny_rep = read_meta(h5_rep, freq, pol)['shape'][0]
    row_start = max(0, int(np.ceil(-offset_azi.min())))
    row_end = min(meta['shape'][0], int(np.floor(ny_rep - offset_azi.max())))
    ref_slc, rep_slc = ref_slc[row_start:row_end], rep_slc[row_start:row_end]

    corr, intf = coherence(ref_slc, rep_slc, looks_azi, looks_rng)
    print(f'Coherence: mean={corr.mean():.3f}  median={np.median(corr):.3f}  shape={intf.shape}')
    del ref_slc, rep_slc

    print('Detrending...')
    intf_detrended = detrend_phase(intf)

    print('Unwrapping...')
    unwrapped = unwrap2d(intf_detrended, corr)

    plot(corr, intf, unwrapped)

if __name__ == '__main__':
    main()
