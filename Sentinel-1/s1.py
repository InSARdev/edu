# ----------------------------------------------------------------------------
# This file is part of the InSARdev project: https://github.com/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
# ----------------------------------------------------------------------------
"""
Sentinel-1 TOPS Interferogram Processor — Educational Version
==============================================================
Forms a wrapped interferogram from two S1 IW burst SLCs in radar coordinates.
Demonstrates TOPS deramp, differential reramp, amplitude cross-correlation
coregistration, and spherical flat-earth removal.

Dependencies: numpy, scipy, tifffile, cv2, ortools, matplotlib
Lines: 243 code + 10 imports + 12 prints + 12 plot + 26 comments + 7 docstrings + 21 empty = 331
"""
import numpy as np
import cv2
import tifffile
import xml.etree.ElementTree as ET
from datetime import datetime
from scipy.constants import speed_of_light
from scipy.ndimage import gaussian_filter

datadir = 'data/123_262886_IW2'
ref_xml  = f'{datadir}/annotation/S1_262886_IW2_20190702T032455_VV_69C5-BURST.xml'
rep_xml  = f'{datadir}/annotation/S1_262886_IW2_20190708T032537_VV_33CA-BURST.xml'
ref_tiff = f'{datadir}/measurement/S1_262886_IW2_20190702T032455_VV_69C5-BURST.tiff'
rep_tiff = f'{datadir}/measurement/S1_262886_IW2_20190708T032537_VV_33CA-BURST.tiff'

# --- TOPS deramp (from annotation XML + TIFF) ---
def deramped_burst(xml_path, tiff_path):
    """Deramp a S1 TOPS burst. The TOPS azimuth chirp phase is range-constant
    to first order (FM rate and Doppler centroid evaluated at center range),
    so the deramp reduces to a 1D azimuth-only phase multiply."""
    root = ET.parse(xml_path).getroot()
    radar_freq = float(root.find('.//radarFrequency').text)
    rng_samp_rate = float(root.find('.//rangeSamplingRate').text)
    near_range = float(root.find('.//imageInformation/slantRangeTime').text) * speed_of_light / 2
    nx = int(root.find('.//numberOfSamples').text)
    dta = float(root.find('.//imageInformation/azimuthTimeInterval').text)
    # steering rate: ks = 2 * v * f * kpsi / speed_of_light
    kpsi = np.pi * float(root.find('.//azimuthSteeringRate').text) / 180.0
    # satellite velocity from nearest orbit record
    t0 = datetime.fromisoformat(root.find('.//productFirstLineUtcTime').text)
    t1 = datetime.fromisoformat(root.find('.//productLastLineUtcTime').text)
    t_mid = t0 + (t1 - t0) / 2
    best_dt = 1e9
    for osv in root.findall('.//orbitList/orbit'):
        dt = abs((datetime.fromisoformat(osv.find('time').text) - t_mid).total_seconds())
        if dt < best_dt:
            best_dt = dt
            v = osv.find('velocity')
            vel = np.array([float(v.find('x').text), float(v.find('y').text), float(v.find('z').text)])
    ks = 2.0 * np.linalg.norm(vel) * radar_freq * kpsi / speed_of_light
    # FM rate and Doppler centroid at center range (c0 coefficients only)
    def parse_sec(s):
        dt = datetime.fromisoformat(s)
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    burst = root.find('.//swathTiming/burstList/burst')
    lpb = int(root.find('.//numberOfLines').text)
    t_brst = parse_sec(burst.find('azimuthTime').text) + dta * lpb / 2.0
    best_fm, best_d = None, 1e9
    for fm in root.findall('.//azimuthFmRateList/azimuthFmRate'):
        d = abs(parse_sec(fm.find('azimuthTime').text) - t_brst)
        if d < best_d: best_d, best_fm = d, fm
    fka_el = best_fm.find('azimuthFmRatePolynomial')
    fka0 = float(fka_el.text.split()[0]) if fka_el is not None else float(best_fm.find('c0').text)
    best_dc, best_d = None, 1e9
    for dc in root.findall('.//dcEstimateList/dcEstimate'):
        d = abs(parse_sec(dc.find('azimuthTime').text) - t_brst)
        if d < best_d: best_d, best_dc = d, dc
    fnc0 = float(best_dc.find('dataDcPolynomial').text.split()[0])
    # valid line range from firstValidSample
    fvs = [int(x) for x in burst.find('firstValidSample').text.split()]
    k_start = next(j for j, f in enumerate(fvs) if f >= 0)
    k_end = len(fvs) - 1 - next(j for j, f in enumerate(reversed(fvs)) if f >= 0)
    n_valid = (k_end - k_start) // 4 * 4
    # deramp: 1D azimuth phase (range-constant at center-range FM rate)
    kt0 = fka0 * ks / (fka0 - ks)
    eta = (np.arange(lpb) - lpb / 2.0 + 0.5) * dta
    phase = -np.pi * kt0 * eta**2 - 2*np.pi * fnc0 * eta
    slc_raw = tifffile.imread(tiff_path).astype(np.complex64)
    slc = (slc_raw[:lpb, :nx] * np.exp(1j * phase)[:, None]).astype(np.complex64)
    slc = slc[k_start:k_start + n_valid]
    rp = dict(fka0=fka0, fnc0=fnc0, ks=ks, dta=dta, lpb=lpb, k_start=k_start)
    params = dict(near_range=near_range, rng_samp_rate=rng_samp_rate,
                  radar_wavelength=speed_of_light / radar_freq)
    return params, slc, slc_raw, rp

print('Deramping...')
prm_ref, slc_ref, slc_ref_raw, rp_ref = deramped_burst(ref_xml, ref_tiff)
prm_rep, slc_rep, slc_rep_raw, rp_rep = deramped_burst(rep_xml, rep_tiff)
print(f'  Ref: {slc_ref.shape}, Rep: {slc_rep.shape}')

# --- Amplitude cross-correlation coregistration (grid + bilinear fit) ---
print('Coregistering...')
ny, nx = slc_ref.shape
range_offset = int(round((prm_ref['near_range'] - prm_rep['near_range']) / (speed_of_light / (2 * prm_ref['rng_samp_rate']))))
half = 256
hann = np.outer(np.hanning(half*2), np.hanning(half*2)).astype(np.float32)
obs = []
for cy in np.linspace(half, ny - half - 1, 6).astype(int):
    for cx in np.linspace(half, nx - half - 1, 12).astype(int):
        cx2 = cx + range_offset
        if cx2 - half < 0 or cx2 + half > slc_rep_raw.shape[1]: continue
        a1 = np.abs(slc_ref_raw[cy-half:cy+half, cx-half:cx+half]).astype(np.float32)
        a2 = np.abs(slc_rep_raw[cy-half:cy+half, cx2-half:cx2+half]).astype(np.float32)
        a1 = (a1 - a1.mean()) / (a1.std() + 1e-10) * hann
        a2 = (a2 - a2.mean()) / (a2.std() + 1e-10) * hann
        (dx, dy), response = cv2.phaseCorrelate(a1, a2)
        if response > 0.1 and abs(dy) < half//2 and abs(dx) < half//2:
            obs.append((cy, cx, dy, dx + range_offset))
obs = np.array(obs)
for _ in range(3):
    A = np.column_stack([np.ones(len(obs)), obs[:,1]/nx, obs[:,0]/ny])
    ca, _, _, _ = np.linalg.lstsq(A, obs[:,2], rcond=None)
    cr, _, _, _ = np.linalg.lstsq(A, obs[:,3], rcond=None)
    res_a, res_r = np.abs(obs[:,2] - A@ca), np.abs(obs[:,3] - A@cr)
    good = (res_a < 2*res_a.std()) & (res_r < 2*res_r.std())
    if good.all(): break
    obs = obs[good]
ca[0] += rp_ref['k_start'] - rp_rep['k_start']
print(f'  {len(obs)} patches, azi={ca[0]:.2f}, rng={cr[0]:.1f}')

# --- Resample deramped secondary via cv2.remap (Lanczos) ---
print('Resampling...')
a2d = np.arange(ny, dtype=np.float32)[:, None] * np.ones(nx, dtype=np.float32)[None, :]
r2d = np.ones(ny, dtype=np.float32)[:, None] * np.arange(nx, dtype=np.float32)[None, :]
map_a = (a2d + ca[0] + ca[1]*r2d/nx + ca[2]*a2d/ny).astype(np.float32)
map_r = (r2d + cr[0] + cr[1]*r2d/nx + cr[2]*a2d/ny).astype(np.float32)
rep_aligned = np.empty_like(slc_ref)
for a0, a1 in [(0, ny//2), (ny//2, ny)]:
    s0 = max(0, int(map_a[a0:a1].min()) - 8)
    s1 = min(slc_rep.shape[0], int(map_a[a0:a1].max()) + 8)
    slab = slc_rep[s0:s1]
    rep_aligned[a0:a1] = cv2.remap(slab.real.astype(np.float32), map_r[a0:a1],
        (map_a[a0:a1] - s0).astype(np.float32), cv2.INTER_LANCZOS4) + \
        1j*cv2.remap(slab.imag.astype(np.float32), map_r[a0:a1],
        (map_a[a0:a1] - s0).astype(np.float32), cv2.INTER_LANCZOS4)
del slc_ref_raw, slc_rep_raw, slc_rep

# --- Differential reramp (TOPS phase doesn't cancel between S1A/S1B) ---
print('Differential reramp...')
def reramp_phase(rp, azi_pix):
    """Simplified TOPS reramp: azimuth-only (range-constant FM rate and Doppler)."""
    kt0 = rp['fka0'] * rp['ks'] / (rp['fka0'] - rp['ks'])
    eta = (azi_pix - rp['lpb']/2 + 0.5 + rp['k_start']) * rp['dta']
    return np.pi * kt0 * eta**2 + 2*np.pi * rp['fnc0'] * eta

azi = np.arange(ny, dtype=np.float64)[:, None] * np.ones(nx)[None, :]
rng = np.ones(ny)[:, None] * np.arange(nx, dtype=np.float64)[None, :]
diff_reramp = reramp_phase(rp_ref, azi) - reramp_phase(rp_rep,
    azi + ca[0] + ca[1]*rng/nx + ca[2]*azi/ny)

def multilook(data, looks_azi=3, looks_rng=12):
    ny2 = (data.shape[0] // looks_azi) * looks_azi
    nx2 = (data.shape[1] // looks_rng) * looks_rng
    return data[:ny2, :nx2].reshape(ny2//looks_azi, looks_azi, nx2//looks_rng, looks_rng).mean(axis=(1, 3)) + 1e-30

# --- Spherical flat-earth removal (from annotation XML orbits) ---
print('Flat-earth removal...')
def nearest_orbit(xml_path):
    """ECEF position/velocity from orbit record closest to burst center."""
    root = ET.parse(xml_path).getroot()
    t0 = datetime.fromisoformat(root.find('.//productFirstLineUtcTime').text)
    t1 = datetime.fromisoformat(root.find('.//productLastLineUtcTime').text)
    t_mid = t0 + (t1 - t0) / 2
    best_dt, pos, vel = 1e9, None, None
    for osv in root.findall('.//orbitList/orbit'):
        dt = abs((datetime.fromisoformat(osv.find('time').text) - t_mid).total_seconds())
        if dt < best_dt:
            best_dt = dt
            p, v = osv.find('position'), osv.find('velocity')
            pos = np.array([float(p.find('x').text), float(p.find('y').text), float(p.find('z').text)])
            vel = np.array([float(v.find('x').text), float(v.find('y').text), float(v.find('z').text)])
    return pos, vel

pos_ref, vel_ref = nearest_orbit(ref_xml)
pos_rep, _       = nearest_orbit(rep_xml)
B_vec = pos_rep - pos_ref
n_along = vel_ref / np.linalg.norm(vel_ref)
n_up = pos_ref / np.linalg.norm(pos_ref)
n_cross = np.cross(n_along, n_up); n_cross /= np.linalg.norm(n_cross)
n_radial = np.cross(n_cross, n_along)
B = np.hypot(np.dot(B_vec, n_cross), np.dot(B_vec, n_radial))
alpha_b = np.arctan2(np.dot(B_vec, n_radial), np.dot(B_vec, n_cross))
root_ref = ET.parse(ref_xml).getroot()
lat = np.radians(np.mean([float(p.text) for p in root_ref.findall('.//geolocationGridPointList//latitude')]))
a_el = float(root_ref.find('.//ellipsoidSemiMajorAxis').text)
b_el = float(root_ref.find('.//ellipsoidSemiMinorAxis').text)
R_e = 1.0 / np.sqrt((np.cos(lat)/a_el)**2 + (np.sin(lat)/b_el)**2)
H = np.linalg.norm(pos_ref)
wvl = prm_ref['radar_wavelength']
drange = speed_of_light / (2 * prm_ref['rng_samp_rate'])
rho = prm_ref['near_range'] + np.arange(nx) * drange
theta = np.arccos(np.clip((rho**2 + H**2 - R_e**2) / (2*rho*H), -1, 1))
drho = np.sqrt(rho**2 + B**2 - 2*rho*B*np.sin(alpha_b - theta)) - rho
phase_flat = -4*np.pi/wvl * drho
print(f'  B={B:.1f}m, fringes={(phase_flat[-1]-phase_flat[0])/(2*np.pi):.0f}')

# --- Form interferogram with all corrections ---
intf = (rep_aligned * np.conj(slc_ref) * np.exp(-1j * (diff_reramp - phase_flat[None, :]))).astype(np.complex64)

intf_ml = multilook(intf)
corr = np.clip(np.abs(intf_ml) / np.sqrt(multilook(np.abs(slc_ref)**2) * multilook(np.abs(rep_aligned)**2)), 0, 1)
# constant azimuth detrend (residual TOPS misregistration)
ga = np.angle(intf_ml[1:] * np.conj(intf_ml[:-1])).mean()
intf_ml *= np.exp(-1j * ga * np.arange(intf_ml.shape[0], dtype=np.float64))[:, None]
print(f'Coherence: mean={corr.mean():.3f}  shape={intf_ml.shape}')

# --- Goldstein filter ---
def goldstein(data, psize=32, alpha=0.5):
    """Goldstein adaptive phase filter."""
    ny, nx = data.shape
    step = psize // 2
    out = np.zeros_like(data)
    wgt = np.zeros((ny, nx))
    for i in range(0, ny - psize + 1, step):
        for j in range(0, nx - psize + 1, step):
            F = np.fft.fft2(data[i:i+psize, j:j+psize])
            out[i:i+psize, j:j+psize] += np.fft.ifft2(F * np.abs(F)**alpha)
            wgt[i:i+psize, j:j+psize] += 1
    return np.where(wgt > 0, out / wgt, 0)

def gaussian(data, sigma=[1, 4]):
    return gaussian_filter(data, sigma)

print('Gaussian + Goldstein filtering...')
intf_gold = goldstein(gaussian(intf.real) + 1j*gaussian(intf.imag))

# --- Branch-cut unwrapping (max-flow/min-cut via OR-Tools) ---
def unwrap_maxflow(phase, corr=None, max_iter=25):
    """Branch-cut phase unwrapping via max-flow/min-cut. Default 3 iterations (~4s)."""
    from ortools.graph.python import max_flow
    pc = phase.ravel().astype(np.float64) / (2*np.pi)
    valid = ~np.isnan(pc)
    n, S = pc.size, np.int64(2**16 - 1)
    nd = np.arange(n).reshape(phase.shape)
    edges = np.vstack([np.c_[nd[:,:-1].ravel(), nd[:,1:].ravel()],
                       np.c_[nd[:-1].ravel(), nd[1:].ravel()]])
    edges = edges[valid[edges[:,0]] & valid[edges[:,1]]]
    i, j = edges.T
    cw = 1 / np.maximum(-np.log(np.maximum((corr.ravel()[i]+corr.ravel()[j])/2, 1e-10)), 0.1) if corr is not None else 1
    jumps = np.zeros(n + 2, np.int64)
    for _ in range(max_iter):
        r = (jumps[j] - jumps[i]) - (pc[i] - pc[j])
        eu, ed = np.abs(r + 1), np.abs(r - 1)
        w = np.maximum(0, eu + ed - 2*np.abs(r)) * cw
        d = (eu - np.abs(r)) * cw
        smf = max_flow.SimpleMaxFlow()
        smf.add_arcs_with_capacity(i.astype(np.int32), j.astype(np.int32), (S*w).astype(np.int64))
        smf.add_arcs_with_capacity(j.astype(np.int32), i.astype(np.int32), np.zeros(len(w), np.int64))
        ws, wt = np.zeros(n), np.zeros(n)
        np.add.at(ws, i, np.maximum(0, d)); np.add.at(wt, i, np.maximum(0, -d))
        np.add.at(ws, j, np.maximum(0, -d)); np.add.at(wt, j, np.maximum(0, d))
        ix = np.arange(n, dtype=np.int32)
        smf.add_arcs_with_capacity(np.full(n, n, np.int32), ix, (S*ws).astype(np.int64))
        smf.add_arcs_with_capacity(ix, np.full(n, n+1, np.int32), (S*wt).astype(np.int64))
        if smf.solve(n, n+1) != max_flow.SimpleMaxFlow.OPTIMAL: break
        jumps[smf.get_source_side_min_cut()] += 1
    out = (phase.ravel() + jumps[:-2] * 2*np.pi).astype(np.float32).reshape(phase.shape)
    out[np.isnan(phase)] = np.nan
    return out

print('Unwrapping...')
intf_gold_ml = multilook(intf_gold)
phase_in = np.angle(intf_gold_ml)
unwrapped = unwrap_maxflow(phase_in, corr)
valid = np.isfinite(unwrapped)
if valid.sum() > 10:
    coeffs = np.polyfit(np.where(valid)[1], unwrapped[valid], 2)
    unwrapped -= np.polyval(coeffs, np.arange(unwrapped.shape[1]))[None, :]

# --- Geocoding (simplified: annotation XML geolocation grid → regular lat/lon grid) ---
# Note: production geocoding uses precise orbit geometry (radar-to-ground projection
# via satellite position, velocity, and DEM). Here we use the pre-computed geolocation
# tie-point grid from the annotation XML, which ESA derived from the same orbit geometry.
print('Geocoding...')
from scipy.interpolate import griddata
root_ref = ET.parse(ref_xml).getroot()
gla, glo, gl, gp = [], [], [], []
for g in root_ref.findall('.//geolocationGridPointList/geolocationGridPoint'):
    gl.append(int(g.find('line').text)); gp.append(int(g.find('pixel').text))
    gla.append(float(g.find('latitude').text)); glo.append(float(g.find('longitude').text))
ul, up = np.unique(gl), np.unique(gp)
lat_g = np.array(gla).reshape(len(ul), len(up))
lon_g = np.array(glo).reshape(len(ul), len(up))
la, lr = ny // corr.shape[0], nx // corr.shape[1]
ml_y, ml_x = np.arange(corr.shape[0]) * la + la/2, np.arange(corr.shape[1]) * lr + lr/2
f = (ml_y - ul[0]) / (ul[1] - ul[0])
lat = (1-f)[:,None]*np.interp(ml_x, up, lat_g[0]) + f[:,None]*np.interp(ml_x, up, lat_g[1])
lon = (1-f)[:,None]*np.interp(ml_x, up, lon_g[0]) + f[:,None]*np.interp(ml_x, up, lon_g[1])
# inverse geocoding maps via griddata on subsampled points + cv2.remap
ds = 0.0005  # ~56m geo pixel
geo_lat = np.arange(lat.max(), lat.min(), -ds)
geo_lon = np.arange(lon.min(), lon.max(), ds)
geo_lat_2d, geo_lon_2d = np.meshgrid(geo_lat, geo_lon, indexing='ij')
step = 100
sub = (slice(None, None, step), slice(None, None, step))
pts = np.c_[lat[sub].ravel(), lon[sub].ravel()]
rows = np.arange(corr.shape[0], dtype=np.float32)[:,None] * np.ones(corr.shape[1])[None,:]
cols = np.ones(corr.shape[0])[:,None] * np.arange(corr.shape[1], dtype=np.float32)[None,:]
map_row = np.nan_to_num(griddata(pts, rows[sub].ravel(), (geo_lat_2d, geo_lon_2d), method='linear'), nan=-1).astype(np.float32)
map_col = np.nan_to_num(griddata(pts, cols[sub].ravel(), (geo_lat_2d, geo_lon_2d), method='linear'), nan=-1).astype(np.float32)

def geocode(data):
    return cv2.remap(data.astype(np.float32), map_col, map_row, cv2.INTER_LINEAR, borderValue=float('nan'))

# --- Plot ---
import matplotlib.pyplot as plt
geo_corr = geocode(corr)
geo_phase = np.angle(geocode(intf_gold_ml.real) + 1j*geocode(intf_gold_ml.imag))  # geocode complex, then angle
geo_unwrap = geocode(unwrapped)
# crop to data extent (remove NaN border)
valid_mask = np.isfinite(geo_corr)
rows_valid = np.where(valid_mask.any(axis=1))[0]
cols_valid = np.where(valid_mask.any(axis=0))[0]
s = (slice(rows_valid[0], rows_valid[-1]+1), slice(cols_valid[0], cols_valid[-1]+1))
geo_corr, geo_phase, geo_unwrap = geo_corr[s], geo_phase[s], geo_unwrap[s]
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
kw = dict(aspect='auto', interpolation='nearest')
axes[0].imshow(geo_corr, cmap='gray', vmin=0.1, vmax=0.8, **kw)
axes[0].set_title(f'Coherence ({corr.mean():.2f})', fontsize=16)
axes[1].imshow(geo_phase, cmap='gist_rainbow_r', vmin=-np.pi, vmax=np.pi, **kw)
axes[1].set_title('Multilook Phase (Goldstein)', fontsize=16)
axes[2].imshow(geo_unwrap, cmap='turbo', **kw)
axes[2].set_title('Unwrapped Phase (branch-cut)', fontsize=16)
for ax in axes: ax.axis('off')
fig.tight_layout()
plt.savefig('s1.png', dpi=150)
plt.show()
