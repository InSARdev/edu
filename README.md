## InSAR.dev — Educational Examples

Minimal, self-contained InSAR processing examples for learning and teaching. Each example implements a complete interferometric pipeline from scratch using only basic scientific Python libraries.

<img src="assets/nisar_numpy.jpg" width="100%">

## Examples

### [NISAR/nisar.py](NISAR/nisar.py) — NISAR Unwrapped Interferogram in 76 Code Lines (38s)

A compact NISAR L-band interferogram processor and unwrapper using **numpy**, **h5py**, **scipy** and **opencv*** (+ matplotlib for plotting). Flat linear script, every step visible top-to-bottom with no function jumps. 76 code lines, 38s on Apple M4.

### [NISAR/nisar_numpy.py](NISAR/nisar_numpy.py) — NISAR Unwrapped Interferogram in 200 Code Lines (54s)

NISAR L-band interferogram processor and unwrapper in 200 lines of Python using only **numpy** and **h5py** (+ matplotlib for plotting). Pure signal processing in radar coordinates, 54s on Apple M4.

Both produce unwrapped interferograms matching NASA ASF product `NISAR_L2_PR_GUNW_005_172_A_008_006_2000_SH_20251122T024618_20251122T024652_20251204T024618_20251204T024653_X05007_N_F_J_001`.

**Data:** Uses NISAR RSLC HDF5 files downloaded by the notebook example [NISAR L-Band HH/HV RGB composite, HH interferogram, and unwrapped phase](https://github.com/InSARdev/core) from the [InSAR.dev](https://InSAR.dev) processing ecosystem.

## License

[MIT](LICENSE)
