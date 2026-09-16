# Reference: `wannier-centers`

## Method

Foster–Boys localization (Foster & Boys, *Rev. Mod. Phys.* **32**, 300
(1960)) of occupied closed-shell RHF/STO-3G MOs via Jacobi pair rotations.
Public API: `cpu::scf::WannierCenters::compute(MeanFieldOutputs, Frame)`.

## Pins (no CP2K binary)

| System | Check | Source |
| ------ | ----- | ------ |
| H₂ (0.74 Å) | 1 center at bond midpoint \(z=0.37\) Å (±1e-4) | unit algebra / symmetry of \(\langle z\rangle\) for \(\sigma_g\) |
| H₂O | 5 centers; \(\Omega_\mathrm{tot}\) decreases vs canonical | Boys functional definition |
| H₂O | classic Boys geometry: O-core, 2 OH-bond, 2 rabbit-ear LP | textbook MLWF of water (e.g. Marzari et al. reviews; PySCF/CP2K qualitative) |
| H₂O | \(\sum_i x_i=\sum_i y_i=0\) (C₂ᵥ) | symmetry |

CP2K `LOCALIZE` numerical centers may be collected offline when a binary is
available; this regression is **`perf_exempt`** and does not require it.

## Geometry

H₂O same as `h2o_rhf_dipole` (Å):

```
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
```
