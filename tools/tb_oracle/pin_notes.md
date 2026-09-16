
## 2026-07-27 — dftb2-dftbplus-energy-pin (H2)

- Binary: DFTB+ **24.1** x86_64-linux release tarball (not built from 25.1; GF 14
  build of 25.1 hit a Fortran field mismatch on this host).
- Parameters: **mio-1-1** `H-H.skf` (dftbparams/mio).
- Geometry: H–H = 0.74 Å, cluster, SCC on, tol 1e-8.
- Total energy: −0.6749509324 Ha → −18.36635052 eV (×27.211386245988).
- ATV matches via transcribed static tables + DFTB+ “new” SK interpolation +
  spline repulsive (see `basis_data/dftb_mio_hh_sk.h`).
- Wall (DFTB+ cold): ~0.15 s; ATV measured in REFERENCE after build.

## 2026-07-29 — am1-water-mopac-energy (H2O, NDDO/AM1)

- Oracle: **OpenMOPAC**, MOPAC **7.00** (public domain) — `mopac` is not
  installed on this host, so the pin is an *archived transcript*, not a run
  here. Twin: `inputs/mopac/am1-water/` (first `inputs/mopac/` entry).
- Keywords `AM1 PRECISE EF T=48H`. Geometry convention: pinned at the
  **AM1-optimized** structure printed by the same run
  (r(O–H) = 0.961267 Å, ∠HOH = 103.53103°) — a published AM1 ΔHf belongs to the
  optimized structure and may not be attached to an arbitrary geometry.
- Pins: TOTAL ENERGY −348.56254 eV; FINAL HEAT OF FORMATION −59.24079 kcal/mol;
  net charges O −0.3827 / H +0.1914 e; `CYCLE 1 HEAT` −57.64871 kcal/mol at the
  un-optimized input geometry (1.0 Å, 105°).
- ATV: −348.56316283 eV / −59.255061 kcal/mol / −0.38271 / +0.19135 e;
  −57.664384 kcal/mol at the input geometry. Bit-identical at 1 and 8 threads.
- **Residual traced, not absorbed:** 6.2e-4 eV. Rebuilding ATV with MOPAC's
  frozen constants (a0 = 0.529167, 1 Ha = 27.21 eV) in place of CODATA-2018
  moves it to −348.56243 eV → residual 1.1e-4 eV. **82 % of the gap is the
  physical-constant convention**, not a defect. ATV keeps CODATA.
- Discrimination: RM1 (same functional form, reparameterized) at the same
  geometry is −345.55785 eV / −38.80 kcal/mol — 1500 energy tolerances away.
- Wall (ATV, 1 thread, whole case): 0.01 s. MOPAC wall not measurable here.
