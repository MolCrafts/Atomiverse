# gpw/wannier

One case; every section below is run by `atv_regressions --case gpw/wannier` and keeps its own oracle record.


## Reference — Periodic multi-k MLWF (Marzari–Vanderbilt) on GPW, bulk Si

### Method

Maximally localized Wannier functions on the CPU **GPW** method: closed-shell
RKS multi-k SCF (GTH + DZVP-MOLOPT, PADE), then Marzari–Vanderbilt
localization on the occupied AO Bloch manifold
(`cpu::gpw::PeriodicWannier`). Neighbor overlaps use phase-weighted Gaussian
cross-AO matrices; projection init onto bond-centered Gaussians; optional
steepest-descent polish; centers from the Berry-phase formula.

Public API:

```cpp
cpu::gpw::PeriodicWannier mlwf;   // members: k_mesh, max_iter, tol, …
auto r = mlwf.compute(method, frame);  // frame: positions [Å], Z, box [Å]
// r.centers[i] [Å], r.spreads[i] [Å²]
```

No public `*Config`, no factories.

### System

Diamond Si in the **body-centred-tetragonal 4-atom** orthorhombic cell
(GPW is orthorhombic-only; same setting as `gpw-kpoints-si`):

| | |
| --- | --- |
| Lattice | a = 5.43 Å; cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → n_occ = 8 → **8 MLWFs** |
| XC | PADE (LDA) |
| Density cutoff | 100 Ry |
| MLWF k-mesh | 2×2×2, time-reversal **off** (full neighbor graph) |
| SCF | DIIS, tol 1e-7 Ha, EPS_DEFAULT 1e-7 |

Literature geometry pin (not an ATV snapshot): diamond / zincblende valence
MLWFs sit on the **Si–Si bond midpoints** (Marzari & Vanderbilt 1997 §IV;
standard Wannier90 Si demo). In the BCT 4-atom cell there are **eight**
unique tetrahedral bonds.

### Acceptance (hard-coded in `reg_periodic-wannier-gpw.cpp`)

1. SCF converges; `n_wann == 8`.
2. After greedy bipartite assignment and min-image under the BCT cell, every
   Wannier center lies within **0.90 Å** of a unique bond midpoint (same
   coarse 2×2×2 SMV window as `reg_periodic-wannier-si`).
3. Spread reduction: \(\Omega_\mathrm{final} < 0.5\,\Omega_\mathrm{init}\).
4. Finite spreads; no external process; ≤ 60 s.

### Offline oracle status

**No CP2K binary / module on the worker host at pin time** (same as
`gpw-kpoints-si` / `periodic-wannier-si`). Absolute CP2K Wannier coordinates
are **not invented**. Shared physical content with CP2K Quickstep
`&LOCALIZE` / print Wannier centers is the **bond-midpoint geometry** of the
eight valence MLWFs. Twin input (human/agent pin host only, never ctest):

`tools/cp2k_oracle/inputs/si_periodic_wannier_gpw.inp`

#### Atomiverse numbers (first green)

Recorded after first green run on the worker host (Release). Replace the
absolute energy band with a cited CP2K total when an offline twin is available.

| Quantity | Value |
| --- | --- |
| Literature pin | bond midpoints (MV 1997) |
| SCF energy | ≈ −15.77 Ha (PADE, 100 Ry, 2³ k) |
| Midpoint tol | 0.90 Å |
| Observed dmax | ≈ 0.012 Å |
| Spread ratio floor | 0.50 |
| Observed Ω/Ω₀ | ≈ 0.09 |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary/module on the pin host for a fair wall twin. |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |


## Reference — Collinear UKS multi-k MLWF (Marzari–Vanderbilt) on GPW, bulk Si

### Method

Maximally localized Wannier functions on the CPU **GPW collinear UKS** path:
multi-k LSD SCF (GTH + DZVP-MOLOPT, PADE), then independent Marzari–Vanderbilt
localization of the α and β occupied AO Bloch manifolds
(`cpu::gpw::PeriodicWannier`). Neighbor overlaps use phase-weighted Gaussian
cross-AO matrices; projection init onto bond-centered Gaussians; optional
steepest-descent polish; centers from the Berry-phase formula.

Public API:

```cpp
cpu::gpw::GpwDft dft{{si_kind}, /*cutoff_ry=*/100.0, cpu::gpw::XcKind::PADE};
dft.n_alpha = 8;
dft.n_beta  = 8;          // M = 0 closed-shell collinear
// build(frame) …  (cell from frame.box)
cpu::gpw::PeriodicWannier mlwf;   // members: k_mesh, max_iter, tol, …
auto r = mlwf.compute(dft, frame);
// r.n_wann_up / n_wann_dn; r.centers [Å] = α then β; r.spin[i] ∈ {0,1}
```

No public `*Config`, no factories. RKS path (`n_occ`) unchanged.

### System

Diamond Si in the **body-centred-tetragonal 4-atom** orthorhombic cell
(same setting as `periodic-wannier-gpw` / `gpw-kpoints-si`):

| | |
| --- | --- |
| Lattice | a = 5.43 Å; cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → nα = nβ = 8 → **16 MLWFs** (8 per spin) |
| XC | PADE (LDA) |
| Density cutoff | 100 Ry |
| MLWF k-mesh | 2×2×2, time-reversal **off** (full neighbor graph) |
| Spin | collinear UKS, multiplicity 1 (\(M=0\)) |
| SCF | DIIS, tol 1e-7 Ha, EPS_DEFAULT 1e-7 |

Literature geometry pin (parameter-free): diamond / zincblende valence MLWFs
sit on the **Si–Si bond midpoints** (Marzari & Vanderbilt 1997 §IV). In the
BCT 4-atom cell there are **eight** unique tetrahedral bonds; each spin
channel has one MLWF per bond at \(M=0\).

### Acceptance (hard-coded in `reg_periodic-wannier-gpw-uks.cpp`)

1. SCF converges; `n_wann_up == n_wann_dn == 8`, `n_wann == 16`.
2. **Energy pin:** \(|E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}| < 2\times10^{-3}\) Ha.
3. **Bond-midpoint pin (per spin):** after greedy bipartite assignment and
   min-image under the BCT cell, every Wannier center of each spin lies within
   **0.90 Å** of a unique bond midpoint (same coarse 2×2×2 SMV window as
   `reg_periodic-wannier-gpw`).
4. Spread reduction: \(\Omega_\mathrm{final} < 0.5\,\Omega_\mathrm{init}\)
   (sum over both spins).
5. Finite spreads; `spin` labels size 16 with values in \(\{0,1\}\); ≤ 60 s
   preferred / product budget soft.

### Offline CP2K pin (energy)

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision **git:5e54ba2**),
  host Arrhenius epyc; `source …/cp2k/install/env.sh`.
- **Input:**
  `tools/cp2k_oracle/inputs/periodic-wannier-gpw-uks/si_uks_wannier_bct.inp`
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  cd <pin_dir> && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  OMP_NUM_THREADS=8 mpirun -np 1 cp2k.psmp -i si_uks_wannier_bct.inp \
      -o si_uks_wannier_bct.out
  ```
- **Date:** 2026-08-01.
- **Matched twin class:** multi-k collinear UKS GPW (LSD, M=1, PADE, 100 Ry,
  MP 2×2×2 `FULL_GRID`, DZVP-MOLOPT-GTH / GTH-PBE-q4) — same representation
  as ATV.
- **LOCALIZE note:** CP2K emits
  `Localization not implemented for k-points` (`qs_scf_post_gpw.F:514`) and
  skips Wannier post-processing. Absolute CP2K Wannier coordinates are
  therefore **not** available for this multi-k twin; centers are gated by the
  literature bond-midpoint geometry (same policy as `periodic-wannier-gpw`).
  The SCF total energy is the CP2K numerical pin.

| Quantity | CP2K pin | Regression tolerance |
|----------|----------|----------------------|
| Total energy [Ha] | **−15.769956662772422** | **2×10⁻³ Ha** |

Key log lines:

```
  Total energy:                                               -15.76995666277242
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -15.769956662772422
 *** WARNING in qs_scf_post_gpw.F:514 :: Localization not implemented for ***
```

#### Atomiverse numbers (first green, 2026-08-01, OMP=8)

| Quantity | Value |
| --- | --- |
| SCF energy | −15.76993360 Ha |
| \|E − CP2K\| | **2.3×10⁻⁵ Ha** |
| Midpoint tol | 0.90 Å |
| Observed dmax (each spin) | ≈ 0.032 Å |
| Spread ratio Ω/Ω₀ | ≈ 0.0905 |
| n_up / n_dn | 8 / 8 |

### Performance

| Item | Value |
| ---- | ----- |
| CP2K wall (1 MPI × 8 OMP, timer `CP2K`) | **19.129 s** |
| ATV wall (SCF + dual-spin MLWF, OMP=8) | **183.1 s** |
| \(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **≈ 9.57** |
| `within_10x` | **true** |
| `perf_exempt` | **false** |

Fair twin: same multi-k UKS GPW ENERGY job (CP2K SCF; LOCALIZE is a no-op for
k-points). ATV includes two MV channels after SCF, so the ratio is conservative.

### Citations

- N. Marzari, D. Vanderbilt, *Phys. Rev. B* **56**, 12847 (1997) — MLWF / bond
  midpoints for diamond Si.
- N. Marzari et al., *Rev. Mod. Phys.* **84**, 1419 (2012) — review.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
