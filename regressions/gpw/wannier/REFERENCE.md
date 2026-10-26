# gpw/wannier

One case; every section below is run by `atv_regressions --case gpw/wannier` and keeps its own oracle record.


The canonical batched exact Fourier-overlap path passed both existing
regression sections on 2026-10-25 (build12, Release, Apple M4,
OpenMP/OpenBLAS 8/8, profiling enabled). All numerical bounds, external
energy pins and the strict whole-section 60 s gate remain unchanged.

### Current native validation

| Section | Whole section | Shared 48 AO links | Channel localization | Maximum midpoint distance | Spread ratio |
| --- | --- | --- | --- | --- | --- |
| `periodic-wannier-gpw` | **7.889850 s** | 4.574 s | 0.027 s | 0.02245 Å | 0.1068 |
| `periodic-wannier-gpw-uks` | **9.886750 s** | 3.971 s | 0.025 / 0.021 s | 0.01726 Å per spin | 0.1041 |

The UKS energy is −15.76993360 Ha, with printed error 2.306724e-5 Ha
against the unchanged CP2K pin. Each case runs RKS then UKS in one process;
whole-section timers include construction, SCF, localization and assertions.
The AO matrices are computed once per call. Exactly equal short wavevectors
share each image-local Fourier integral before distinct ket Bloch phases are
accumulated; both spin contractions borrow the resulting operators.

Both sections still print that the MV loop did not flag converged. The
existing SCF, center/spread/energy and runtime gates pass; **a stationary MV
optimum is not established by these measurements**.

[Raw output](../../../tools/cp2k_oracle/inputs/periodic-wannier-gpw-uks/native-validation.txt),
[metadata/hashes](../../../tools/cp2k_oracle/inputs/periodic-wannier-gpw-uks/native-validation.json)
and [reproduction instructions](../../../tools/cp2k_oracle/inputs/periodic-wannier-gpw-uks/README.md)
preserve the current proof. Earlier exact-kernel timing failures are retained
as diagnostics: [build10](../../../tools/cp2k_oracle/inputs/periodic-wannier-gpw-uks/native-build10.txt)
and [build11](../../../tools/cp2k_oracle/inputs/periodic-wannier-gpw-uks/native-build11.txt).
The older native numbers and external timing records below remain historical;
this run supplies no refreshed fair ATV/CP2K timing ratio.

### Exact Gaussian Fourier overlaps

Berry and Wannier links share `prim_overlap_fourier` in the existing GPW
one-electron integral kernel. It evaluates

\[
F_{ab}(\mathbf q)=\int a(\mathbf r)b(\mathbf r)
  e^{-i\mathbf q\cdot\mathbf r}\,d^3r
= c_a c_b\left(\frac{\pi}{p}\right)^{3/2}
  e^{-|\mathbf q|^2/(4p)-i\mathbf q\cdot\mathbf P}
  \prod_{d=x,y,z}\sum_{t=0}^{l_{a,d}+l_{b,d}}
    E_t^{l_{a,d},l_{b,d}}(-i q_d)^t,
\]

where \(p=\alpha_a+\alpha_b\),
\(\mathbf P=(\alpha_a\mathbf A+\alpha_b\mathbf B)/p\), and the existing
McMurchie–Davidson coefficients \(E_t\) include the Gaussian product's
center-separation attenuation. The wavevector \(\mathbf q\) is the short,
unwrapped ket-minus-bra reciprocal step in Bohr⁻¹; primitive coordinates
are in Bohr and the Fourier sign is **negative**. Periodic image phases
retain the ket's Bloch convention and wrapped k labels.

Every angular term contributes. In particular a same-center s/pₓ product
has zero ordinary overlap but Fourier integral
\(-i q_x F_{ss}(\mathbf q)/(2p)\). Multiplying the ordinary overlap by only
a scalar damping/center phase loses that term and the higher p/d moments.
The shared kernel uses the exact Hermite polynomial instead.

The added pure `GpwOneElectron.Fourier*` tests target closed-form s/s,
odd s/p and d/p, p/p and d/d second/fourth moments, a separated-center p/d
integral, zero-wavevector overlap identities, and conjugation/exchange
identities. They do not solve an SCF or use native snapshots as expected
values. In build13, all 17 `GpwOneElectron` tests (including seven primitive
Fourier tests) and all three `GpwFourierLinks` tests passed. The batched tests
cover analytic image sums for equal/distinct wavevectors, split/batch
identity, mixed-angular translation/conjugation covariance, and invalid-domain
exception propagation. The [actual unit-output extract](../../../tools/cp2k_oracle/inputs/gpw-berry-phase/native-math-validation.txt)
and [metadata/hashes](../../../tools/cp2k_oracle/inputs/gpw-berry-phase/native-math-validation.json)
record all **1771/1771** registered units passing in **46.66 s**. Build13 adds
test registration; native mathematical code is unchanged from the build12
regression measurements above.



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

### Historical external performance record

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
5. Finite spreads; `spin` labels size 16 with values in \(\{0,1\}\);
   both complete sections enforce ≤ 60 s at the registered thread width.

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

### Historical external performance record

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
