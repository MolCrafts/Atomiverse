# gpw/vdw-df

One case; every section below is run by `atv_regressions --case gpw/vdw-df` and keeps its own oracle record.


## Reference — periodic water-dimer GPW PBE + vdW-DF (DRSLL) nonlocal correlation

### Method

**Nonlocal vdW-DF correlation** (Dion–Rydberg–Schröder–Langreth–Lundqvist /
**DRSLL** / libvdwxc **DF1**) as a post-SCF additive on top of Γ-point **GPW
PBE**. Ice-relevant H-bonded water dimer in a periodic 12 Å cell (same geometry
as `gpw-ts-ice` / `gpw-pbe-d3bj-ice` / `gpw-mbd-ice`).

Public API: `cpu::gpw::GpwDft::vdw_df = true` (and `GPWEvaluator::vdw_df`).
Nonlocal energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`.

Formula (Dion et al., *Phys. Rev. Lett.* **92**, 246401 (2004); Román-Pérez &
Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)):

\[
E_c^{\mathrm{nl}}
  = \tfrac12 \iint \rho(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\rho(\mathbf r')
    \,\mathrm d\mathbf r\,\mathrm d\mathbf r'
\]

with \(q_0(\rho,|\nabla\rho|^2)\) (DF1 \(Z_{ab}=-0.8491\), \(q_{\mathrm{cut}}=5\))
and the community RPS kernel tables (libvdwxc 0.5.0 / QE). Evaluation is
**post-SCF** on the final total density (non-self-consistent potential);
spectral \(\lvert\nabla\rho\rvert^2\) from the GPW FFT gradient.

CP2K twin surface: `&VDW_POTENTIAL` / `POTENTIAL_TYPE NON_LOCAL` /
`&NON_LOCAL TYPE DRSLL` (additive on the chosen XC).

### Reference value (hard gate)

**E_disp (PBE density + DRSLL E_c^{nl}, periodic water dimer, 12 Å cell,
CUTOFF 200 Ry, DZVP-MOLOPT-SR-GTH + GTH-PBE) =
+1.5194390480303990×10⁻¹ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry / density | E [Ha] |
| ------ | ------------------ | ------ |
| **libvdwxc 0.5.0** `VDWXC_DF1` (synthetic 16³ Gaussian blob, L=8 Bohr) | fixed ρ,σ | **+1.2367739353586624×10⁻¹** |
| Atomiverse `vdw_df_drsll` (same synthetic) | unit test | matches libvdwxc to **&lt; 1e−6 Ha** |
| Atomiverse GPW PBE + `vdw_df` (this regression) | post-SCF ρ | **+1.5194390480303990×10⁻¹** |

Note: pure \(E_c^{\mathrm{nl}}\) on a molecular density can be positive; the
dispersion *binding* is recovered in interaction differences. Pairwise D3/TS
energies are a different functional form and are **not** comparable in
magnitude.

#### Why not a live CP2K QS wall?

**CP2K** is not available on the pin host for a fair full-QS twin wall this
wave. The kernel is pinned to **libvdwxc** (the same engine CP2K links when
`libvdwxc` is enabled). Twin input notes under
`tools/cp2k_oracle/inputs/gpw-vdw-df-ice/`.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Tolerance

**5 × 10⁻⁵ Ha** on E_disp (regression). Unit tests use **1e−6 Ha** on the
standalone `vdw_df_drsll` kernel vs libvdwxc.

### Perf gate

Kernel RPS is \(O(N_g \log N_g \times n_\alpha)\) with \(n_\alpha=20\); dominated
by the electronic SCF on the 6-atom cell. **perf_exempt = true**: primary
deliverable is E_disp pin parity (same policy as `gpw-ts-ice` / `gpw-mbd-ice`).
No fair CP2K QS wall on the pin host.


## Reference — periodic water-dimer GPW PBE + rVV10 nonlocal correlation

### Method

**Nonlocal rVV10 correlation** (Sabatini–Gorni–de Gironcoli revised VV10) as a
post-SCF additive on top of Γ-point **GPW PBE**. Ice-relevant H-bonded water
dimer in a periodic 12 Å cell (same geometry as `gpw-ts-ice` / `gpw-vdw-df-ice`).

Public API: `cpu::gpw::GpwDft::rvv10 = true` (and `GPWEvaluator::rvv10`).
Nonlocal energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Defaults: b = 6.3, C = 0.0093.

Formula (Sabatini et al., *Phys. Rev. B* **87**, 041108(R) (2013); RPS FFT
of Román-Pérez & Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)):

\[
E_c^{\mathrm{nl}}
  = \beta\int\rho\,\mathrm d\mathbf r
  + \tfrac12\iint \theta(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\theta(\mathbf r')
\]

with \(q_0 = w_0/\kappa\) saturated to the q-mesh, \(\theta_\alpha = c\,\rho^{3/4}
p_\alpha(q_0)\), and the community rVV10 kernel table (CP2K
`rVV10_kernel_table.dat`). Evaluation is **post-SCF** on the final total
density; spectral \(|\nabla\rho|^2\) from the GPW FFT gradient.

CP2K twin surface: `&VDW_POTENTIAL` / `POTENTIAL_TYPE NON_LOCAL` /
`&NON_LOCAL TYPE RVV10` (PARAMETERS 6.3 0.0093).

### Reference value (hard gate)

**E_disp (PBE density + rVV10 E_c^{nl}, periodic water dimer, 12 Å cell,
CUTOFF 200 Ry, DZVP-MOLOPT-SR-GTH + GTH-PBE) =
+6.11534922097146×10⁻² Ha**

| Source | Geometry / density | E [Ha] |
| ------ | ------------------ | ------ |
| **CP2K 2026.1** TYPE RVV10 (SC, OT SCF) | water dimer | **+6.116467960721×10⁻²** |
| Atomiverse GPW PBE + `rvv10` (this regression) | post-SCF ρ | **+6.11534922097146×10⁻²** |

Δ(ATV post-SCF − CP2K SC) ≈ **−1.1×10⁻⁵ Ha** (expected: post-SCF vs
self-consistent density; float32 kernel tables).

#### Offline notes

`tools/cp2k_oracle/inputs/gpw-rvv10-ice/`.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Tolerance

**5 × 10⁻⁵ Ha** on E_disp (regression dual-frozen pin).

### Perf gate

**perf_exempt = true**: primary deliverable is E_disp pin parity (same policy
as `gpw-vdw-df-ice` / `gpw-ts-ice`). CP2K wall ≈ 121 s (OMP=8) for full QS;
ATV SCF+kernel is dominated by electronic SCF.


## Reference — GPW PBE + rVV10 self-consistent nonlocal potential + forces

### Method

**Self-consistent** nonlocal rVV10 correlation on Γ-point **GPW PBE** (periodic
water dimer, same geometry as `gpw-rvv10-ice`). Product path:

1. Each SCF iteration: \(V_{\mathrm{nl}}[\rho]=\delta E_c^{\mathrm{nl}}/\delta\rho\)
   (Soler / CP2K `qs_dispersion_nonloc`) added to KS \(V_{\mathrm{eff}}\).
2. Final \(E_{\mathrm{disp}}=E_c^{\mathrm{nl}}\) on the SC density →
   `GpwResult::e_disp` [Hartree].
3. Analytic forces reuse \(V_{\mathrm{nl}}\) inside `assemble_gpw_forces` veff.

Public API: `cpu::gpw::GpwDft::rvv10 = true` (also `vdw_df` / `vdw_df2` for
DRSLL / LMKLL). Same exclusivity rules as the ice pins.

CP2K twin: `&VDW_POTENTIAL` / `POTENTIAL_TYPE NON_LOCAL` /
`&NON_LOCAL TYPE RVV10` PARAMETERS 6.3 0.0093 (self-consistent by default).

### Reference values (hard gate)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp (gate)** | **CP2K 2026.1** TYPE RVV10 SC | **+6.116467960721×10⁻² Ha** |
| E_disp (ATV SC product) | this regression | +6.11479080989958×10⁻² Ha |
| E_disp (post-SCF ATV, historical) | `gpw-rvv10-ice` | +6.115349220971×10⁻² Ha |
| Force FD | ATV analytic vs central FD \(h=5\times10^{-4}\) Bohr on O0 \(x\) | \|ΔF\| ≈ **1×10⁻⁶ Ha/Bohr** (gate 5×10⁻⁴) |

Regression tolerance on E_disp: **1×10⁻⁴ Ha** vs CP2K SC pin
(Δ(ATV SC − CP2K) ≈ −1.7×10⁻⁵ Ha; single-grid vs CP2K multigrid residual).

#### Offline notes

Input twin: `tools/cp2k_oracle/inputs/gpw-vdw-scf-potential/water_dimer_rvv10_force.inp`
(and `gpw-rvv10-ice` energy twin). Host: Arrhenius `work/cp2k/install` CP2K 2026.1.

```text
Dispersion energy = 0.06116467960721 Ha
Total energy      = -34.27712583714492 Ha
wall ≈ 121 s (OMP=8)
```

### Geometry

Same water dimer as `gpw-rvv10-ice` (12 Å cubic, DZVP-MOLOPT-SR-GTH + GTH-PBE,
CUTOFF 200 Ry, PERIODIC XYZ).

### Perf gate

**perf_exempt = true**: SC RPS costs \(O(N_{\mathrm{SCF}}\times n_\alpha)\) FFTs;
primary deliverable is potential/force correctness (same policy as ice E_disp pins).


## REFERENCE — gpw-vdw-stress

### Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-vdw-stress/water_dimer_vdw_df_stress.inp` |
| Class | Quickstep PBE + `NON_LOCAL TYPE DRSLL` + `STRESS_TENSOR NUMERICAL` |
| System | Water dimer periodic box 12 Å, DZVP-MOLOPT-SR-GTH, CUTOFF 200 Ry |

### Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| \(E_{\mathrm{c}}^{\mathrm{nl}}\) | ~0.15194 Ha (DRSLL class) | \(5\times10^{-4}\) Ha |
| Analytic vs multi-SCF FD max\|Δσ_diag\| | product dual-gate | \(5\times10^{-4}\) Ha/Bohr³ |
| Nonlocal contribution | \(\|\sigma_{\mathrm{vdw}}-\sigma_{\mathrm{bare}}\| > 10^{-6}\) | wiring |

### Fair wall

| Side | Wall |
| --- | --- |
| ATV SCF (OMP=8, first green 2026-08-04) | 10.51 s |
| Analytic stress assembly | ~25 s |
| Multi-SCF FD | ~59 s (gate only; not fair twin) |
| CP2K ENERGY/stress twin | run offline; gate \(r\le 10\) on SCF |


## Reference — Γ collinear-UKS GPW PBE + nonlocal vdW-DF (DRSLL)

### Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Γ collinear UKS**
(LSD). \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\) is a functional
of the **total** density \(\rho=\rho_\alpha+\rho_\beta\) and is
spin-independent, so it is added identically to both spin potentials every
SCF iteration (`build_uks_H`; CP2K `qs_dispersion_nonloc`). The
self-consistent \(E_c^{\mathrm{nl}}\) is folded onto `GpwResult::e_disp`
[Hartree] at every UKS exit (DIIS, OT, fixed-density, budget-exhaust).

Public API: `n_alpha`/`n_beta` + `vdw_df` (also `vdw_df2` / `rvv10`) on
`cpu::gpw::GpwDft` / `GPWEvaluator`. This removes the former
`GpwDft::build` Γ-UKS fail-fast ("E_c^nl would be silently dropped");
the **multi-k** + nonlocal-vdW reject remains (no nonlocal fold on the k
path — `gpw-vdw-kpoints` gate).

Extends `gpw-vdw-scf-potential` (SC V_nl, Γ RKS) × `gpw-uks-pbe`
(O₂ triplet Γ-UKS PBE twin).

### Part A — O₂ triplet LSD PBE + DRSLL vs offline CP2K

| Item | Value |
|------|-------|
| Molecule | O₂, R(O–O) = 1.2075 Å, centered along *z* |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, **PERIODIC XYZ** (RPS kernel needs periodic Poisson) |
| Basis / PP | DZVP-MOLOPT-GTH + GTH-PBE-q6 (12 valence e⁻) |
| XC | PBE + `&NON_LOCAL TYPE DRSLL` (vdW_kernel_table.dat, SCALE 1.0) |
| Cutoff | CP2K: 400 Ry / REL_CUTOFF 60 / NGRIDS 4; Atomiverse: 400 Ry single grid |
| Spin | LSD, MULTIPLICITY 3 → Nα = 7, Nβ = 5 |
| SCF | diagonalization + Broyden (CP2K), EPS_SCF 1e−6; ATV DIIS, tol 1e−7 |

#### Offline CP2K pin

- **Engine:** CP2K **2026.1** (Development Version, source revision
  **85b8a9b49c**, `cp2k.psmp`, `~/work/cp2k/install` on Arrhenius epyc,
  GCC 11.5 / MPICH; libxc + fftw3 flags).
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-uks/o2_uks_pbe_drsll.inp`
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2
  export LD_LIBRARY_PATH=$HOME/work/cp2k/install/lib:$LD_LIBRARY_PATH
  cd /tmp/run && ln -sf $DATA/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 ~/work/cp2k/install/bin/cp2k.psmp -i o2_uks_pbe_drsll.inp \
      -o o2_uks_pbe_drsll.out
  ```
- **Date:** 2026-07-31. SCF converged in 14 steps.
- **Wall (CP2K, 1 rank × 8 OMP):** **159.5 s** (`CP2K` timer).

| Quantity | CP2K pin | Regression tolerance |
|----------|----------|----------------------|
| Total energy [Ha] | **−31.815658522210054** | **1e−3 Ha** (single-grid vs NGRIDS-4 twin; `gpw-uks-pbe` precedent) |
| Dispersion energy \(E_c^{\mathrm{nl}}\) [Ha] | **+0.11311983030413** | **1e−4 Ha** (vdW-family precedent; observed 1e−5 class) |
| ⟨S²⟩ (single det.) | 2.005576 | (not gated here; pinned by `gpw-uks-pbe`) |

Key log lines:

```
 Dispersion energy:                                            0.11311983030413
 Total energy:                                               -31.81565852221005
 Ideal and single determinant S**2 :                    2.000000       2.005576
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.815658522210054
```

### Part B — closed-shell spin-degeneracy identity

Water dimer (same geometry as `gpw-vdw-df-ice` / `gpw-vdw-scf-potential`;
12 Å cubic periodic cell, DZVP-MOLOPT-SR-GTH + GTH-PBE-q1/q6, 200 Ry,
PBE + `vdw_df`, tol 1e−7): 16 valence e⁻; UKS with nα = nβ = 8 must
reproduce RKS with n_occ = 8 (double-occupied).

| Gate | Tolerance |
|------|-----------|
| \(\lvert E_{\mathrm{UKS}}-E_{\mathrm{RKS}}\rvert\) | **3e−4 eV** |
| \(\lvert E_{c,\mathrm{UKS}}^{\mathrm{nl}}-E_{c,\mathrm{RKS}}^{\mathrm{nl}}\rvert\) | **1e−5 Ha** |
| \(E_c^{\mathrm{nl}} \neq 0\) folded on `e_disp` | hard (the old fail-fast's silent-drop scenario) |

#### Atomiverse numbers (verified first green 2026-07-31, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| \|E_tot − CP2K\| (Part A) | **6.8e−5 Ha** | 1e−3 Ha |
| \|E_disp − CP2K\| (Part A) | **1.3e−5 Ha** | 1e−4 Ha |
| \|E_UKS − E_RKS\| (Part B) | **2.6e−8 eV** | 3e−4 eV |
| \|E_disp UKS − RKS\| (Part B) | **1.9e−7 Ha** | 1e−5 Ha |
| SCF | A: 55 iters, B: both converged | |

### Perf gate

CP2K QS wall **159.5 s** (1 × 8 OMP, NGRIDS 4). Atomiverse Part A wall
**~196 s** (OMP=8, single-grid 400 Ry; Part B +21 s). Ratio
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 1.23\) (within_10x).
vdW-family policy (`gpw-vdw-scf-potential` / `gpw-uks-pbe` 177 s twin):
primary deliverable is pin parity; SC RPS costs
\(O(N_{\mathrm{SCF}}\times n_q)\) FFTs every SCF iteration.

### Citations

- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) — vdW-DF (DRSLL).
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) —
  RPS FFT evaluation.
- T. Thonhauser et al., *Phys. Rev. B* **76**, 125112 (2007) — SC V_nl.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.


## Reference — multi-k GPW PBE + vdW-DF (DRSLL) on the closed-shell RKS path

### Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Monkhorst-Pack
k-point** **GPW PBE** (closed-shell RKS) — the combination the multi-k runner
used to reject fail-fast (`"nonlocal vdW (vdw_df / vdw_df2 / rvv10)"` at
`GpwDft::build`). SCF \(V_{\mathrm{nl}} = \delta E_c^{\mathrm{nl}} / \delta\rho\)
is evaluated on the finest-grid k-summed density and added to `veff` every
iteration **before** `last_veff` / multigrid restriction; the post-SCF
\(E_c^{\mathrm{nl}}\) is reported via `e_disp` (never in the SCF convergence
energy). Placement mirrors the Γ runner (`gpw-vdw-scf-potential` /
`gpw-vdw-multigrid`).

Public API: `cpu::gpw::GpwDft` members `vdw_df` (also `vdw_df2` / `rvv10`)
together with `k_mesh` / explicit `k_points` become legal for closed-shell
RKS (single-grid or REL_CUTOFF multigrid). UKS + nonlocal vdW stays
fail-fast (queued `gpw-vdw-uks`).

CP2K twin surface: `&KPOINTS SCHEME MONKHORST-PACK 2 1 1` + `&MGRID NGRIDS 4 /
REL_CUTOFF 40` + `&VDW_POTENTIAL POTENTIAL_TYPE NON_LOCAL / &NON_LOCAL TYPE
DRSLL`. KPOINTS forces diagonalization SCF (no OT) on the CP2K side; ATV uses
the multi-k DIIS path.

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision `85b8a9b49c`),
  Arrhenius local install (same binary as `gpw-vdw-multigrid` /
  `gpw-vdw-scf-potential`).
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-kpoints/water_dimer_vdw_df_kp.inp`
  (water dimer, 12 Å cell, PBE, DZVP-MOLOPT-SR-GTH + GTH-PBE, CUTOFF 200,
  NGRIDS 4 / REL_CUTOFF 40, EPS_SCF 1e-7, EPS_DEFAULT 1e-10, DIAGONALIZATION
  + DIRECT_P_MIXING; DRSLL kernel `vdW_kernel_table.dat`; MP 2×1×1).
- **Run (offline only; never in ctest):**
  ```bash
  export LD_LIBRARY_PATH=$CP2K_INSTALL/lib:$LD_LIBRARY_PATH
  ln -sf $CP2K_DATA/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 cp2k.psmp -i water_dimer_vdw_df_kp.inp -o water_dimer_vdw_df_kp.out
  ```
- **Date:** 2026-07-31. Key lines
  (`grep 'ENERGY| Total\|Dispersion energy'`):

| Quantity | CP2K MP 2×1×1 NGRIDS 4 |
|----------|------------------------|
| E_tot [Ha] | **−34.186333514352548** |
| Dispersion energy [Ha] | **+0.15193559113836** |
| SCF | converged, 15 steps |
| wall (OMP=8) | **18.4 s** |

Context: the Γ multigrid twin (`gpw-vdw-multigrid`) is E_tot
−34.186333524093456 Ha / E_disp +0.15193559412669 Ha — the large 12 Å cell
makes MP 2×1×1 essentially Γ for this dimer (Δ ~1e-11 Ha class on the
oracle).

### Geometry

Same water dimer as `gpw-vdw-df-ice` / `gpw-vdw-multigrid`:

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell **ABC 12.0 12.0 12.0 Å**, PERIODIC XYZ. Basis `DZVP-MOLOPT-SR-GTH`,
PP `GTH-PBE-q1`/`GTH-PBE-q6`. ATV: CUTOFF 200 Ry, DIIS, tol 1e-7 Ha;
G1 single-grid; G2 `ngrids=4`, `rel_cutoff=40`, `k_mesh={2,1,1}`,
`k_time_reversal=false`.

### Gates (measured 2026-07-31, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| G1 \|E_tot(kΓ) − E_tot(Γ)\| | **9.7e-10 Ha** | ≤ 1e-8 Ha |
| G1 \|E_disp(kΓ) − E_disp(Γ)\| | **9.7e-10 Ha** | ≤ 1e-8 Ha |
| G2 \|E_disp(MP mg) − CP2K\| | **4.02e-5 Ha** | ≤ 1e-4 Ha |
| G2 \|E_tot(MP mg) − CP2K\| | **7.93e-3 Ha** | ≤ 1e-2 Ha |

ATV G2 values: E_tot = −34.178407374593 Ha, E_disp = +0.151895422335 Ha.
The absolute E_tot offset is the **pre-existing Γ-dimer twin-family systematic**
(identical to `gpw-vdw-multigrid` Δ ≈ 7.93e-3 Ha); the capability statements
are the Γ-limit identity and the E_disp pin.

### Out of scope here

- UKS nonlocal vdW (`gpw-vdw-uks`).
- Analytic k-point vdW force/stress gates (V_nl already rides the k force
  assembly's grid-term `veff`; explicit gates are a follow-up).
- CUDA.

### Perf

| Run | Wall (OMP=8) |
|-----|--------------|
| CP2K MP 2×1×1 NGRIDS 4 | **18.4 s** |
| ATV G2 only (MP multigrid) | **13.7 s** |
| ATV full regression (G1×2 + G2) | **48.0 s** (≤ 60 s budget) |

\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.74\) on the G2 parity twin (≤ 10×;
no exemption needed).


## Reference — multi-k GPW PBE + vdW-DF (DRSLL) analytic ionic forces

### Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Monkhorst–Pack
k-point** **GPW PBE** (closed-shell RKS) with **analytic ionic forces**.
SCF \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\) is evaluated on the
finest-grid k-summed density and added to `veff` every iteration; the same
\(V_{\mathrm{nl}}\) is folded into the multi-k force assembly
(`assemble_gpw_forces_kpoints` grid-term `veff` in `gpw_forces.cpp`).

Public API: `cpu::gpw::GpwDft` with `vdw_df=true` + `k_mesh` (or explicit
`k_points`) + `GpwDft::forces(frame)` (eV/Å on the Frame). No public
`*Config`, no factories. Energy product already closed by
`reg_gpw-vdw-kpoints`; Γ SC potential + FD by `reg_gpw-vdw-scf-potential`.

### System

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ=8`) |
| XC | PBE + NON_LOCAL **DRSLL** (kernel `vdW_kernel_table.dat`) |
| Grid | CUTOFF **200** Ry, NGRIDS **4**, REL_CUTOFF **40** |
| k mesh | MONKHORST-PACK **2×1×1** (ATV `k_time_reversal=false` for 1:1 twin) |
| SCF | EPS_SCF / tol **1e-7** Ha; DIAGONALIZATION + DIRECT_P_MIXING |

### Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, source build `…/cp2k/install/env.sh`, git:**5e54ba2**) |
| Host | Arrhenius login, offline only |
| Parallelism | **1 MPI × OMP 8** (`mpirun -np 1`, `OMP_NUM_THREADS=8`) |
| Input | `tools/cp2k_oracle/inputs/gpw-vdw-kpoint-forces/water_dimer_vdw_df_kp_forces.inp` |
| Wall (ENERGY_FORCE twin) | **13.216 s** (CP2K internal `CP2K` timer TOTAL TIME maximum) |
| Date | 2026-08-01 |

#### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| **Dispersion energy** | **0.15193559149613** |
| **Total** | **−34.186333514352626** |

(Matches the energy-only multi-k twin of `gpw-vdw-kpoints` to print residual.)

#### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` table)

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.35844769E−02 | 1.49028331E−09 | 2.51422839E−10 |
| H0 | 4.16583157E−02 | −5.35060271E−09 | 1.31835454E−01 |
| H1 | 4.16583337E−02 | 4.10205702E−09 | −1.31835315E−01 |
| O1 | −7.89346446E−02 | 2.67651054E−11 | 6.49135595E−09 |
| H2 | 3.76866359E−02 | −4.43451105E−09 | 1.35635778E−01 |
| H3 | 3.76865671E−02 | 4.54720810E−09 | −1.35635393E−01 |

CP2K `FORCES| Sum` residual ≈ 3.8×10⁻³ Ha/Bohr (finite basis / grid).

#### Re-run (offline only — never ctest)

```bash
source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
cd /tmp && mkdir -p atv-pin && cd atv-pin
ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
export OMP_NUM_THREADS=8
mpirun -np 1 cp2k.psmp \
  -i …/tools/cp2k_oracle/inputs/gpw-vdw-kpoint-forces/water_dimer_vdw_df_kp_forces.inp \
  -o water_dimer_vdw_df_kp_forces.out
## Record: ENERGY| Total; Dispersion energy; FORCES|; CP2K timer TOTAL TIME
```

### Offline FD identity (decisive correctness)

On the **single-grid** multi-k product path (`ngrids=1`, same mesh / DRSLL /
CUTOFF 200, Release, OMP=8, 2026-08-01), central FD of \(E_{\mathrm{total}}\)
w.r.t. O0 \(x\) (\(h=5\times10^{-4}\) Bohr):

| Quantity | Value |
| -------- | ----- |
| \(F_{\mathrm{ana},O0,x}\) | −8.976047×10⁻² Ha/Bohr |
| \(F_{\mathrm{FD},O0,x}\) | −8.975976×10⁻² Ha/Bohr |
| \(\|F_{\mathrm{ana}}-F_{\mathrm{FD}}\|\) | **7.07×10⁻⁷ Ha/Bohr** |

Proves multi-k analytic forces with SC \(V_{\mathrm{nl}}\) are \(-\nabla E\) of
the multi-k energy. Not re-run in ctest: three multi-k SCFs exceed the ≤ 60 s
regression budget (a first-green band frozen after offline FD).

### Atomiverse twin (first green, 2026-08-01, OMP=8)

| Quantity | Value |
| -------- | ----- |
| E_tot | −34.17840737459292 Ha (Δ vs CP2K **7.926e−3** — pre-existing Γ-dimer systematic) |
| E_disp | +0.15189542233538 Ha (Δ **4.02e−5**) |
| n_kpts | 2 |
| max\|F_atv − F_cp2k\| | **7.25e−3** Ha/Bohr (signal ~0.136) |
| F_O0_x (ATV pin) | **−8.976000e−02** Ha/Bohr |
| \|∑F\| | ~1.5e−2 Ha/Bohr |
| Twin wall | **20.8–44 s** (SCF + force; host load) |
| Fair ratio | **1.6–3.3** (≤ 10) |

### Gates / tolerances

| Gate | Tol | Why |
| ---- | --- | --- |
| \|E_disp − CP2K\| | **1×10⁻⁴ Ha** | Same class as `gpw-vdw-kpoints` E_disp pin |
| \|E_tot − CP2K\| | **1×10⁻² Ha** | Twin-family systematic ~8e−3 Ha (not a force defect) |
| max\|F − CP2K\| | **1×10⁻² Ha/Bohr** | Same residual class on absolute forces; decisive correctness is offline FD |
| \|F_O0_x − ATV pin\| | **5×10⁻⁵ Ha/Bohr** | Frozen product reproducibility |
| fair wall ratio | **≤ 10** | vs real CP2K ENERGY_FORCE twin only |
| product wall | **≤ 60 s** | e2e tier budget |

### Perf

| Run | Wall (OMP=8) |
|-----|--------------|
| CP2K ENERGY_FORCE twin | **13.216 s** |
| ATV twin SCF + forces | **~21–44 s** |
| \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **~1.6–3.3** (≤ 10; no exemption) |

### Out of scope

- UKS multi-k vdW forces
- Analytic stress with nonlocal vdW
- CUDA / cross-representation (PW NC-UPF) twins

### Provenance

- Spec: `.claude/specs/gpw-vdw-kpoint-forces/`
- Deps: `gpw-vdw-kpoints` ✓, `gpw-vdw-scf-potential` ✓
- Oracle family: **cp2k** (matched GTH-GPW + NON_LOCAL + KPOINTS ENERGY_FORCE)


## Reference — Γ GPW PBE + vdW-DF (DRSLL) on the REL_CUTOFF multigrid (NGRIDS 4)

### Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on Γ-point **GPW PBE**
with **CP2K-style CUTOFF/REL_CUTOFF multigrid product collocation**
(`ngrids > 1 && rel_cutoff > 0`) — the combination the Γ runner used to
reject (`"GPW vdw_df: multigrid not supported (single-grid only)"`).

Why the gate simply opens: nonlocal vdW is a functional of the **finest-grid**
total density. The product path already prolongs every level's collocated
density to the finest grid before Poisson/XC and restricts the finest `veff`
back per level (exact adjoint). \(V_{\mathrm{nl}}[\rho]\) is evaluated on that
fine ρ and added to `veff` at the same point as Hartree + XC, **before** the
restriction — so the integrated KS matrix stays the exact gradient of the
collocated \(E_H + E_{xc} + E_c^{\mathrm{nl}}\) and the SCF is variational on
both paths. The final \(E_c^{\mathrm{nl}}\) re-collocates the SC density
through the same products + spectral prolongation.

Public API: `cpu::gpw::GpwDft` members `vdw_df` (also `vdw_df2` / `rvv10`),
`ngrids`, `rel_cutoff` — they merely become legal together. k-point vdW stays
fail-fast rejected (queued `gpw-vdw-kpoints`). **Found rot fixed in the same
cut:** Γ UKS + nonlocal vdW silently dropped \(E_c^{\mathrm{nl}}\)
(`e_disp` stayed 0 while the run "succeeded") — now rejected fail-fast at
`GpwDft::build`.

CP2K twin surface: `&MGRID NGRIDS 4 / REL_CUTOFF 40` (NGRIDS 4 is CP2K's
default) + `&VDW_POTENTIAL POTENTIAL_TYPE NON_LOCAL / &NON_LOCAL TYPE DRSLL`.

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision `85b8a9b49c`),
  Arrhenius local install `~/work/cp2k/install` (same binary as the
  `gpw-vdw-scf-potential` pin).
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-multigrid/water_dimer_vdw_df_mg.inp`
  (water dimer, 12 Å cell, PBE, DZVP-MOLOPT-SR-GTH + GTH-PBE, CUTOFF 200,
  EPS_SCF 1e-7, EPS_DEFAULT 1e-10, OT/DIIS; DRSLL kernel
  `vdW_kernel_table.dat`). NGRIDS 1 context = same input with `NGRIDS 1`.
- **Run (offline only; never in ctest):**
  ```bash
  ln -sf $CP2K_DATA/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 cp2k.psmp -i water_dimer_vdw_df_mg.inp -o water_dimer_vdw_df_mg.out
  ```
- **Date:** 2026-07-31. Key lines (`grep 'ENERGY| Total\|Dispersion energy'`):

| Quantity | CP2K NGRIDS 4 (gate) | CP2K NGRIDS 1 (context) |
|----------|----------------------|-------------------------|
| E_tot [Ha] | **−34.186333524093456** | −34.186334128002194 |
| Dispersion energy [Ha] | **+0.15193559412669** | +0.15193559998706 |
| SCF | converged, 30 steps | converged, 30 steps |
| wall (OMP=8) | 24.7 s | 24.2 s |

CP2K's own NGRIDS 4 ↔ 1 gap: 6.0e-7 Ha (E_tot), 5.9e-9 Ha (E_disp) — the
multigrid is a cost optimization, not a physics change, which is what the
internal-consistency gate pins on the ATV side.

### Geometry

Same water dimer as `gpw-vdw-df-ice` / `gpw-vdw-scf-potential`:

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell **ABC 12.0 12.0 12.0 Å**, PERIODIC XYZ. Basis `DZVP-MOLOPT-SR-GTH`,
PP `GTH-PBE-q1`/`GTH-PBE-q6`. ATV: CUTOFF 200 Ry, `ngrids=4`,
`rel_cutoff=40`, DIIS, tol 1e-7 Ha.

### Gates (measured 2026-07-31, OMP=8, eps_default 1e-8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| \|E_tot(mg) − E_tot(sg)\| (ATV internal) | **4.28e-6 Ha** | 2e-5 Ha |
| \|E_disp(mg) − CP2K NGRIDS 4\| | **4.02e-5 Ha** | 1e-4 Ha |
| \|E_tot(mg) − CP2K NGRIDS 4\| | **7.93e-3 Ha** | 1e-2 Ha |

ATV values: E_tot sg = −34.178411658610, mg = −34.178407380745 Ha;
E_disp sg = +0.151895580634, mg = +0.151895426360 Ha.

The absolute ATV ↔ CP2K E_tot offset is the **pre-existing systematic of this
Γ dimer twin family** (never gated by the shipped single-grid vdW siblings):
it is identical on both grid paths (Δ_sg = 7.922e-3 vs Δ_mg = 7.926e-3 Ha —
they agree to the ATV internal-consistency level), and `eps_default = 1e-10`
moves E_tot by < 1e-8 Ha (measured), so it is not tail screening. The gate
pins that the multigrid total stays inside the single-grid envelope; the
capability statements are rows 1–2.

### Out of scope here

- k-point vdW (`gpw-vdw-kpoints`, queued) and UKS vdW (fail-fast reject).
- vdW forces/stress × multigrid: the analytic force assembly keeps its
  single-grid fine-cutoff collocation for **all** multigrid runs (pre-existing,
  functional-independent) — unchanged by this slug.

### Perf

CP2K NGRIDS 4 wall 24.7 s (OMP=8, same host). ATV regression wall **20.6 s
for BOTH solves** (sg + mg, OMP=8) →
\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.83\) even charging both ATV
solves against CP2K's single multigrid run (≤ 10×; no exemption needed).
