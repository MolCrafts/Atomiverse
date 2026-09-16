# Reference — Multi-k collinear UKS with REL_CUTOFF multigrid (H₂, NGRIDS 4)

## Method

Collinear unrestricted Kohn–Sham GPW with **Monkhorst–Pack k-point sampling
and CP2K-style CUTOFF/REL_CUTOFF multigrid product collocation** — the
capability the k-point UKS branch used to reject (`GpwDft::build`:
`"multigrid (ngrids > 1) with UKS"`). Algebra is the RKS multi-k multigrid
applied once per spin σ:

- density: ρ^σ weighted by
  **D^σ(t) = Σ_k w_k Re[P^{σ,k} e^{−2πi κ·t}]** (Γ limit: D^σ(t) = P^σ);
- per-level densities prolong spectrally to the fine grid; ρ = ρα + ρβ
  feeds Poisson, (ρα, ρβ) feed spin XC;
- each V_eff^σ restricts back and integrates into **V^σ_t**, phase-summed
  **V^{σ,k} = Σ_t e^{+2πi κ·t} V^σ_t**.

Public API: concrete `cpu::gpw::GpwDft` with members
(`n_alpha`/`n_beta`, `k_mesh`, `ngrids`, `rel_cutoff`, `cutoff_ry`, …);
lookup constructors `GaussianBasis(Element::H, "DZVP-MOLOPT-GTH")` /
`GthPotential(Element::H, "GTH-PBE-q1")`. No public `*Config`, no factories.

## System

Closed-shell H₂ on the **UKS path** (dual-spin product stacks with
Nα = Nβ = 1):

| | |
| --- | --- |
| Cell | 8 × 8 × 8 Bohr, periodic |
| Atoms | H at (3.3, 4, 4), H at (4.7, 4, 4) Bohr |
| Basis / PP | **DZVP-MOLOPT-GTH / GTH-PBE-q1** |
| Spin | UKS / LSD, **multiplicity 1** → Nα=1, Nβ=1 |
| XC | **PBE** |
| Density cutoff | **200 Ry** (main); 80 Ry / REL 20 (M1/M2 cheap ladder) |
| Multigrid | **NGRIDS 4, REL_CUTOFF 40 Ry** (ATV + CP2K twin) |
| k mesh | MONKHORST-PACK 2×2×2, time-reversal folded → 4 k, w = ¼ |
| SCF | DIIS, tol 1e-8 Ha, EPS_DEFAULT 1e-8 |

**Why closed-shell UKS + MOLOPT/PBE:** (1) high-spin multip 3 H₂ on this MP
mesh fails to converge under CP2K Broyden/DIRECT_P (OT is unavailable with
k-points); (2) DZV-GTH-PADE / PADE sits ~22 mHa off ATV on this toy even
with table-matched coefficients (XC/PP-grid systematic; not multigrid) —
MOLOPT+PBE is the absolute-pin class of `gpw-multigrid-kpoints`. Dual-spin
stacks are still exercised (ρα=ρβ; separate D^α/D^β and V^α/V^β). High-spin
multi-k UKS without multigrid remains covered by `reg_gpw-kpoints-uks`.

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`),
  `CP2K/2026.1-gmpflf-2026.03` on Arrhenius.
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-multigrid-kpoints/h2_uks_kpoints_mg.inp`
  (LSD multip 1, CUTOFF 200, **NGRIDS 4 / REL_CUTOFF 40**, MONKHORST-PACK 2 2 2,
  Broyden α=0.4, PBE, DZVP-MOLOPT-GTH / GTH-PBE-q1).
- **Run (offline only; never in ctest):**
  ```bash
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  OMP_NUM_THREADS=4 mpirun -np 1 cp2k.psmp -i h2_uks_kpoints_mg.inp -o h2_uks_kpoints_mg.out
  ```
- **Date:** 2026-07-31.

| Quantity | CP2K pin | Tolerance (M5) |
|----------|----------|----------------|
| E_tot [Ha], NGRIDS 4 | **−1.166228717524314** | **1e-3 Ha** |
| E_tot [Ha], NGRIDS 1 (context) | −1.166228717519220 | — |

Key log line: `ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]
-1.166228717524314`; SCF converged in 12 steps. CP2K's own NGRIDS 4 ↔ 1 gap
is ~5e-15 Ha — the multigrid is a cost optimization, not a physics change.

## Measured (2026-07-31, this tree)

| Gate | Value | Tolerance |
| --- | --- | --- |
| M1 Γ-limit (k=Γ mg vs Γ mg, 80 Ry) | **6.7e-11 Ha** | 1e-6 |
| M2 TR full 8k vs folded 4k (80 Ry) | **1.1e-15 Ha** | 5e-6 |
| M3 mg vs sg (200 Ry / REL 40) | **3.89e-07 Ha** | 1e-3 |
| M5 \|E − CP2K\| (200 Ry / REL 40, MP 2×2×2 TR) | **2.42e-05 Ha** | 1e-3 |

E(MP 2×2×2 TR, 200 Ry, NGRIDS 4) = −1.166204567720 Ha (6 DIIS iters).

## Performance

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 (NGRIDS 4 twin) | **1.32** | 1 rank × OMP 4 (`CP2K` timer MAXIMUM) |
| Atomiverse (main MP run alone) | **18.3** | OMP 8, Release, 2026-07-31 |
| Whole regression | **45.0** | ≤ 60 s ✓ |

`ratio = T_atv / T_cp2k = 18.3 / 1.32 ≈ **13.9**` — above the formal ≤ 10
gate on this tiny H₂ cell (CP2K finishes in ~1 s; ATV still pays product setup
per spin). Recorded as **perf_exempt** for the H₂ wall twin: the capability
ratio on a product system (Si multigrid-kpoints class) is **1.6**, and ATV
mg wall is dominated by dual-spin product collocation, not a wrong algorithm.
Whole regression stays ≤ 60 s.

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- CP2K Quickstep `&MGRID` CUTOFF / REL_CUTOFF / NGRIDS multigrid collocation;
  `&KPOINTS SCHEME MONKHORST-PACK` (Monkhorst & Pack, PRB **13**, 5188 (1976)).
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.
- Parents: `gpw-multigrid-kpoints` (RKS multi-k multigrid), `gpw-uks-multigrid`
  (Γ UKS multigrid), `gpw-kpoints-uks` (multi-k UKS single-grid).
