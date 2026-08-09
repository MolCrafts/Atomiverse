# Reference — Multi-k GPW RKS with REL_CUTOFF multigrid (diamond Si, NGRIDS 4)

## Method

Closed-shell GPW Kohn–Sham DFT (PBE) with **Monkhorst–Pack k-point sampling
and CP2K-style CUTOFF/REL_CUTOFF multigrid product collocation** — the
capability the k-point branch used to reject (`GpwDft::build`:
`"multigrid (ngrids > 1)"`). The geometry-fixed primitive-pair product list is
the same one the Γ multigrid uses; under Bloch sampling each product's ket
image shift t becomes a phase index:

- density: ρ(g) weighted per product by the real-space stack
  **D(t) = Σ_k w_k Re[P^k e^{−2πi κ·t}]** (Γ limit: D(t) = P);
- per-level densities prolong spectrally to the finest grid for Poisson + XC;
- the fine potential restricts back per level (exact adjoint) and integrates
  into the real per-cell stack **V_t**, phase-summed per k into the Hermitian
  grid KS matrix **V^k = Σ_t e^{+2πi κ·t} V_t**.

Public API: concrete `cpu::gpw::GpwDft` with members
(`k_mesh`, `ngrids`, `rel_cutoff`, `cutoff_ry`, …); lookup constructors
`GaussianBasis(Element::Si, "DZVP-MOLOPT-GTH")` /
`GthPotential(Element::Si, "GTH-PBE-q4")`. No public `*Config`, no factories.

## System

Silicon diamond in the body-centred-tetragonal 4-atom cell (the
orthorhombic-compatible setting of the fcc diamond lattice) — identical to
`regressions/gpw/gpw-kpoints-si`:

| | |
| --- | --- |
| Lattice | a = 5.43 Å (10.26120 Bohr); cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si), in-tree table |
| Pseudopotential | GTH-PBE-q4 (16 valence e⁻ → n_occ = 8) |
| XC | PBE |
| Density cutoff | 150 Ry (main); 40 Ry / REL 20 (M2/M3 cheap-ladder twin) |
| Multigrid | **NGRIDS 4, REL_CUTOFF 60 Ry** (ATV + CP2K twin) |
| k mesh | MONKHORST-PACK 2×2×2, time-reversal folded → 4 k, w = ¼ |
| SCF | DIIS depth 6, tol 1e-8 Ha, EPS_DEFAULT 1e-7 |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), module `CP2K/2026.1` on Arrhenius
  (visible after `module load buildtool-easybuild/5.2.1-hpca3ef7d197
  GCC/14.3.0 MPICH/4.3.2`).
- **Input:** `tools/cp2k_oracle/inputs/gpw-multigrid-kpoints/si_kpoints_mg.inp`
  (identical to the `gpw-kpoints-si` twin except `&MGRID`: CUTOFF 150,
  **NGRIDS 4, REL_CUTOFF 60**).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  OMP_NUM_THREADS=4 mpirun -np 1 cp2k.psmp -i si_kpoints_mg.inp -o si_kpoints_mg.out
  ```
- **Date:** 2026-07-31.

| Quantity | CP2K pin | Tolerance (M5) |
|----------|----------|----------------|
| E_tot [Ha], NGRIDS 4 | **−15.802416412503845** | **1e-3 Ha** |
| E_tot [Ha], NGRIDS 1 (context) | −15.802416412655166 | — |

Key log line: `ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]
-15.802416412503845`; SCF converged in 13 steps. CP2K's own NGRIDS 4 ↔ 1 gap
is 1.5e-10 Ha — the multigrid is a cost optimization, not a physics change,
which is what gate M3 pins on the ATV side.

## Found & fixed while opening the gate

The shipped **periodic** product path (Γ multigrid) screened whole products
and window tails at `eps_default` as a *value* threshold; a dropped diffuse
product discards **integrated mass** up to (π/p)^{3/2} ≈ 70× that value, and
a periodic cell holds thousands of image pairs at threshold. On this system
the collocated density integrated to **15.910 of 16 e⁻** and the SCF sat
4.1e-2 Ha off the single grid *even with every level at the fine cutoff*
(the shipped multigrid regressions are MT-isolated, where image pairs are
absent, so this never fired). Fixed by the derived screen margin
`kProductMassMargin = 1e-4` in `internal/multigrid.cpp` (Γ and k paths
alike); after the fix the eps ladder converges:

| eps_default | E(mg) − E(sg) [Ha] |
|---|---|
| 1e-7 (this regression) | +3.5e-5 |
| 1e-9 | +2.6e-6 vs sg at 1e-9 |

Also found on the same gate: nonlocal vdW (`vdw_df`/`vdw_df2`/`rvv10`) was
silently dropped on the whole k path (semilocal XC ran, E_c^nl never added) —
now rejected fail-fast at `build()`.

## Measured (2026-07-31, this tree)

Standing regression gates:

| Gate | Value | Tolerance |
| --- | --- | --- |
| M3 mg vs sg (40 Ry / REL 20) | 1.13e-5 Ha | 1e-3 |
| M5 \|E − CP2K\| (150 Ry / REL 60, MP 2×2×2 TR) | **4.73e-7 Ha** | 1e-3 |

E(MP 2×2×2, 150 Ry, NGRIDS 4) = −15.8024168854 Ha (7 DIIS iters).

Bring-up exactness identities (full-SCF twins measured once here, then kept as
unit tests — each multigrid SCF costs ~15–39 s and the ≤ 60 s budget goes to
the pin; mechanisms unit-pinned in `tests/cpu/gpw/test_cpu_gpw_multigrid.cpp`
`GpwMultigridCells.*`):

| Identity | Measured |
| --- | --- |
| Γ-limit: explicit k = (0,0,0) mg k-path vs Γ mg product path (95 Ry) | 2.0e-14 Ha |
| Time reversal: full 8k vs TR-folded 4k (95 Ry mg) | 2.8e-14 Ha |
| Time reversal (60 Ry mg) / (40 Ry mg) | 2.0e-14 / 2.3e-14 Ha |
| mg vs sg at 150 Ry / REL 60 | 2.28e-5 Ha |

## Performance

This file carries no wall numbers. The single binding fair-wall source for
this twin — host, CP2K binary provenance, PRINT_LEVEL/timer convention,
reproduction recipe, and the current \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) —
is [`.claude/notes/gpw-perf-baseline.md`](../../../.claude/notes/gpw-perf-baseline.md).
Case-local walls were last measured here 2026-07-31 and are superseded by
that note.

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- CP2K Quickstep `&MGRID` CUTOFF / REL_CUTOFF / NGRIDS multigrid collocation;
  `&KPOINTS SCHEME MONKHORST-PACK` (Monkhorst & Pack, PRB **13**, 5188 (1976)).
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.
