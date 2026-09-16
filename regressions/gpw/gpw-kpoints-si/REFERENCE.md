# Reference — Monkhorst-Pack k-point GPW RKS on diamond Si (bct 4-atom cell)

## Method

Closed-shell GPW Kohn–Sham DFT (PBE) with **Monkhorst–Pack k-point
sampling** — the CP2K Quickstep `&KPOINTS SCHEME MONKHORST-PACK` loop shape:
complex Bloch-phased analytic matrices per k
(S_k / T_k lattice sums, phased Kleinman–Bylander GTH nonlocal), per-k
pre-summed complex basis-on-grid values for the k-summed density
ρ(r) = Σ_k w_k Σ_μν P^k_μν χ^k_μ conj(χ^k_ν), per-k complex Hermitian
canonical-orthogonalization + zheevd diagonalization, and k-summed Pulay
DIIS. CP2K MP convention κ_d = (2r − N − 1)/(2N) (N = 2 → ±1/4, Γ-avoiding);
time-reversal folding of ±k (exact for RKS).

Public API: concrete `cpu::gpw::GpwDft` with members
(`k_mesh`, `k_time_reversal`, explicit `k_points`/`k_point_weights`,
`cutoff_ry`, …); lookup constructors
`GaussianBasis(Element::Si, "DZVP-MOLOPT-GTH")` /
`GthPotential(Element::Si, "GTH-PBE-q4")`. No public `*Config`, no factories.
Result payload: `GpwResult::{k_frac, k_weights, eps_k}`.

## System

Silicon diamond in the **body-centred-tetragonal 4-atom cell**
(the orthorhombic-compatible setting of the fcc diamond lattice):

| | |
| --- | --- |
| Lattice | a = 5.43 Å (10.26120 Bohr); cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) — in-tree table (`gpw_tables.h`) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → n_occ = 8 |
| XC | PBE |
| Density cutoff | 150 Ry (main run); 95 Ry (identity gates — commensurate supercell grids, unit-test-pinned) |
| k mesh | MONKHORST-PACK 2×2×2, time-reversal folded → 4 k, w = ¼ |
| SCF | DIIS depth 6, tol 1e-8 Ha, EPS_DEFAULT 1e-7 |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), module `CP2K/2026.1` on
  Arrhenius (visible after `module load buildtool-easybuild/5.2.1-hpca3ef7d197
  GCC/14.3.0 MPICH/4.3.2`).
- **Input:** the exact twin `tools/cp2k_oracle/inputs/gpw-kpoints-si.inp`
  (CUTOFF 150, NGRIDS 1, MONKHORST-PACK 2 2 2, PBE, EPS_SCF 1e-8).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  OMP_NUM_THREADS=4 mpirun -np 1 cp2k.psmp -i gpw-kpoints-si.inp -o si_kpoints_sg.out
  ```
- **Date:** 2026-07-31 (replaces the 2026-07-25 ATV-frozen first-green band —
  no CP2K module was visible then without the easybuild toolchain loaded).

| Quantity | CP2K pin | Tolerance (A7) |
|----------|----------|----------------|
| E_tot [Ha] | **−15.802416412655166** | **1e-3 Ha** |

Key log line (CP2K 2026.1): `ENERGY| Total FORCE_EVAL ( QS ) energy
[hartree]            -15.802416412655166`; SCF converged in 13 steps; wall
≈ 3.3 s (1 rank × OMP 4). ATV first green −15.8023941333 Ha sits 2.2e-5 Ha
from the CP2K number.

Supporting internal identities (unchanged):

1. **Internal exactness identities** — the decisive k-point correctness
   checks; each holds to SCF tolerance *independent of any oracle*:
   - **Supercell folding** (gate A2): E(bct, mesh 1×1×2 ⇒ κ_z = ±¼,
     genuinely complex phases {±1, ±i}) ≡ E(z-doubled 8-atom cell at the
     folded zone-boundary point κ = (0,0,½))/2. Measured **|Δ| = 6.9e-14 Ha**
     (tol 5e-5).
   - **Time reversal** (A3): full 8-k mesh vs TR-folded 4-k mesh —
     **|Δ| = 0.0** (tol 5e-6).
   - **Γ limit** (A1): explicit k = (0,0,0) through the complex machinery vs
     the pre-existing real Γ path (compact-basis periodic H₂ toy,
     DZV-GTH-PADE) — **|Δ| = 1.4e-11 Ha** (tol 1e-6).
   The A2 identity found a real bug at first run: the phased KB nonlocal
   assembly conjugated the wrong side (V_nl evaluated at −k; Hermitian,
   Γ-exact, TR-invisible — but 2.09 Ha off at κ = ¼). Fixed in
   `internal/kpoints.cpp`; the identity is now machine-exact.
2. **Absolute CP2K pin** (A7): the offline CP2K 2026.1 number above.

### Atomiverse numbers (first green, 2026-07-25)

| Quantity | Value |
| --- | --- |
| **E(MP 2×2×2 TR, 150 Ry)** | **−15.8023941333 Ha** (SCF 6 iters) |
| E(MP 2×2×2 TR, 95 Ry) | −15.8024022460 Ha |
| E(explicit Γ, 150 Ry) | −15.2031525100 Ha |
| k-sampling shift \|E_MP − E_Γ\| | 5.99e-01 Ha (gate floor 1e-3) |
| E(mesh 1×1×2, 95 Ry) | −15.3483059090 Ha |
| E(8-atom supercell, κ_z = ½, 95 Ry)/2 | −15.3483059090 Ha |
| Sampled-k gap (VBM 0.137858 Ha, CBM 0.283137 Ha) | 3.953 eV (floor 0.1 eV) |

## Tolerances

- A1 ≤ 1e-6 Ha; A2 ≤ 5e-5 Ha; A3 ≤ 5e-6 Ha (identities).
- A4 ≥ 1e-3 Ha activation floor; A5 gap ≥ 0.1 eV.
- A7 absolute: 1e-3 Ha vs the offline CP2K 2026.1 pin.

## What is checked

1. Γ-limit, supercell-folding, and time-reversal identities (A1–A3).
2. k-sampling activation and insulating gap + k payload shapes (A4–A5).
3. SCF convergence, finiteness (A6); absolute CP2K pin (A7).
4. No external process spawn; ≤ 60 s.

## Performance

| Item | Value |
| ---- | ----- |
| T_cp2k (twin input, 1 rank × OMP 4, 2026-07-31) | ≈ 3.3 s |
| T_atv (main MP 2×2×2 run alone, OMP=8) | ~4 s |
| r = T_atv/T_cp2k | ≈ 1.2 (≤ 10 ✓) |
| `within_10x` | true |
| T_atv (whole regression: 7 SCF solves, OMP=8, Release) | ~17 s |

## Provenance

- Spec: `.claude/specs/gpw-kpoints-si/` (deleted on completion; git history)
- MP mesh: Monkhorst & Pack, PRB **13**, 5188 (1976); CP2K KPOINTS convention
- GTH: Goedecker–Teter–Hutter, PRB **54**, 1703 (1996); Krack, TCA **114**,
  145 (2005); MOLOPT: VandeVondele & Hutter, JCP **127**, 114105 (2007)
- Twin input: `tools/cp2k_oracle/inputs/gpw-kpoints-si.inp`
