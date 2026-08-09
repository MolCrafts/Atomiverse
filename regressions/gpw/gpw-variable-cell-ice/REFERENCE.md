# Reference — GPW variable-cell relax vs CP2K CELL_OPT (ice-like water chain)

## Method

Productizes GPW strain-FD stress (`gpw-stress-water`) into **variable-cell
relaxation**: `cpu::minimizer::Minimize` + `AnisotropicCellRelax`
(per-axis mask `relax_x` only; `kStress` is the extensive Voigt virial
intensive `kStress` [eV/Å³]) + `StressConverged` criterion, with
`cpu::gpw::GPWEvaluator` (`compute_stress = true`,
`stress_axes = {x only}`) bound at the `force` phase. `GPWEvaluator::launch`
re-targets the solver cell to the frame box each call (the old
"GPW does not support a variable cell" throw is gone); `GpwDft::run` rebuilds
grids / lattice images / S/T/Vpp per call, so the relax loop rescales
`frame.box` freely at **fixed fractional coordinates**.

CP2K twin problem: `RUN_TYPE CELL_OPT`, `TYPE DIRECT_CELL_OPT`,
`KEEP_ANGLES`, **`CONSTRAINT YZ`** (only the chain axis relaxes) with **all
atoms fixed** (`FIXED_ATOMS 1..3`) — verified: CP2K keeps the scaled
(fractional) coordinates while optimizing the cell, i.e. exactly the
functional E(a) the ATV loop minimizes.

## System (honest minimal "ice")

Periodic proton-ordered **water chain** — 1 H₂O per cell, H-bonded to its own
x-images (the ice Ih c-axis strand motif) — `PERIODIC XYZ`, b = c = 8 Å
vacuum, chain axis `a` variable. Loop precedent for the ice row:
`gpw-pbe-d3bj-ice` pinned a periodic H-bonded pair. The full ice Ih
(24-atom orthorhombic ice XI) cell exceeds the 60 s regression budget by
orders of magnitude at any product cutoff — follow-on after analytic stress
+ periodic-collocation perf work.

Fractional coordinates (fixed; CP2K `SCALED T`; monomer = gas-phase water
geometry at a = 3.0 Å):

| Atom | f_x | f_y | f_z |
| ---- | --- | --- | --- |
| O  | 0.000000 | 0 | 0.000000 |
| H1 | 0.319067 | 0 | 0.000000 |
| H2 | −0.079900 | 0 | 0.115830 |

### Basis / XC pivot (measured; binding for this twin)

The queue row's DZVP-MOLOPT-GTH is unusable for a *periodic* ≤ 60 s
regression today: ATV's periodic MOLOPT image-sum collocation costs
**> 10 min per relax launch** (3 SCFs) on this chain — the same wall
`gpw-pbesol-si` documented when it retreated to nonperiodic MT. With the
compact CP2K standard GTH basis the identical launch costs **1.9 s**
(≈ 300×). The variable-cell machinery is basis-agnostic; the twin therefore
uses:

| Knob | Value |
| ---- | ----- |
| XC | **PADE LDA** (CP2K `XC_FUNCTIONAL PADE` ↔ ATV `XcKind::PADE`) |
| Basis O | **SZV-GTH-PADE** (exponents/coefficients verified identical, ATV table ↔ CP2K `BASIS_SET` file) |
| Basis H | **DZV-GTH-PADE** (verified identical) |
| PP | GTH-PADE-q6 (O), GTH-PADE-q1 (H) |
| CUTOFF | 300 Ry (CP2K REL_CUTOFF 60, NGRIDS 4; ATV single grid) |
| SCF | EPS_SCF 1e-7 (CP2K OT/DIIS; ATV diagonalization + DIIS), ATV `tol = 1e-7` |
| Cell | a variable from a₀ = 3.34 Å (4% expanded); b = c = 8 Å fixed |
| Relax | ATV: α = 1.0, bulk_mod = 0.15 eV/Å³ (= measured dσ_xx/dε at a\*, near-Newton), max_scale 0.05, strain-FD h = 2e-3, stress axes {x} |
| Converge | `StressConverged{0, 5e-4 eV/Å³, x-only}` (= 2.7×10⁻⁶ Ha/Bohr³) |

MOLOPT re-twin is queued follow-on work for the GPW perf campaign.

## Offline CP2K pins (CP2K 2026.1, never launched from ctest)

Engine: **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`),
Arrhenius epyc login node, 1 rank × 4 OMP. Inputs:
`tools/cp2k_oracle/inputs/gpw-variable-cell-ice/` (`chain_scan_template.inp`,
`chain_stress_num.inp`, `chain_cellopt.inp`).

### Headline pin — CELL_OPT (fixed ions, CONSTRAINT YZ, from a₀ = 3.34 Å)

| Quantity | Value |
| -------- | ----- |
| Relaxed chain axis | **a\* = 3.211148 Å** (b = c stayed exactly 8.0) |
| Final energy | **E = −17.068354201408205 Ha** |
| Wall (analytic stress CELL_OPT) | **3.9 s** |

### Cross-check 1 — E(a) scan (fixed SCALED coords, RUN_TYPE ENERGY)

| a [Å] | E [Ha] |
| ----- | ------ |
| 2.70 | −17.045452431317443 |
| 2.80 | −17.054908578734924 |
| 2.90 | −17.061333800406835 |
| 3.00 | −17.065404074088541 |
| 3.10 | −17.067590375788491 |
| 3.16 | −17.068196163707853 |
| 3.20 | −17.068345641019036 |
| 3.22 | −17.068351206038702 |
| 3.26 | −17.068234622883431 |
| 3.30 | −17.067960699998029 |
| 3.40 | −17.066676425438327 |
| 3.50 | −17.064657525590967 |
| 3.60 | −17.062077594476058 |

Cubic fit (3.10–3.30): a\* = 3.2124 Å; quartic (full range): 3.2126 Å;
E_min(fit) = −17.068355 Ha. Curvature d²E/da² = 0.111 Ha/Å² →
dσ_xx/dε(a\*) = −0.151 eV/Å³ (the regression's `bulk_mod = 0.15`).

### Cross-check 2 — `STRESS_TENSOR NUMERICAL` (same-method FD twin)

CP2K prints bar; 1 Ha/Bohr³ = 2.94210107994716×10⁸ bar (CP2K table).

| a [Å] | σ_xx [bar] | σ_xx [Ha/Bohr³] | wall |
| ----- | ---------- | ---------------- | ---- |
| 3.10 | +9.69527097361×10³ | +3.2954×10⁻⁵ | 8.2 s |
| 3.26 | −3.36748937738×10³ | −1.1446×10⁻⁵ | 7.1 s |

Linear zero crossing: a\* ≈ 3.219 Å. (σ_xz is nonzero — the chain breaks xz
shear symmetry — but shear is out of scope: orthorhombic v1, `KEEP_ANGLES`.)

Consensus spread over the three routes (3.2111 / 3.2124 / 3.219): ≤ 0.008 Å,
well inside the 0.02 Å gate.

### Single-SCF wall (fair-ratio denominator)

CP2K `RUN_TYPE ENERGY` at a = 3.21: E = −17.068353959979657 Ha,
wall **0.949 s**.

## Hard gates (regression asserts)

| Gate | Tolerance | Measured (2026-07-25, Release, 4 OMP) |
| ---- | --------- | ------------------------------------- |
| relax converged via criterion | — | `stress_converged` ✓ |
| \|a_ATV − a\*_CP2K(CELL_OPT)\| | ≤ 0.02 Å | **1.0×10⁻⁴ Å** (3.211249 vs 3.211148) |
| b, c untouched | bit-identical | exact 8.0 / 8.0 ✓ |
| residual \|σ_xx\| | ≤ 1.2×stol = 6×10⁻⁴ eV/Å³ | **4.4×10⁻⁷ eV/Å³** (2.4×10⁻⁹ Ha/Bohr³) |
| \|E_ATV − E_CP2K\| | ≤ 1×10⁻³ Ha | **3.5×10⁻⁵ Ha** |
| wall | ≤ 60 s | **5.1 s** |

Tolerance justification: a-gate covers the CP2K-internal consensus spread
(0.008 Å) plus strain-FD h and ATV-vs-CP2K grid differences; energy gate is
the sibling GPW soft-|ΔE| level (grid/multigrid representation differences,
measured 3.5×10⁻⁵ Ha here); stress gate is ~2% of the initial |σ_xx(a₀)|.

## Perf gate

Fair wall, both engines solving the same constrained relax from a₀ = 3.34 Å
(1 rank × 4 OMP, Arrhenius epyc login):

| Metric | Value |
| ------ | ----- |
| T_cp2k (CELL_OPT, analytic stress) | **3.9 s** |
| T_atv (regression: relax with strain-FD stress, 3 SCFs/launch) | **5.1 s** |
| ratio T_atv/T_cp2k | **1.29** |
| ratio vs CP2K single-SCF ENERGY (0.949 s) | 5.3 |
| within_10x | **true** |
| perf_exempt | false |

ATV pays FD stress (base + 2 strained SCFs per launch, warm-started) vs
CP2K's analytic stress; masked axes (`stress_axes`) already cut 4 of 7 SCFs
per launch.

## Oracle commands (offline)

```bash
module load buildenv-gcc/2026.03-mpich
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
export OMP_NUM_THREADS=4
cd tools/cp2k_oracle/inputs/gpw-variable-cell-ice
# scan: sed "s/AAA/<a>/g" chain_scan_template.inp; stress: chain_stress_num.inp
mpirun -np 1 cp2k.psmp -i chain_cellopt.inp -o cellopt.out
# rg "CELL| Vector a" cellopt.out ; rg "ENERGY|" cellopt.out
```

## Named debt found while landing this slug (reported, not silent)

1. **`kStress` convention divergence (pre-existing):** classical pair/bond
   writers accumulate the extensive virial [eV]; PW/GPW evaluators write
   intensive stress [eV/Å³] on the same meta key.
   `IsotropicCellRelax` / `PressureConverged` divide by V (virial
   convention). This slug makes the convention explicit at the read site
   (`AnisotropicCellRelax`, `StressConverged` docs);
   unifying the schema key is a dedicated follow-up.
2. **`copy_gpw` dropped `d4` / `plus_u` / `atom_spin_guess`** in
   `internal/gpw_stress.cpp` — strained stress SCFs silently lost D4/DFT+U/
   spin-seed terms when enabled. Fixed in this slug.
3. **Periodic MOLOPT image-sum cost:** ~10 min per 3-SCF launch on a 3-atom
   chain (vs CP2K ~1 s/SCF) — the blocker that forced this twin's basis
   pivot and previously forced `gpw-pbesol-si` to MT. GPW perf campaign
   item.
4. **`dev` HEAD does not configure in a clean checkout** (a6865f43):
   `CMakeLists.txt` references five sources
   (`src/cpu/observe/charge_autocorr.cpp`, `elastance_observer.cpp`,
   `src/cpu/dynamics/const_pot_maze.cpp`, `fix_dfield.cpp`, `fix_move.cpp`)
   that exist only as untracked in-flight metalwalls files in the staff tree.
   This worktree carries uncommitted local copies purely to build; they are
   **not** part of this branch's commit. The metalwalls loop must land them.
