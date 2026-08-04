# Reference — GPW OT vs DIIS on isolated water (MOLOPT)

## Method

Closed-shell Γ-point **GPW** DFT with **PBE**, **DZVP-MOLOPT-GTH** basis and
**GTH-PBE** pseudopotentials, **Martyna–Tuckerman** isolated electrostatics
(`nonperiodic = true`). Atomiverse product path:

- OT: `cpu::gpw::GPWEvaluator::use_ot = true` (Orbital Transformation SCF)
- DIIS: `use_ot = false` (default diagonalization + Pulay DIIS)

OT formulation (in-tree): Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020)
§VII C; VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003).

Public units on Frame: energy [eV] via `kEnergy`. Method result channel:
`last_energy_hartree()` / `GpwResult::e_total` [Hartree].

## System

CP2K getting-started `h2o` geometry in a **10 Å** cubic cell (coordinates [Å]):

| Atom | x | y | z |
| ---- | ---: | ---: | ---: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

Settings: cutoff **250 Ry** (regression budget; 400 Ry tutorial class is
`examples/09_gpw.cpp`), `max_iter=200`, `tol=1e-7`, mix `0.3` (DIIS path).

## Offline CP2K oracle

- **Version:** CP2K **2026.1** (`CP2K/2026.1-gmpflf-2026.03` on Arrhenius)
- **Binary:** `cp2k.psmp` (1 MPI rank, `OMP_NUM_THREADS=4`)
- **Data:** `$CP2K_ROOT/share/cp2k/data/{BASIS_MOLOPT,GTH_POTENTIALS}`
- **Inputs (in-tree):**
  - `tools/cp2k_oracle/inputs/ot-minimizer-water-ot.inp` — `&OT` +
    `PRECONDITIONER FULL_SINGLE_INVERSE`, `MINIMIZER DIIS`, outer SCF, **CUTOFF 250**
  - `tools/cp2k_oracle/inputs/ot-minimizer-water-diag.inp` — traditional
    diagonalization + Broyden mixing (no `&OT`)

### Pinned energies (2026-07-23, CUTOFF 250)

| Path | \(E_{\mathrm{tot}}\) [Ha] | Notes |
| ---- | ------------------------: | ----- |
| **CP2K OT** | **−17.212848291913666** | primary pin |
| Atomiverse OT | −17.2118583480 | 31 OT iters (OMP=1) |
| Atomiverse DIIS | −17.2118583479 | 9 DIIS iters; \|OT−DIIS\| ~4e-11 Ha |

```text
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -17.212848291913666
```

At 400 Ry (tutorial twin, not the regression pin) CP2K OT was
−17.212337205347978 (19 steps); DIAG −17.212337010579070 — retained as
cross-check of the energy class.

### Exact offline commands

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=.../CP2K/2026.1-gmpflf-2026.03/share/cp2k/data
# patch BASIS/POTENTIAL paths to absolute in the .inp copies, then:
cp2k.psmp -i ot-minimizer-water-ot.inp -o ot.out
```

**ctest never launches CP2K.**

## Regression gates

| Check | Tolerance |
| ----- | --------: |
| \(\|E_{\mathrm{ATV,OT}} - E_{\mathrm{CP2K,OT}}\|\) | **1.5×10⁻³ Ha** |
| \(\|E_{\mathrm{ATV,OT}} - E_{\mathrm{ATV,DIIS}}\|\) | **1×10⁻⁴ Ha** |

Milli-Ha pin tol at 250 Ry absorbs grid/multigrid defaults that still differ
from CP2K (NGRIDS/REL_CUTOFF/collocation). OT vs DIIS in-tree is near machine
parity on the same grids. Product path stays **single-grid** on this twin:
product multigrid is a net loss for 3-atom / 10 Å at 250 Ry.

### Fair wall (2026-08-04, OMP=4 bind, arrhenius1)

| Side | Wall [s] | Notes |
| --- | ---: | --- |
| ATV reg (OT + DIIS dual) | **~9.6** | 28 OT + 9 DIIS KS builds |
| ATV OT-only (prof) | **~5.5–6** | setup ~0.7 + 28×~0.18 s |
| CP2K OT quiet | **~4.3** | 19 OT steps, NGRIDS 4 |

Dominant KS cost: GGA spectral `gradient`/`divergence` FFT (~60% of KS).
Next levers: OT-DIIS / OUTER with mixing (cut outer KS count), faster XC FFT.

Measured (2026-07-23): \(\|E_{\mathrm{OT}}-E_{\mathrm{CP2K}}\|=9.90\times10^{-4}\) Ha;
\(\|E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\|\approx 4\times10^{-11}\) Ha.

## Fair wall (perf gate)

Comparable settings (cutoff 250 Ry, PBE, MT, single rank, OMP=4):

| Twin | Wall [s] | Notes |
| ---- | -------: | ----- |
| CP2K OT | **4.31** | `cp2k.psmp`, OMP=4 |
| ATV OT+DIIS (regression binary) | **41.73** | OMP=4, both paths |
| ATV OT+DIIS | **54.01** | OMP=1 (≤60 s gate) |

Fair OT-only estimate (OMP=4): ATV OT ≈ 39 s from regression timestamps →
\(r \approx 39/4.31 \approx \mathbf{9.0}\) ≤ 10. Full regression (OT+DIIS)
vs CP2K OT alone is not a fair twin; report OT-only ratio.

`within_10x=true` (fair OT, OMP=4).
