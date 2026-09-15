# Γ-UKS GPW Hirshfeld-scaled MBD@rsSCS — O₂ / PBE / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-hirshfeld-uks` |
| Regression | `regressions/gpw/gpw-hirshfeld-uks/` |
| Public API | `cpu::gpw::GpwDft` / `cpu::gpw::GPWEvaluator`: `n_alpha`/`n_beta`, `mbd`, `hirshfeld`; `GpwResult::e_disp` [Ha], `volume_ratios` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion — Hirshfeld-scaled MBD (Γ-UKS product) |

## Summary

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) from the final **Γ-point
unrestricted (UKS) GPW PBE** total valence density
\(\rho=\rho_\alpha+\rho_\beta\) scale free-atom **MBD@rsSCS** polarizabilities
and \(C_6\) (Tkatchenko–Scheffler family). This product pin is **triplet
O₂** under Martyna–Tuckerman isolation: same centered geometry as
`gpw-uks-pbe` / `gpw-uks-o2-gamma`, with post-SCF MBD on the UKS density
path rather than free-atom volumes.

Public path: construct `cpu::gpw::GpwDft` (or `GPWEvaluator`) with one
`Kind` / `GpwAtom` per site via lookup constructors `GaussianBasis` /
`GthPotential`, set members (`n_alpha`, `n_beta`, `mbd`, `hirshfeld`,
`cutoff_ry`, `xc`, `nonperiodic`, …), `build` → `run`. Dispersion lands on
`GpwResult::e_disp` [Hartree] and is folded into `e_total`; ratios on
`volume_ratios`. **No** public `*Config` bag and **no** `make_*` /
`build_*` factories. Product gate: `reg_gpw-hirshfeld-uks`. Fair wall is
**perf_exempt** (E_disp pin; stock CP2K has no TS/MBD twin).

Related RKS footholds: free-atom MBD (`gpw-mbd-ice`), Hirshfeld RKS
(`gpw-mbd-hirshfeld`), multi-k Hirshfeld RKS (`gpw-hirshfeld-kpoints`).
Open-shell energy/spin baseline: `gpw-uks-o2-gamma` / `gpw-uks-pbe`.

## Method

**GPW host (Quickstep family).** Contracted Gaussian AOs; density on a
uniform grid; FFT Poisson Hartree; XC on the grid; GTH pseudopotentials.
Citation: Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997).

**UKS / collinear spin.** Separate \(\rho_\alpha\), \(\rho_\beta\) and
\(H_\alpha\), \(H_\beta\); Hartree from total \(\rho\); integer occupations
`n_alpha` / `n_beta`. For O GTH-PBE-q6, 6 valence e⁻ per O → 12 total;
multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\). Löwdin
\(\langle S^2\rangle\) is available on the result but is **not** the hard
gate of this slug (that is E_disp).

**Hirshfeld volumes (in-tree free-atom model).** After SCF, partition the
**total** valence density with exponential free-atom prototypes from
`basis_data/hirshfeld_tables.h`:

\[
\rho_Z^{\mathrm{free}}(r)=A\exp(-\alpha_Z r),\quad
A=N_{\mathrm{val}}\alpha_Z^3/(8\pi),\quad
V_Z^{\mathrm{free}}=60\,N_{\mathrm{val}}/\alpha_Z^3.
\]

Oxygen: \(N=6\), \(\alpha=2.2\) Bohr⁻¹. Ratio \(v_A = V_A / V_A^{\mathrm{free}}\)
with \(V_A = \int w_A(\mathbf{r})\,\rho(\mathbf{r})\,d\mathbf{r}\) and
Hirshfeld weights \(w_A = \rho_A^{\mathrm{free}} / \sum_B \rho_B^{\mathrm{free}}\).
**UKS uses \(\rho=\rho_\alpha+\rho_\beta\)** (not spin-resolved volumes).

**MBD@rsSCS scaling.** Free-atom α₀ / C₆ / R_vdW (libMBD tables) scale as
\(\alpha\propto v\), \(C_6\propto v^2\), \(R\propto v^{1/3}\); then
range-separated self-consistent screening MBD (PBE β=0.83). Citation:
Tkatchenko & Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009);
Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014). On this slug
MBD is **isolated** on the nuclear coordinates (`mbd_k_mesh = {0,0,0}`);
periodic Ewald MBD is a separate product (`gpw-mbd-periodic`).

**API contract.**

- `mbd=true`, `hirshfeld=true` → post-SCF volumes + scaled MBD →
  `e_disp`, `volume_ratios`.
- `mbd=true`, `hirshfeld=false` → free-atom volumes (\(v=1\)); geometry-only
  MBD (same tables as `gpw-mbd-ice`).
- `hirshfeld=true` **without** `ts` or `mbd` throws
  (`"hirshfeld requires ts or mbd"`).
- `hirshfeld` does **not** apply to D3/D3(BJ)/D4/XDM/vdW-DF/rVV10.

**Why not stock CP2K QS?** CP2K 2026.1 `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no stock TS/MBD. The offline pin is the
**ATV product** under fixed UKS settings, cross-checked vs free-atom MBD
and the dual in-tree Hirshfeld kernel. Notes live under
`tools/cp2k_oracle/inputs/gpw-hirshfeld-uks/` (not a live oracle).

**Key exclusions (product scope).**

- No CP2K QS energy twin for \(E_{\mathrm{disp}}\) (no stock MBD).
- No standalone Hirshfeld (must pair with `ts` or `mbd`).
- No spin-resolved Hirshfeld volumes (total density only).
- No multi-k UKS Hirshfeld on this regression (multi-k Hirshfeld is RKS:
  `gpw-hirshfeld-kpoints`; UKS multi-k energy/forces are other slugs).
- Absolute QS \(E_{\mathrm{tot}}\) micro-Ha parity out of scope (grid/DIIS
  vs CP2K multigrid).
- No CUDA GPW; no analytic UKS dispersion forces as the gate.

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; **no** `*Config`,
**no** `make_*` / `build_*`). Regression uses the concrete solver:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

constexpr double kA2B = 1.0 / 0.52917720859;
// O₂, R = 1.2075 Å, centered in 12 Å cubic cell (MT / nonperiodic)
constexpr double L_ang = 12.0;
constexpr double R_ang = 1.2075;
const double L = L_ang * kA2B;
const double z0 = 0.5 * L - 0.5 * R_ang * kA2B;
const double z1 = 0.5 * L + 0.5 * R_ang * kA2B;

const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-GTH");
const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
std::vector<GpwAtom> atoms;
{
    const double p0[3] = {0.5 * L, 0.5 * L, z0};
    const double p1[3] = {0.5 * L, 0.5 * L, z1};
    atoms.emplace_back(p0, bO, pO, 8);
    atoms.emplace_back(p1, bO, pO, 8);
}

GpwDft gpw;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0;
gpw.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
gpw.n_beta = 5;
gpw.max_iter = 150;
gpw.tol = 1e-6;
gpw.mix = 0.35;
gpw.diis_depth = 10;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = true;
gpw.mbd = true;
gpw.hirshfeld = true;

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha], res.volume_ratios, res.e_total, res.s2, res.converged
```

Equivalent evaluator members: `eval.mbd = true; eval.hirshfeld = true;`
with `eval.n_alpha` / `eval.n_beta` (Frame path via `cpu::single_point`).

Build (GPW needs FFTW3) and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-hirshfeld-uks
  ./build/regressions/reg_gpw-hirshfeld-uks
  # or: cmake --build build --target run_reg_gpw-hirshfeld-uks
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Solver members | `include/atomiverse/cpu/gpw/gpw_solver.h` (`mbd`, `hirshfeld`, `GpwResult`) |
| Evaluator mirror | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Hirshfeld volumes | `include/atomiverse/cpu/gpw/internal/hirshfeld_volumes.h`, `src/cpu/gpw/internal/hirshfeld_volumes.cpp` |
| Free-atom tables | `include/atomiverse/cpu/gpw/basis_data/hirshfeld_tables.h` |
| MBD kernel | `include/atomiverse/cpu/gpw/internal/mbd_dispersion.h` |
| UKS SCF + post-SCF fold | `src/cpu/gpw/gpw_solver.cpp` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_hirshfeld.cpp` |
| Regression | `regressions/gpw/gpw-hirshfeld-uks/reg_gpw-hirshfeld-uks.cpp` |
| Oracle notes | `tools/cp2k_oracle/inputs/gpw-hirshfeld-uks/o2_uks_hirshfeld.note` |

## Validation (offline product pin)

Source: `regressions/gpw/gpw-hirshfeld-uks/REFERENCE.md` (hard-coded;
**never** live CP2K/libMBD in ctest). Free-atom MBD tables already pinned
vs libMBD Fortran on `gpw-mbd-ice`; this slug gates the **UKS volume-ratio
layer**.

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | **−1.0379964698969957×10⁻⁴** | **1×10⁻⁸ Ha** | PBE-MBD@rsSCS, Hirshfeld \(v_A\), CUTOFF 200 Ry |
| \(v\) (O, O) | ≈ **0.6666, 0.6666** | physical bounds (0.2, 2.0) in reg | compressed vs free-atom \(v=1\) |
| SCF | `res.converged` | — | `max_iter=150`, `tol=1e-6` |
| \(E_{\mathrm{disp}} < 0\) | attractive | — | Hirshfeld reduces \|E_disp\| vs free-atom MBD |
| Closed-shell UKS≈RKS (unit) | water dimer E_disp | 5×10⁻⁶ Ha | `test_cpu_gpw_hirshfeld` (Nα=Nβ path) |

**Geometry / protocol (must match REFERENCE).**

| Item | Value |
| ---- | ----- |
| Molecule | O₂, bond along *z*, **centered** |
| \(R(\mathrm{O–O})\) | **1.2075 Å** |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`nonperiodic`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| XC | PBE |
| Cutoff | **200 Ry** |
| Spin | UKS, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| Dispersion | `mbd=true`, `hirshfeld=true` (isolated MBD) |
| SCF | max_iter 150, tol 1e-6, mix 0.35, diis_depth 10 |

Cross-checks (offline operator notes only):

| Source | Note |
| ------ | ---- |
| Free-atom MBD (`v=1`) on same O₂ | more attractive than Hirshfeld-scaled |
| Dual in-tree Hirshfeld kernel | free-atom limit / volume-scaled MBD unit gates |
| CP2K QS | **no** stock TS/MBD — not used as E_disp oracle |

## Performance

Source: closed-spec / handoff policy (the case `REFERENCE.md`,
wave 7 2026-07-26). **Not** a ctest gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| O₂ UKS-PBE Hirshfeld-MBD | — | SCF-dominated | **perf_exempt** | n/a | No stock CP2K MBD twin; primary deliverable is E_disp + \(v_A\) |

Gate: `perf_exempt = true` (same policy as `gpw-mbd-hirshfeld` /
`gpw-hirshfeld-kpoints`). Hirshfeld grid work is
\(O(N_{\mathrm{grid}} N_{\mathrm{atoms}})\); wall is dominated by UKS SCF.

## Limitations

- **Product pin, not CP2K QS bit-identity:** stock Quickstep lacks TS/MBD;
  do not expect a live CP2K E_disp twin in the oracle tree.
- **Total-density Hirshfeld only:** open-shell spin density does not enter
  \(v_A\); \(\rho_\alpha-\rho_\beta\) effects appear only through the
  converged total \(\rho\).
- **Isolated MBD geometry:** nuclear MBD is non-periodic on this pin even
  though the SCF cell is MT-isolated; periodic Ewald MBD is
  `gpw-mbd-periodic` (closed-shell product).
- **CUTOFF 200 Ry** is the product pin setting (faster than the 400 Ry
  UKS-PADE energy foothold); retuning cutoff changes both density and
  \(v_A\) slightly — re-pin before tightening tol.
- **Geometry hygiene:** off-center molecules under MT corrupt the density
  tails and thus volumes; always center isolated pins.
- **Follow-ons not this page:** multi-k UKS + Hirshfeld as a dedicated
  energy/stress product; spin-projected volumes; TS+Hirshfeld open-shell
  forces as a separate regression gate.

## References

- Offline pin: [`regressions/gpw/gpw-hirshfeld-uks/REFERENCE.md`](../../../regressions/gpw/gpw-hirshfeld-uks/REFERENCE.md)
- Regression consumer: [`reg_gpw-hirshfeld-uks.cpp`](../../../regressions/gpw/gpw-hirshfeld-uks/reg_gpw-hirshfeld-uks.cpp)
- Oracle notes: [`tools/cp2k_oracle/inputs/gpw-hirshfeld-uks/`](../../../tools/cp2k_oracle/inputs/gpw-hirshfeld-uks/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Related regressions: RKS Hirshfeld `regressions/gpw/gpw-mbd-hirshfeld/`,
  multi-k RKS Hirshfeld `regressions/gpw/gpw-hirshfeld-kpoints/`,
  free-atom MBD `regressions/gpw/gpw-mbd-ice/`;
  Γ-UKS energy tutorial [`gpw-uks-o2-gamma.md`](gpw-uks-o2-gamma.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md) §8
- A. Tkatchenko, M. Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009) — TS volumes
- A. Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014) — MBD@rsSCS
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
