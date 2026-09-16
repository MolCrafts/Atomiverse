# GPW multi-k analytic cell stress + periodic MBD@rsSCS

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-mbd-kpoint-stress` |
| Regression | `regressions/gpw/gpw-mbd-kpoint-stress/` |
| Public API | `cpu::gpw::GPWEvaluator` members `k_mesh`, `mbd`, `mbd_k_mesh`, `compute_stress`, `stress_use_fd`; Frame `kStress` [eV/Å³]; `last_stress_ha_bohr3()` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion + §10 Periodic KS |

## Summary

**Monkhorst–Pack multi-k GPW** analytic cell stress with **periodic
MBD@rsSCS** dispersion folded in. Closes the residual that the Γ fixed-\(P\)
stress Lagrangian already rebuilt \(E_{\mathrm{disp}}^{\mathrm{MBD}}\) at
every strained cell (`gpw-mbd-stress`) while the **k-summed** Lagrangian
(`energy_fixed_P_k`) refused MBD with a throw. The product path is:

1. One multi-k SCF (`k_mesh = {1,1,2}`, time-reversal folded) supplies
   \(\{P_k, W_k\}\).
2. Central FD of the **non-self-consistent** k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates.
3. The geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{MBD}}(\{\mathbf R\},\mathbf h)\) is
   **rebuilt at every strained cell** (same `mbd_rsscs_pbe` call as the Γ
   path), so the stress derivative sees the true strained dispersion energy.
4. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; Hartree / Bohr³ internally).

Public surface: existing members only — no `*Config` bag, no new symbols.
Frame reports `kStress` in eV/Å³; internal comparison via
`last_stress_ha_bohr3()`.

## Method

**MBD@rsSCS** (Tkatchenko et al., range-separated self-consistent screening)
with **free-atom** volume ratios (\(V/V_{\mathrm{free}} = 1\)); Hirshfeld
scaling is a sibling product, not this slug. Dispersion here is **periodic**:
Ewald lattice sum on its own MBD k-mesh **2×2×2**, independent of the
electronic SCF mesh.

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| PBE \(\beta\) | 0.83 | dimensionless |
| Fermi \(a\) | 6 | dimensionless |
| MBD k-mesh | 2×2×2 | — |
| Electronic k-mesh | 1×1×2 | Monkhorst–Pack |
| Strain step \(h\) | 2×10⁻³ | dimensionless |

Because free-atom MBD is geometry-only it does not perturb the SCF density;
the additivity identity per diagonal axis must hold:

\[
\sigma_{aa}(\mathrm{mbd}) - \sigma_{aa}(\mathrm{plain})
  = -\frac{\Delta E_{\mathrm{disp}}}{2h\,\Omega},
\]

with the right-hand side pinned externally by pymbd at each strained cell.

**Why not a CP2K stress twin?** CP2K 2026.1 stock `PAIR_POTENTIAL TYPE` is
only `DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD comes via external libMBD when
linked. The offline oracle is therefore **pure-Python pymbd**
(`mbd_energy_species`, libmbd git `88069c6`) for the dispersion energies at
the unstrained and all six strained cells, plus a **multi-SCF strain-FD**
dual gate for the total stress — same policy as `gpw-mbd-stress` /
`gpw-mbd-kpoint-forces`.

**Key exclusions.**

- Hirshfeld density volumes — sibling slugs
- UKS × multi-k × MBD stress
- Off-diagonal (shear) strain — orthorhombic diagonal only; off-diagonals
  gated exactly zero
- Absolute multi-k QS \(E_{\mathrm{tot}}\) vs CP2K
- CUDA GPW; live CP2K / libMBD / pymbd in ctest

## Implementation

Public surface only (regression product path):

```cpp
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// Periodic water dimer, 12 Å cube (regions/state on the Frame — sole token)
cpu::Frame frame(6);
// span(kAtom_type) = {8,1,1,8,1,1}; span(kAtom_x/y/z) as REFERENCE.md
frame.box = cpu::Box::cubic(12.0);   // Å, PERIODIC XYZ

const cpu::gpw::Kind O{molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

cpu::gpw::GPWEvaluator eval({O, H}, /*cutoff_ry=*/120.0, /*max_iter=*/60,
                            /*tol=*/1e-8, /*mix=*/0.3,
                            cpu::gpw::XcKind::PBE, /*nonperiodic=*/false);
eval.k_mesh[2] = 2;            // electronic multi-k 1×1×2
eval.mbd = true;               // MBD@rsSCS, free-atom volumes
eval.mbd_k_mesh[0] = eval.mbd_k_mesh[1] = eval.mbd_k_mesh[2] = 2;
eval.compute_stress = true;    // analytic fixed-P_k Lagrangian
eval.stress_use_fd = false;

cpu::Evaluate ev;
ev.on(evaluate::Phase::evaluate).add_ref(eval);
ev.build();
ev.run(frame);
// frame.meta.get(kEnergy) [eV]; frame kStress [eV/Å³]
// eval.last_stress_ha_bohr3() for Ha/Bohr³ comparison
// eval.last_result().e_disp [Ha]
```

Build and run the regression (scoped targets only):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-mbd-kpoint-stress
  ./build/regressions/reg_gpw-mbd-kpoint-stress
  # or: cmake --build build --target run_reg_gpw-mbd-kpoint-stress
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `k_mesh`, `mbd`, `mbd_k_mesh`, `compute_stress`, `stress_use_fd` |
| `include/atomiverse/cpu/gpw/internal/gpw_stress.h` | fixed-\(P_k\) Lagrangian; multi-k MBD fold (was a throw) |
| `include/atomiverse/cpu/gpw/internal/mbd_dispersion.h` | `mbd_rsscs_pbe` geometry kernel (private) |
| `regressions/gpw/gpw-mbd-kpoint-stress/reg_gpw-mbd-kpoint-stress.cpp` | Product gate |
| `regressions/gpw/gpw-mbd-kpoint-stress/REFERENCE.md` | Offline pins + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-mbd-kpoint-stress/pin_mbd_strain.py` | pymbd pin generator (never at ctest) |

## Validation (pymbd / CP2K-campaign offline pins)

Source:
[`regressions/gpw/gpw-mbd-kpoint-stress/REFERENCE.md`](../../../regressions/gpw/gpw-mbd-kpoint-stress/REFERENCE.md).
Oracle: offline **pure-Python pymbd** `mbd_energy_species` (libmbd git
`88069c6`) for the dispersion energies; **multi-SCF strain-FD**
(`gpw_stress_fd`, full re-SCF at each strained cell on the same 1×1×2 mesh)
for the total stress. Campaign home is the CP2K method loop; CP2K 2026.1
stock has no MBD pair type, so pymbd is the hard external pin. Never seeded
from ATV output; never launched at ctest time.

### System

| Item | Value |
| ---- | ----- |
| System | Water dimer (6 atoms, O H H O H H), same nuclei as `gpw-mbd-periodic` / `gpw-mbd-stress` |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ`; Å→Bohr = `1/0.52917720859` |
| Basis / PP | `DZVP-MOLOPT-SR-GTH`; O `GTH-PBE-q6`, H `GTH-PBE-q1` (16 valence e⁻, `n_occ = 8`) |
| XC / cutoff | PBE, 120 Ry |
| Dispersion | MBD@rsSCS free-atom, \(\beta=0.83\), \(a=6\), MBD k-mesh 2×2×2 |

### pymbd pin lines (offline; strain h = 2×10⁻³, fixed fractional coords)

```
E_disp unstrained            = -1.4965720704989138e-03 Ha
E_disp x +h              = -1.4925684743305467e-03 Ha
E_disp x -h              = -1.5005781658326001e-03 Ha
sigma_MBD_xx (FD, pymbd)    = -1.7171785426e-07 Ha/Bohr3
E_disp y +h              = -1.4965139646472281e-03 Ha
E_disp y -h              = -1.4966306893944292e-03 Ha
sigma_MBD_yy (FD, pymbd)    = -2.5024338482e-09 Ha/Bohr3
E_disp z +h              = -1.4957889518587564e-03 Ha
E_disp z -h              = -1.4973531298823062e-03 Ha
sigma_MBD_zz (FD, pymbd)    = -3.3534037337e-08 Ha/Bohr3
```

The unstrained value is bit-identical to the `gpw-mbd-periodic` pin
(provenance cross-checked at pin time, 2026-07-31).

### Gates and measured results (2026-07-31, first green)

| Gate | Quantity | Oracle | Tol | Measured |
| ---- | -------- | ------ | --- | -------- |
| A | \(E_{\mathrm{disp}}\) multi-k vs Γ pin (k-mesh independence) | pymbd | 1×10⁻⁸ Ha | **1.07×10⁻¹¹** |
| B | \(E_{\mathrm{disp}}\) at 6 strained cells | fresh pymbd pins | 1×10⁻⁸ Ha | **1.78×10⁻¹⁵** |
| C | \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF strain-FD | `gpw_stress_fd` | 2×10⁻⁵ Ha/Bohr³ | **1.36×10⁻⁹** |
| D | \(\sigma(\mathrm{mbd})-\sigma(\mathrm{plain})\) vs pymbd FD term | pymbd | 2×10⁻⁹ Ha/Bohr³ | **4.00×10⁻¹²** |
| E | off-diagonals (orthorhombic) | exact | 0 | **0** |
| E | `kStress` eV/Å³ conversion | units | 1×10⁻⁹ | exact |

Measured diagonal stress (mbd, analytic, 120 Ry):
\(\sigma_{xx} = -3.622\times10^{-5}\), \(\sigma_{yy} = -8.570\times10^{-5}\),
\(\sigma_{zz} = -3.117\times10^{-6}\) Ha/Bohr³; MBD shifts per axis
(−1.7171×10⁻⁷, −2.500×10⁻⁹, −3.353×10⁻⁸ Ha/Bohr³) match the pymbd FD terms.

**Tolerance rationale.** Matching free-atom tables + geometry track pymbd to
femto-Hartree on the strained cells. Analytic vs multi-SCF FD carries SCF
re-convergence noise → the Γ MBD-stress tier 2×10⁻⁵ Ha/Bohr³ (measured 4
orders better). Gate D residual is bounded by pin round-off
\(\le 2\cdot10^{-8}/(2h\Omega) \approx 4\times10^{-10}\) plus SCF noise.

## Performance

Source: `REFERENCE.md` (**perf_exempt**).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS stress (context) | n/a (no stock MBD pair type in CP2K 2026.1) | — | **perf_exempt** | — | pymbd + strain-FD dual gate is the hard pin |
| Product gate (reg, ≤60 s budget) | n/a | **13.6** (analytic; FD twin 22.0; SCF 11 iters, nk = 1) | **perf_exempt** | — | wall budget met with 4.6× margin |

**perf_exempt = true.** No fair oracle wall exists for an MBD stress run
(CP2K stock has no MBD); same policy as `gpw-mbd-stress` /
`gpw-mbd-kpoint-forces`. Re-measure only if a libMBD-linked CP2K fair twin is
defined.

## Limitations

- **Free-atom volumes only** — Hirshfeld-scaled MBD stress is out of scope.
- **RKS multi-k** only; UKS × multi-k × MBD stress not gated here.
- **Diagonal (orthorhombic) strain only** — shear components are gated
  exactly zero for this cell, not validated for general cells.
- **Fixed-\(P_k\)** Hellmann–Feynman stress: the Lagrangian FD is
  non-self-consistent by design; the multi-SCF FD oracle bounds the error.
- **Electronic `k_mesh` ≠ `mbd_k_mesh`** — SCF Brillouin mesh vs MBD Ewald
  lattice-sum mesh (default `{0,0,0}` = isolated).
- **Mutual exclusion** with `d3`/`d3bj`/`d4`/`ts`/`xdm`/vdW-DF — conflicting
  pairs throw at run.
- Absolute QS \(E_{\mathrm{tot}}\) vs CP2K not asserted.
- Public energy/stress units: internal Ha, Ha/Bohr³ for pins; Frame reports
  eV (`kEnergy`) and eV/Å³ (`kStress`).

## References

- Offline pin: `regressions/gpw/gpw-mbd-kpoint-stress/REFERENCE.md`
- Pin generator: `tools/cp2k_oracle/inputs/gpw-mbd-kpoint-stress/pin_mbd_strain.py`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion, §10 Periodic KS
- Sibling stress pins: `regressions/gpw/gpw-mbd-stress/`,
  `regressions/gpw/gpw-kpoint-stress/`
- Sibling force / energy pins: `regressions/gpw/gpw-mbd-kpoint-forces/`,
  `regressions/gpw/gpw-mbd-periodic/`
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- A. Tkatchenko, R. A. DiStasio Jr., R. Car, M. Scheffler, *Phys. Rev. Lett.*
  **108**, 236402 (2012) (MBD@rsSCS)
- pymbd / libMBD (git `88069c6`), `src/pymbd/pymbd.py` `mbd_energy_species`
