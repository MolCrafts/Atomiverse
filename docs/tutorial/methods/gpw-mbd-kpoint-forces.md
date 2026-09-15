# GPW multi-k analytic forces + MBD@rsSCS

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-mbd-kpoint-forces` |
| Regression | `regressions/gpw/gpw-mbd-kpoint-forces/` |
| Public API | `cpu::gpw::GpwDft::mbd` + `k_mesh` / `GPWEvaluator::mbd`; multi-k `gpw_forces_from_scf` / `GPWEvaluator(compute_forces=true)`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion + §10 Periodic KS |

## Summary

**Monkhorst–Pack multi-k GPW** ionic forces with additive **MBD@rsSCS**
(geometry-only free-atom volumes). Closes the residual that Γ
`assemble_gpw_forces` already folded MBD (`gpw-mbd-forces`) while
`assemble_gpw_forces_kpoints` counted MBD for mutual exclusion but **skipped**
adding \(F_{\mathrm{MBD}}\). The product path is:

1. Multi-k SCF (`GpwDft::k_mesh`) supplies electronic \(P_k\), \(W_k\).
2. Analytic k-summed Hellmann–Feynman / Pulay assembly (same as
   `gpw-kpoint-forces`).
3. Geometry-only \(F_{\mathrm{MBD}} = -\nabla E_{\mathrm{MBD}}\) from
   `mbd_rsscs_pbe` is **added** on the multi-k path (identical switch as Γ:
   isolated when `mbd_k_mesh={0,0,0}`; Ewald + MBD k-mesh when non-zero).

Public surface: bool `mbd` on `GpwDft` / `GPWEvaluator` plus electronic
`k_mesh` (no `*Config` bag, no new force-API symbols). Hard gates:
**libMBD energy** (water dimer free-atom) + **force dual-gate** + **multi-k
wiring** on compressed H₂. Wall ratio is **perf_exempt** (same policy as
`gpw-mbd-forces` / `gpw-kpoint-forces`).

## Method

**MBD@rsSCS** (Tkatchenko et al. range-separated self-consistent screening)
builds a many-body dispersion energy from free-atom polarizabilities
\(\alpha_0\), \(C_6\), and \(R_{\mathrm{vdW}}\). For this pin, volume ratios
are **free-atom** (\(V/V_{\mathrm{free}} = 1\)); Hirshfeld scaling is a
sibling product (`gpw-mbd-hirshfeld` / `gpw-hirshfeld-kpoints`), not this slug.

**PBE MBD parameters:**

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(\beta\) | 0.83 | dimensionless |
| Fermi \(a\) | 6 | dimensionless |

Free-atom tables: `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).

**Force product (multi-k):**

\[
\mathbf{F}_A^{\mathrm{tot}}
  = \mathbf{F}_A^{\mathrm{k\text{-}HF/Pulay}}
  + \mathbf{F}_A^{\mathrm{MBD}},
\quad
\mathbf{F}_A^{\mathrm{MBD}} = -\nabla_{R_A} E_{\mathrm{MBD}}.
\]

Because free-atom MBD is geometry-only, it does **not** change the SCF density.
Therefore the multi-k wiring identity must hold bit-for-bit within tol:

\[
\mathbf{F}^{k+\mathrm{MBD}} - \mathbf{F}^{k,\mathrm{plain}}
  \approx \mathbf{F}_{\mathrm{MBD}}^{\mathrm{geom}}.
\]

**BC switch (same as Γ MBD):**

| `mbd_k_mesh` | Dispersion BC |
| ------------ | ------------- |
| `{0,0,0}` (default) | Isolated free-atom MBD@rsSCS |
| non-zero | Periodic Ewald + MBD k-mesh (`gpw-mbd-periodic`) |

This regression uses the **default free-atom** path on both the water-dimer
energy/force dual-gate and the multi-k H₂ wiring cell.

**Why not CP2K QS energy as the hard pin?** CP2K 2026.1 stock
`PAIR_POTENTIAL TYPE` is only `DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via
external **libMBD** when linked. The offline oracle is therefore **libMBD
Fortran energy** + **independent central-difference force dual-gate** +
**multi-k product wiring**, same policy as sibling MBD slugs. Twin notes:
`tools/cp2k_oracle/inputs/gpw-mbd-kpoint-forces/`.

**Key exclusions.**

- Hirshfeld density volumes (`hirshfeld=true`) — separate slugs
- Periodic Ewald MBD product gate depth — `gpw-mbd-periodic`
- UKS multi-k × MBD forces
- MBD cell stress product gate
- Absolute multi-k QS \(E_{\mathrm{tot}}\) vs CP2K (OT/grid/DIIS)
- CUDA GPW
- Live CP2K / libMBD in ctest

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no `*Config`, no
`make_*` / `build_*` factories). Regression path (compressed H₂ multi-k +
free-atom MBD; water-dimer geometry dual-gate is separate in the same binary):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::gpw_forces_from_scf;
using ATV::cpu::gpw::XcKind;

// H₂ [Bohr] — same geometry as gpw-kpoint-forces (REFERENCE.md).
const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
std::vector<GpwAtom> atoms = {
    GpwAtom({5.5, 6.0, 6.0}, basis, pot, 1),
    GpwAtom({6.5, 6.0, 6.0}, basis, pot, 1),
};

GpwDft cfg;
cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0; // Bohr
cfg.cutoff_ry = 100.0;
cfg.n_occ = 1;
cfg.tol = 1e-8;
cfg.max_iter = 60;
cfg.nonperiodic = false;
cfg.local_pp_on_grid = true;
cfg.xc = XcKind::PADE;
cfg.k_mesh[0] = 1;
cfg.k_mesh[1] = 1;
cfg.k_mesh[2] = 2;   // electronic multi-k
cfg.mbd = true;      // free-atom MBD@rsSCS (mbd_k_mesh default {0,0,0})

cfg.build(atoms);
const auto& scf = cfg.run(atoms);
// scf.e_disp [Ha] finite and < 0; scf.n_kpts / C_occ_k multi-k payload
const auto F_mbd = gpw_forces_from_scf(atoms, cfg, scf);
// F_mbd = F_k-HF/Pulay + F_MBD(geom); compare to plain multi-k assembly
```

Driver / Frame path (component; same members):

```cpp
// kinds = {H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/100.0, /*max_iter=*/60,
                                 /*tol=*/1e-8, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PADE,
                                 /*nonperiodic=*/false);
eval.mbd = true;
// set electronic multi-k on the owned GpwDft / evaluator mesh members
// bind on Evaluate / Minimize / MD force slot with compute_forces=true
// eval.last_result().e_disp  [Hartree]
// public Frame energy remains eV via frame.meta.get(kEnergy)
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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-mbd-kpoint-forces
  ./build/regressions/reg_gpw-mbd-kpoint-forces
  # or: cmake --build build --target run_reg_gpw-mbd-kpoint-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::mbd`, `k_mesh`, `mbd_k_mesh`; `GpwResult::e_disp` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::mbd`, `mbd_k_mesh` |
| `include/atomiverse/cpu/gpw/internal/mbd_dispersion.h` | `mbd_rsscs_pbe` geometry kernel (private) |
| `include/atomiverse/cpu/gpw/internal/gpw_forces.h` | multi-k + Γ force assembly |
| `src/cpu/gpw/internal/gpw_forces.cpp` | multi-k path folds \(F_{\mathrm{MBD}}\) |
| `regressions/gpw/gpw-mbd-kpoint-forces/reg_gpw-mbd-kpoint-forces.cpp` | Product gate |
| `regressions/gpw/gpw-mbd-kpoint-forces/REFERENCE.md` | Offline pins + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-mbd-kpoint-forces/` | Twin notes (libMBD policy) |

## Validation (libMBD / CP2K-campaign offline pins)

Source:
[`regressions/gpw/gpw-mbd-kpoint-forces/REFERENCE.md`](../../../regressions/gpw/gpw-mbd-kpoint-forces/REFERENCE.md).
Oracle: offline **libMBD** Fortran (`cmbd_mbd_scs_energy`, git `88069c6`) for
\(E_{\mathrm{disp}}\); independent central FD for forces; multi-k wiring vs
geometry kernel. Campaign home is the **CP2K method loop** (GPW / PW /
periodic) — CP2K stock pair types do not include MBD, so the hard energy pin
is libMBD (same family CP2K links when MBD is enabled). External engine pin —
not a golden of Atomiverse’s own output; **never** launch live CP2K or libMBD
from ctest.

### A — Geometry-only MBD (water dimer, isolated free-atom)

| Item | Value |
| ---- | ----- |
| System | Water dimer (same nuclei as `gpw-mbd-forces` / `gpw-mbd-ice`) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Å→Bohr | `1/0.52917720859` (ATV) |
| Method | free-atom MBD@rsSCS, PBE \(\beta=0.83\), Fermi \(a=6\) |
| Pin line | libMBD `cmbd_mbd_scs_energy` → **−1.4928804171407961×10⁻³ Ha** |
| FD step | \(h = 10^{-5}\) Bohr (central) |

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −1.4928804171407961×10⁻³ | libMBD Fortran (git `88069c6`) | \(1\times 10^{-8}\) Ha | Hard gate A2 |
| max\|F_product − F_fd\| [Ha/Bohr] | ≤ 1×10⁻⁶ | independent FD dual-gate | \(1\times 10^{-6}\) | Hard gate A3 |
| \(\|\sum_A F_A\|\) [Ha/Bohr] | ≤ 1×10⁻⁸ | translation invariance | \(1\times 10^{-8}\) | Hard gate A4 |

### B — Multi-k product wiring (H₂, k-mesh 1×1×2)

| Item | Value |
| ---- | ----- |
| System | Compressed H₂ (same as `gpw-kpoint-forces`) |
| Geometry [Bohr] | H (5.5, 6.0, 6.0); H (6.5, 6.0, 6.0) |
| Cell | **12 × 12 × 12 Bohr**, periodic |
| Basis / PP | `DZV-GTH-PADE` + `GTH-PADE-q1` |
| XC / mesh | PADE; electronic **k = 1×1×2**; free-atom MBD (`mbd_k_mesh` default zero) |
| Occupations | RKS, `n_occ = 1` |

| Quantity | Gate | Tol | Notes |
| -------- | ---- | --- | ----- |
| SCF converged; multi-k payload | `n_kpts ≥ 1`, `C_occ_k` non-empty | hard | Γ path reports `n_kpts==0` |
| max\|F^{k+MBD} − F^{k,plain} − F_MBD^geom\| | ≤ 1×10⁻⁸ Ha/Bohr | \(1\times 10^{-8}\) | Hard gate A5 wiring |
| \|F_MBD,0x^geom\| nontrivial | > 1×10⁻⁸ Ha/Bohr | hard | avoids trivial zero-force pass |
| `e_disp` on multi-k SCF | finite and &lt; 0 | hard | geometry-only materialization |

**Tolerance rationale.** Matching free-atom tables and geometry track libMBD
energy to nano-Hartree. Analytic MBD forces must match independent central FD
to 1×10⁻⁶ Ha/Bohr. Because MBD does not enter the KS residual, the multi-k
force difference after one shared SCF must equal the standalone geometry
kernel to 1×10⁻⁸ Ha/Bohr — the regression that proved the multi-k assembly
fold.

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

MBD is \(O(N^3)\) geometry-only; multi-k H₂ force assembly is a single
gradient pass after one SCF. Full CP2K QS wall is not a fair twin (no stock
MBD pair type on CP2K 2026.1; primary gates are E_disp + FD dual-gate +
wiring).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (no stock MBD pair type) | — (not fair-twinned) | **perf_exempt** | — | libMBD energy / force dual-gate is the hard pin |
| Product gate (reg) | n/a | short multi-k SCF + MBD forces | **perf_exempt** | — | E_disp / FD / wiring deliverable |

**perf_exempt = true.** Reason (from REFERENCE / handoff / queue wave 10):
primary deliverables are offline libMBD \(E_{\mathrm{disp}}\) parity, force
dual-gate, and multi-k wiring identity. Same policy as `gpw-mbd-forces` and
`gpw-kpoint-forces`. Re-measure a strict full-SCF twin later only if product
policy requires \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on a host
with libMBD-linked CP2K and a defined fair twin.

## Limitations

- **Free-atom volumes only** on this pin — density-scaled Hirshfeld MBD is
  gated by sibling slugs, not here.
- **RKS multi-k** wiring cell (H₂); UKS × multi-k × MBD forces are out of
  scope for this regression.
- **No MBD stress product gate** on this slug (ionic forces only).
- **Electronic multi-k ≠ MBD k-mesh** — `k_mesh` is the SCF Brillouin mesh;
  `mbd_k_mesh` selects isolated vs periodic Ewald MBD (default isolated).
- **Absolute QS \(E_{\mathrm{tot}}\)** vs CP2K is not asserted (OT/grid/DIIS;
  MBD is geometry-only).
- **Mutual exclusion** with other dispersion flags (`d3` / `d3bj` / `d4` /
  `ts` / `xdm` / nonlocal vdW-DF family) — any conflicting pair throws at run.
- Wall **perf_exempt** for this wave; not a speed optimization target.
- **Public energy units:** internal Ha for pins / `e_disp`; Frame still reports
  eV on `frame.meta.get(kEnergy)`.

## References

- Offline pin: `regressions/gpw/gpw-mbd-kpoint-forces/REFERENCE.md`
- Oracle notes: `tools/cp2k_oracle/inputs/gpw-mbd-kpoint-forces/`,
  `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion, §10 Periodic KS
- Sibling force pins: `regressions/gpw/gpw-mbd-forces/`,
  `regressions/gpw/gpw-kpoint-forces/`
- Related energy pins: `regressions/gpw/gpw-mbd-ice/`,
  `regressions/gpw/gpw-mbd-periodic/`
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-mbd-kpoint-forces/` (deleted
  at done, harness wave 10 2026-07-26)
- A. Tkatchenko, R. A. DiStasio Jr., R. Car, M. Scheffler, *Phys. Rev. Lett.*
  **108**, 236402 (2012) (MBD@rsSCS)
- libMBD Fortran backend (energy pin family; git `88069c6` as cited in
  REFERENCE)
