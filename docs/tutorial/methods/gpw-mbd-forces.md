# GPW MBD@rsSCS ionic forces — water dimer (free-atom)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-mbd-forces` |
| Regression | `regressions/gpw/gpw-mbd-forces/` |
| Public API | `cpu::gpw::GpwDft::mbd` / `cpu::gpw::GPWEvaluator::mbd`; forces via `assemble_gpw_forces` / `compute_forces=true`; `MbdResult::{energy,forces}`; `GpwResult::e_disp` [Ha] |
| Related energy pin | `gpw-mbd-ice` (E_disp only); periodic mesh `gpw-mbd-periodic` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Many-body dispersion MBD@rsSCS** ionic forces on top of Γ-point **GPW**
(PBE). Free-atom volume ratios \(V/V_{\mathrm{free}}=1\) (no Hirshfeld density
chain). Same water-dimer nuclei as `gpw-mbd-ice` / `gpw-ts-ice`. Product path
evaluates geometry-only \(E_{\mathrm{disp}}\) matching offline **libMBD**
(`cmbd_mbd_scs_energy`, git `88069c6`), then returns
\(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\) [Ha/Bohr] from a high-order
central finite-difference of that same kernel; `assemble_gpw_forces` **adds**
these forces to electronic Hellmann–Feynman / Pulay terms when
`GpwDft::mbd = true` (no longer energy-only).

**Why not CP2K QS ENERGY_FORCE?** CP2K 2026.1 stock `PAIR_POTENTIAL TYPE` is
only `DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is external libMBD when linked. Hard
pin is **libMBD energy** + **force dual-gate** (product \(\nabla E\) vs
independent central FD), same policy as `gpw-mbd-ice` / `gpw-ts-ice`. Twin
notes under `tools/cp2k_oracle/inputs/gpw-mbd-forces/`.

Public surface: bool members `mbd` / `mbd_k_mesh` on `GpwDft` and
`GPWEvaluator` (no `*Config` bag). Product gate:
`regressions/gpw/gpw-mbd-forces` (`reg_gpw-mbd-forces`). Wall ratio is
**perf_exempt** (E_disp + force dual-gate; full QS twin unfair / unavailable).

## Method

**MBD@rsSCS** (Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014);
Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023) / libMBD):

1. **Range-separated self-consistent screening** of dynamical polarizabilities
   (Fermi × Gaussian dipole tensor, \(\beta\) damping; free-atom or
   volume-scaled \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\)).
2. **Long-range MBD Hamiltonian** diagonalization → zero-point energy
   difference vs non-interacting oscillators (isolated path here;
   periodic Ewald + \(k\)-mesh is `gpw-mbd-periodic`).

\[
E_{\mathrm{disp}} = E_{\mathrm{MBD@rsSCS}}\!\left(\{\mathbf{R}_A\},
  \{\alpha_0^A, C_6^A, R_{\mathrm{vdW}}^A\}\right).
\]

**Ionic forces** (geometry-only product path, free-atom / fixed volume ratio):

\[
\mathbf{F}_A^{\mathrm{MBD}}
  = -\frac{\partial E_{\mathrm{disp}}}{\partial \mathbf{R}_A}
  \quad\text{[Ha/Bohr]}.
\]

Implementation: central FD of the same energy kernel that matches libMBD
(product step \(h = 10^{-4}\) Bohr; dual-gate re-check uses independent
\(h = 10^{-5}\) Bohr). Density-chain Hirshfeld \(\partial V/\partial R\) is
**out of scope** for this slug (see `gpw-mbd-hirshfeld` for volume ratios;
Hirshfeld-force chain not productized).

**Total GPW ionic force** when `mbd=true` and forces are assembled:

\[
\mathbf{F}_A
  = \mathbf{F}_A^{\mathrm{HF+Pulay+grid/core}}
  + \mathbf{F}_A^{\mathrm{MBD}}.
\]

**PBE defaults** (libMBD / ATV):

| Parameter | Value | Notes |
| --------- | ----: | ----- |
| \(\beta\) | 0.83 | Range separation |
| Fermi \(a\) | 6 | Steepness |
| Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\) | `basis_data/ts_tables.h` | libMBD `ts_vdw_params` |
| \(V/V_{\mathrm{free}}\) | 1 (all atoms) | Free-atom; optional `vol_ratio` arg |

**Twin settings (binding):**

| Knob | Value |
| ---- | ----- |
| System | Water dimer, 6 atoms (order **O H H O H H**) |
| Geometry | Same as `gpw-mbd-ice` / `gpw-ts-ice` (Å) |
| MBD BC | **Isolated free-atom** (`mbd_k_mesh = {0,0,0}`, `periodic=false` on energy kernel) |
| SCF cell (host only) | Cubic **12 Å**, PERIODIC XYZ, short Γ-RKS GPW |
| XC host | PBE |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Cutoff (SCF) | 200 Ry (E_disp and F_MBD are basis/grid-independent) |
| Observable | \(E_{\mathrm{disp}}\) [Ha]; \(\mathbf{F}^{\mathrm{MBD}}\) [Ha/Bohr] dual-gate |

**Geometry [Å]:**

| Atom | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: |
| O0 | 0.000000 | 0.000000 | 0.000000 |
| H0 | 0.758602 | 0.000000 | 0.504284 |
| H1 | 0.758602 | 0.000000 | −0.504284 |
| O1 | 3.000000 | 0.000000 | 0.000000 |
| H2 | 3.758602 | 0.000000 | 0.504284 |
| H3 | 3.758602 | 0.000000 | −0.504284 |

Å→Bohr: \(1/0.52917720859\) (ATV).

**Key exclusions (this page / acceptance):**

- Periodic Ewald + \(k\)-mesh MBD forces as a separate hard pin
  (`gpw-mbd-periodic` is energy; mesh forces follow the same kernel switch).
- Hirshfeld volume-ratio density chain / \(\partial\rho\) forces
  (`gpw-mbd-hirshfeld` gates volumes + energy, not force chain).
- Analytic closed-form MBD gradients from libMBD C API as the ctest gate
  (optional offline cross-check only).
- CP2K Quickstep absolute \(E_{\mathrm{tot}}\) / ENERGY_FORCE parity (no stock
  MBD pair type).
- D3/D4/TS/XDM force paths (mutually exclusive; see `gpw-d4-forces`, etc.).
- CUDA GPW, hybrid/GAPW, UKS-specific MBD wiring.
- Live libMBD / CP2K from ctest — offline pins only.

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no public `*Config`,
no `make_*` / `build_*` factories). Regression uses the geometry-only MBD
kernel + a short `GpwDft` SCF with `mbd=true` for `e_disp` materialization:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::mbd_rsscs_pbe;
using ATV::cpu::gpw::MbdResult;
using ATV::cpu::gpw::XcKind;

constexpr double kA2B = 1.0 / 0.52917720859;
// pos[6][3] in Bohr from the Å table above …

std::vector<int> Z = {8, 1, 1, 8, 1, 1};
std::vector<std::array<double, 3>> pos(/* 6 rows Bohr */);
const double cell0[3] = {0.0, 0.0, 0.0};

// Geometry-only: energy + forces (isolated free-atom MBD@rsSCS)
const MbdResult mbd = mbd_rsscs_pbe(Z, pos, cell0, /*periodic=*/false);
// mbd.energy  [Ha]     ↔ libMBD pin
// mbd.forces  [Ha/Bohr] ↔ dual-gated vs independent FD of energy

// Product path: short GPW SCF folds E_disp; assemble_gpw_forces adds F_MBD
GpwDft gpw;
const double L = 12.0 * kA2B;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0;
gpw.n_occ = 8;
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.mbd = true;              // exclusive vs d3/d3bj/d4/ts/xdm/vdW-DF*
// kinds: DZVP-MOLOPT-SR-GTH + GTH-PBE-q* → GpwAtom rows …
gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp  [Ha] within 1e-8 of libMBD pin; included in res.e_total
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false,
                                 /*compute_forces=*/true);
eval.mbd = true;
// single_point / Evaluate / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// Frame kAtom_f* [eV/Å] include MBD contribution after assembly
```

Build and run the regression (requires GPW + HF + PW + regressions):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-mbd-forces
  ./build/regressions/reg_gpw-mbd-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/internal/mbd_dispersion.h` | `mbd_rsscs_pbe`, `MbdResult` |
| `src/cpu/gpw/internal/mbd_dispersion.cpp` | Energy kernel + product FD forces |
| `src/cpu/gpw/internal/gpw_forces.cpp` | Adds \(\mathbf{F}^{\mathrm{MBD}}\) when `cfg.mbd` |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::mbd`, `GpwResult::e_disp` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::mbd`, `compute_forces` |
| `include/atomiverse/cpu/gpw/basis_data/ts_tables.h` | Free-atom TS/MBD parameters |
| `regressions/gpw/gpw-mbd-forces/reg_gpw-mbd-forces.cpp` | Product gate |
| `regressions/gpw/gpw-mbd-forces/REFERENCE.md` | Offline pins + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-mbd-forces/` | Oracle notes (libMBD policy) |
| `tests/cpu/gpw/test_cpu_gpw_mbd.cpp` | Unit energy / force checks |

## Validation (libMBD / dual-gate)

Source:
[`regressions/gpw/gpw-mbd-forces/REFERENCE.md`](../../../regressions/gpw/gpw-mbd-forces/REFERENCE.md).
Oracle: offline **libMBD** Fortran `cmbd_mbd_scs_energy` (git `88069c6`) for
energy; force dual-gate is product \(\nabla E\) vs independent central FD of
that energy. External-engine / analytic pins — **not** a golden of Atomiverse’s
own output; **never** live libMBD or CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | Table above (order O H H O H H) |
| MBD | Free-atom MBD@rsSCS, isolated, \(\beta=0.83\), \(a=6\) |
| SCF host | Γ-RKS GPW PBE, 12 Å cubic, short SCF (`max_iter=8`) |
| Pin energy | libMBD git `88069c6`, ATV Å→Bohr |
| Force product | `mbd_rsscs_pbe` forces (\(h=10^{-4}\) Bohr internal) |
| Dual-gate FD | Independent central FD, \(h=10^{-5}\) Bohr, of \(E_{\mathrm{disp}}\) |

**Hard gates (hard-coded in `reg_gpw-mbd-forces`):**

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | **−1.4928804171407961×10⁻³** | \(1\times 10^{-8}\) Ha | libMBD free-atom isolated |
| \(E_{\mathrm{disp}}\) via `GpwDft` [Ha] | same pin | \(1\times 10^{-8}\) Ha | `res.e_disp` after short SCF |
| max\|F_product − F_fd\| [Ha/Bohr] | dual-gate | \(1\times 10^{-6}\) | \(h=10^{-5}\) independent FD |
| \(\|\sum_A \mathbf{F}_A\|\) [Ha/Bohr] | translation | \(1\times 10^{-8}\) | Isolated free molecule |

**Tolerance rationale.** Matching free-atom parameters and the same MBD@rsSCS
Hamiltonian as libMBD track energy to nano-Hartree on this dimer. Force
dual-gate at \(10^{-6}\) Ha/Bohr is dominated by FD truncation
(\(O(h^2)\) of product \(10^{-4}\) vs independent \(10^{-5}\)), not chemical
accuracy. Translation residual at \(10^{-8}\) Ha/Bohr confirms absence of
spurious net force on the isolated system.

Offline notes (human only; never ctest):

```text
# tools/cp2k_oracle/inputs/gpw-mbd-forces/water_dimer_mbd_forces.note
E_disp = -1.4928804171407961e-03 Ha (libMBD git 88069c6)
max |F_product − F_fd(h=1e-5)| ≤ 1e-6 Ha/Bohr
|sum F| ≤ 1e-8 Ha/Bohr
# Optional host with libMBD C API: request do_gradients on same coords
```

## Performance

Source: `REFERENCE.md` / handoff (the case `REFERENCE.md`):
**perf_exempt**.

MBD energy is \(O(N^3)\) on a \(3N\times 3N\) matrix; product force FD is
\(6N\) energy evals (still microseconds on 6 atoms). Full GPW SCF wall
dominates if present. A fair full-SCF ratio vs CP2K is unavailable (no stock
MBD pair type on CP2K 2026.1) and would be unfair for a geometry-only hard gate
(same policy as `gpw-mbd-ice` / `gpw-ts-ice` / `gpw-d4-forces`).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS ENERGY_FORCE | n/a (no stock MBD in CP2K 2026.1) | — | **perf_exempt** | — | Twin notes only |
| E_disp + force dual-gate | n/a | short reg (MBD + optional SCF) | **perf_exempt** | — | Geometry-only deliverable |

**perf_exempt = true.** Reason: primary deliverable is E_disp pin parity plus
force dual-gate; full twin SCF wall vs CP2K is unavailable/unfair.

## Limitations

- **Isolated free-atom MBD only** on this pin — periodic Ewald / \(k\)-mesh
  forces use the same API switch (`mbd_k_mesh` non-zero) but are not the
  hard force dual-gate of this regression.
- **Free-atom \(V/V_{\mathrm{free}}=1\)** — Hirshfeld-scaled energies are
  `gpw-mbd-hirshfeld`; density-chain force contributions are not gated here.
- **Product forces are high-order central FD of the energy kernel**, not a
  closed-form analytic libMBD gradient (optional offline C-API cross-check).
- **E_disp / F_MBD hard gates** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS
  is not asserted.
- **`d3` / `d3bj` / `d4` / `ts` / `xdm` / `mbd` mutually exclusive** — any two
  true throws in energy and force assembly.
- **Frame forces are eV/Å** on the public surface; pin comparison uses Ha/Bohr
  (`units::kEvPerAngstromInHartreePerBohr`).
- Wall **perf_exempt** for this wave; not a speed optimization target.
- **CUDA GPW** out of scope.

## References

- Offline pin: `regressions/gpw/gpw-mbd-forces/REFERENCE.md`
- Twin notes: `tools/cp2k_oracle/inputs/gpw-mbd-forces/water_dimer_mbd_forces.note`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related energy pin: `regressions/gpw/gpw-mbd-ice/`
- Related periodic MBD: `regressions/gpw/gpw-mbd-periodic/`
- Related Hirshfeld volumes: `regressions/gpw/gpw-mbd-hirshfeld/`
- Related D4 forces: `regressions/gpw/gpw-d4-forces/`
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- A. Ambrosetti, A. M. Reilly, R. A. DiStasio Jr., A. Tkatchenko,
  *J. Chem. Phys.* **140**, 18A508 (2014) (MBD@rsSCS)
- J. Hermann, M. Stöhr, S. Góger, S. Chaudhuri, B. Aradi, R. J. Maurer,
  A. Tkatchenko, *J. Chem. Phys.* **159**, 174802 (2023) (libMBD)
- libMBD (Fortran / Python reference for offline pins)
