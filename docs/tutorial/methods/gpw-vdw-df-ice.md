# GPW PBE + vdW-DF (DRSLL) — periodic water dimer (nonlocal correlation)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-vdw-df-ice` |
| Regression | `regressions/gpw/gpw-vdw-df-ice/` |
| Public API | `cpu::gpw::GpwDft::vdw_df` / `cpu::gpw::GPWEvaluator::vdw_df`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**vdW-DF** nonlocal correlation (Dion–Rydberg–Schröder–Langreth–Lundqvist /
**DRSLL**, libvdwxc **DF1**) is wired into the **Γ-point GPW** path as a
**post-SCF** additive on the final total density, on top of PBE. This product
pin uses the same **periodic water-dimer cell** as `gpw-pbe-d3bj-ice` /
`gpw-ts-ice` / `gpw-mbd-ice` (12 Å cubic, ice-relevant H-bonded pair — not bulk
ice Ih) and gates **\(E_c^{\mathrm{nl}}\)** (reported as `e_disp`) against a
dual-frozen Atomiverse pin after a **libvdwxc 0.5.0** kernel unit gate.

**Why libvdwxc rather than live CP2K QS?** CP2K `VDW_POTENTIAL NON_LOCAL TYPE
DRSLL` uses the same Román–Pérez–Soler kernel family as **libvdwxc DF1**. A fair
full-QS twin wall was not available on the pin host this wave; the hard kernel
oracle is libvdwxc (the engine CP2K links when `libvdwxc` is enabled). Twin
input notes are retained under `tools/cp2k_oracle/inputs/gpw-vdw-df-ice/`.

Public surface: bool member `vdw_df` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Nonlocal energy is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`. Mutually exclusive
with `d3` / `d3bj` / `d4` / `ts` / `mbd` (any two true throws at run). Product
gate: `regressions/gpw/gpw-vdw-df-ice` (`reg_gpw-vdw-df-ice`). Wall ratio is
**perf_exempt** (E_disp pin; full QS twin wall unfair / unavailable — see
Performance).

## Method

**Nonlocal vdW-DF correlation** (Dion et al., *Phys. Rev. Lett.* **92**, 246401
(2004); Román-Pérez & Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)):

\[
E_c^{\mathrm{nl}}
  = \tfrac12 \iint \rho(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\rho(\mathbf r')
    \,\mathrm d\mathbf r\,\mathrm d\mathbf r',
\]

with \(q_0(\rho,|\nabla\rho|^2)\) and the community **RPS** FFT kernel tables
(DF1 / DRSLL):

| Parameter | Value | Notes |
| --------- | ----: | ----- |
| \(Z_{ab}\) | −0.8491 | DF1 gradient-correction prefactor |
| \(q_{\mathrm{cut}}\) | 5.0 Bohr⁻¹ | Soft q-cutoff (libvdwxc / QE) |
| Kernel tables | RPS DF1 | In-tree `vdw_df_kernel_data` (float32 community tables) |

**Evaluation path (product).** Post-SCF on the **final total density** from
Γ-RKS GPW PBE: collocate \(\rho\), spectral \(|\nabla\rho|^2\) from the GPW FFT
gradient, then `vdw_df_drsll(grid, rho, sigma)`. The nonlocal **potential** is
**not** back-propagated into the SCF (non-self-consistent \(E_c^{\mathrm{nl}}\)
on the converged PBE density). Single orthorhombic grid only — multigrid and
nonperiodic (MT) Poisson throw.

**CP2K surface.** Twin notes map to:

```text
&VDW_POTENTIAL
  POTENTIAL_TYPE NON_LOCAL
  &NON_LOCAL
    TYPE DRSLL
  &END NON_LOCAL
&END VDW_POTENTIAL
```

additive on the chosen XC (`&XC_FUNCTIONAL PBE`).

**Sign / interpretation.** Pure \(E_c^{\mathrm{nl}}\) on a molecular density can
be **positive**; the dispersion *binding* is recovered in interaction energy
differences. Pairwise D3 / TS / MBD `e_disp` values use a different functional
form and are **not** comparable in magnitude to this pin.

**Electronic host.** The regression runs Γ-RKS GPW (PBE,
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`, CUTOFF 200 Ry) so `e_disp` is materialized
on a real `GpwResult` and included in `e_total`. Absolute QS
\(E_{\mathrm{tot}}\) micro-Ha parity vs CP2K is **out of scope** (OT/grid/DIIS;
no full QS twin wall this wave).

**Key exclusions (from product / REFERENCE).**

- Self-consistent vdW-DF potential / forces / stress
- vdW-DF2, optB88-vdW, VV10 / rVV10, XDM
- Multigrid density, nonperiodic MT, k-points, UKS as part of this slug
- Full ice Ih supercell / phonons
- Pairwise D3/D4/TS/MBD simultaneous with `vdw_df`
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS
- CUDA GPW

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no `*Config`, no
`make_*` / `build_*` factories). Regression uses the concrete solver:

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

// Geometry [Å] → Bohr — water dimer (REFERENCE.md / same as D3(BJ) pin).
constexpr double kA2B = 1.0 / 0.52917720859;
// … fill 6 GpwAtom rows (2×O + 4×H) with DZVP-MOLOPT-SR-GTH + GTH-PBE-q* …

GpwDft gpw;
const double L = 12.0 * kA2B;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0;
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 40;
gpw.tol = 1e-7;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.vdw_df = true;     // exclusive vs d3 / d3bj / d4 / ts / mbd

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 5e-5 of offline pin
// res.e_total includes e_disp
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.vdw_df = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
```

Build and run the regression:

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
  cmake --build build --target reg_gpw-vdw-df-ice -j$(nproc)
  ./build/regressions/reg_gpw-vdw-df-ice
'
```

Scoped unit kernel gate (optional, merge gate suite):

```bash
cmake --build build -j$(nproc) --target atomiverse test_cpu_gpw_vdw_df
ctest --test-dir build -R '^test_cpu_gpw_vdw_df$' --output-on-failure
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `vdw_df` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::vdw_df` |
| `include/atomiverse/cpu/gpw/internal/vdw_df.h` | `vdw_df_drsll`, `vdw_df_q0`, DF1 constants |
| `include/atomiverse/cpu/gpw/internal/vdw_df_kernel_data.h` | RPS DF1 kernel tables |
| `src/cpu/gpw/internal/vdw_df.cpp` | Román–Pérez–Soler FFT evaluation |
| `tests/cpu/gpw/test_cpu_gpw_vdw_df.cpp` | Unit gate vs offline libvdwxc |
| `regressions/gpw/gpw-vdw-df-ice/reg_gpw-vdw-df-ice.cpp` | Product gate |
| `regressions/gpw/gpw-vdw-df-ice/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-vdw-df-ice/` | libvdwxc pin notes + CP2K twin sketch |

## Validation (CP2K / libvdwxc)

Source:
[`regressions/gpw/gpw-vdw-df-ice/REFERENCE.md`](../../../regressions/gpw/gpw-vdw-df-ice/REFERENCE.md).
Oracle: offline **libvdwxc 0.5.0** `VDWXC_DF1` for the RPS kernel (same family
as CP2K `TYPE DRSLL`); full-cell product pin is dual-frozen after that unit
gate. External engine pin — not a golden of Atomiverse’s own output; **never**
launch live libvdwxc or CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + nonlocal | PBE host + vdW-DF DRSLL post-SCF |
| Grid | CUTOFF **200 Ry**, single grid |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Kernel pin | libvdwxc DF1 on 16³ L=8 Bohr Gaussian blob: \(E_{\mathrm{nl}}=1.2367739353586624\times 10^{-1}\) Ha |
| Product pin | \(E_{\mathrm{disp}}=+1.5194390480303990\times 10^{-1}\) Ha |

| Quantity | Atomiverse / pin | Oracle (offline) | Tol | Notes |
| -------- | ---------------: | ----------------: | --- | ----- |
| \(E_{\mathrm{nl}}\) synthetic blob [Ha] | unit `vdw_df_drsll` | libvdwxc 0.5.0 **+0.12367739353586624** | \(1\times 10^{-6}\) Ha | Kernel gate |
| \(E_{\mathrm{disp}}\) GPW PBE+DRSLL [Ha] | **+0.15194390480303990** | dual-frozen product pin (post unit gate) | \(5\times 10^{-5}\) Ha | Hard product gate |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (CP2K QS wall N/A on pin host) | — | Not absolute-parity gated |

**Tolerance rationale.** Float32 RPS kernel tables ⇒ ~1e−7 absolute kernel
noise; unit pin at **1e−6 Ha**. Post-SCF density + grid discretization on the
dimer needs a looser product tol (**5×10⁻⁵ Ha**). Matching \(q_0\), \(Z_{ab}\),
and RPS tables track libvdwxc DF1 / CP2K DRSLL kernel family on the same
\((\rho,\sigma)\).

Offline kernel re-pin (human only; never ctest) — see
`tools/cp2k_oracle/inputs/gpw-vdw-df-ice/water_dimer_vdw_df.note`:

```text
# 16^3 cubic L=8 Bohr, rho = exp(-0.5 r^2) blob at center, analytic sigma
E_nl DRSLL/DF1 = 1.2367739353586624e-01 Ha
```

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

RPS evaluation is \(O(N_g \log N_g \times n_\alpha)\) with \(n_\alpha = 20\)
spline channels; wall on the 6-atom cell is dominated by the electronic SCF,
not the nonlocal kernel. A fair **kernel-only** microbenchmark vs libvdwxc is
not the product gate; a full twin SCF ratio vs CP2K is unavailable on the pin
host this wave.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (no CP2K QS twin wall this wave) | — (not fair-twinned) | **perf_exempt** | — | Twin notes retained under `tools/cp2k_oracle/inputs/gpw-vdw-df-ice/` |
| E_disp product gate | n/a | short SCF + RPS (reg) | **perf_exempt** | — | Post-SCF density deliverable |

**perf_exempt = true.** Reason (from REFERENCE / handoff): primary deliverable
is E_disp pin parity (same policy as `gpw-ts-ice` / `gpw-mbd-ice` /
`gpw-pbe-d3bj-ice`). Re-measure a strict full-SCF twin later if product policy
requires \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on a host with
libvdwxc-linked CP2K.

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic RPS /
  post-SCF density wiring; not a structural ice benchmark.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS is
  not asserted (and full QS twin wall was unavailable on the pin host).
- **Post-SCF only** — nonlocal potential not fed back into the KS Hamiltonian;
  no analytic vdW-DF forces / stress product gate.
- **DRSLL / DF1 only** — no vdW-DF2, optB88-vdW, VV10.
- **Single periodic grid** — multigrid and nonperiodic (MT) throw with
  `vdw_df`.
- **`d3` / `d3bj` / `d4` / `ts` / `mbd` / `vdw_df` mutually exclusive** — any
  two true throws at run.
- **Positive \(E_c^{\mathrm{nl}}\)** on this density is physical for the pure
  nonlocal term; do not compare magnitude to pairwise D3/TS pins.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-vdw-df-ice/REFERENCE.md`
- Oracle notes: `tools/cp2k_oracle/inputs/gpw-vdw-df-ice/`,
  `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related D3(BJ) GPW pin: [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md)
- Related D4 GPW pin: [gpw-d4-water](gpw-d4-water.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) (vdW-DF / DRSLL)
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)
  (efficient FFT evaluation of the nonlocal kernel)
- libvdwxc 0.5.0 (CLI / library pin backend; same DF1 family CP2K links as
  `TYPE DRSLL`)
