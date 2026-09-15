# GPW PBE + vdW-DF2 (LMKLL) — periodic water dimer (ice-relevant H-bond)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-vdw-df2-ice` |
| Regression | `regressions/gpw/gpw-vdw-df2-ice/` |
| Public API | `cpu::gpw::GpwDft::vdw_df2` / `cpu::gpw::GPWEvaluator::vdw_df2`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Nonlocal vdW-DF2 correlation** (Lee–Murray–Kong–Lundqvist–Langreth /
**LMKLL** / libvdwxc **DF2**) is wired into the **Γ-point GPW** path as a
**post-SCF** additive nonlocal correlation energy on the final total density,
hosted by **PBE** semi-local XC. This product pin uses a **periodic
water-dimer cell** (ice-relevant H-bonded pair in a 12 Å cubic box — not a full
ice Ih supercell) and gates **\(E_c^{\mathrm{nl}}\)** reported on
`GpwResult::e_disp` [Hartree].

Public surface: bool member `vdw_df2` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). The nonlocal energy is folded into
`e_total`. Mutually exclusive with `d3` / `d3bj` / `d4` / `ts` / `mbd` /
`vdw_df` / `rvv10`. Product gate: `regressions/gpw/gpw-vdw-df2-ice`
(`reg_gpw-vdw-df2-ice`). Wall ratio is **perf_exempt** (E_disp pin; full QS twin
wall unavailable / unfair on the pin host — see Performance).

**Oracle lineage.** CP2K `VDW_POTENTIAL` / `POTENTIAL_TYPE NON_LOCAL` /
`&NON_LOCAL TYPE LMKLL` links the same Román–Pérez–Soler (RPS) engine family as
**libvdwxc DF2**. The hard unit kernel pin is offline **libvdwxc 0.5.0**; the
full-GPW product pin freezes post-SCF \(E_c^{\mathrm{nl}}\) on the ATV PBE
density after that unit gate. Twin notes:
`tools/cp2k_oracle/inputs/gpw-vdw-df2-ice/`. **Never** launch live CP2K (or
libvdwxc) from ctest.

## Method

Nonlocal correlation (Dion et al. skeleton; DF2 \(Z_{ab}\)):

\[
E_c^{\mathrm{nl}}
  = \tfrac12 \iint \rho(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\rho(\mathbf r')
    \,\mathrm d\mathbf r\,\mathrm d\mathbf r',
\]

with \(q_0(\rho,|\nabla\rho|^2)\) and the community RPS kernel tables
(libvdwxc 0.5.0 / Quantum ESPRESSO — **shared with DF1 / DRSLL**). Functional
variants differ only in the gradient-correction prefactor:

| Functional | CP2K `NON_LOCAL TYPE` | libvdwxc | \(Z_{ab}\) |
| ---------- | --------------------- | -------- | ---------: |
| vdW-DF (DF1) | DRSLL | `VDWXC_DF1` | −0.8491 |
| **vdW-DF2 (DF2)** | **LMKLL** | **`VDWXC_DF2`** | **−1.887** |

Also \(q_{\mathrm{cut}} = 5\) Bohr⁻¹ (soft cutoff, matching libvdwxc / QE).

**Evaluation policy (product path).**

- **Post-SCF energy only** on the final total density (non-self-consistent
  nonlocal potential — no SCF feedback of \(v_c^{\mathrm{nl}}\)).
- Spectral \(\lvert\nabla\rho\rvert^2\) from the GPW FFT gradient on the
  orthorhombic uniform grid.
- Host XC is **PBE** (`XcKind::PBE`); LMKLL is additive nonlocal correlation,
  not a replacement for PBE \(E_c^{\mathrm{loc}}\).
- Pure \(E_c^{\mathrm{nl}}\) on a molecular density **can be positive**;
  dispersion *binding* is recovered in interaction energy differences.
  Pairwise D3/TS and DRSLL \(E_c^{\mathrm{nl}}\) are different functional forms
  and are **not** comparable one-to-one in magnitude.

**Key exclusions.**

- Full ice Ih supercell / phonons / cell stress
- Self-consistent nonlocal potential (post-SCF only)
- Stock VV10 (distinct from rVV10), XDM, optB88-vdW
- k-points, UKS, CUDA GPW
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K Quickstep
- Live libvdwxc / CP2K in ctest

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

// Geometry [Å] → Bohr — same water dimer as gpw-vdw-df-ice / gpw-pbe-d3bj-ice.
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
gpw.vdw_df2 = true;    // exclusive vs d3/d3bj/d4/ts/mbd/vdw_df/rvv10

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
eval.vdw_df2 = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// Public frame.meta energy remains eV; internal GpwResult energies are Ha.
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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-vdw-df2-ice
  ./build/regressions/reg_gpw-vdw-df2-ice
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `vdw_df2` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::vdw_df2` |
| `include/atomiverse/cpu/gpw/internal/vdw_df.h` | `vdw_df_lmkll`, `kVdwDf2Zab` (private) |
| `tests/cpu/gpw/test_cpu_gpw_vdw_df.cpp` | Unit gate vs libvdwxc DF2 synthetic |
| `regressions/gpw/gpw-vdw-df2-ice/reg_gpw-vdw-df2-ice.cpp` | Product gate |
| `regressions/gpw/gpw-vdw-df2-ice/REFERENCE.md` | Offline pins + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-vdw-df2-ice/` | CP2K / libvdwxc twin notes |

## Validation (CP2K / libvdwxc)

Source:
[`regressions/gpw/gpw-vdw-df2-ice/REFERENCE.md`](../../../regressions/gpw/gpw-vdw-df2-ice/REFERENCE.md).
Oracle: offline **libvdwxc 0.5.0** (kernel; same engine CP2K links for
`TYPE LMKLL`) + dual-frozen full-GPW pin after unit gate. Twin notes:
`tools/cp2k_oracle/inputs/gpw-vdw-df2-ice/water_dimer_vdw_df2.note`. External
engine pins — not goldens of Atomiverse’s own output; **never** launch live
oracle from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| Host XC | PBE (`XcKind::PBE`) |
| Nonlocal | LMKLL / DF2 (\(Z_{ab}=-1.887\)) post-SCF on final \(\rho\) |
| Grid / cutoff | GPW CUTOFF **200 Ry** (regression) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| CP2K twin surface | `&VDW_POTENTIAL` / `NON_LOCAL TYPE LMKLL` (when QS+libvdwxc available) |

| Quantity | Atomiverse / pin | Oracle (offline) | Tol | Notes |
| -------- | ---------------: | ----------------: | --- | ----- |
| \(E_c^{\mathrm{nl}}\) product [Ha] | \(+1.3011875519702767\times 10^{-1}\) | same dual-frozen pin | \(5\times 10^{-5}\) Ha | Hard gate `reg_gpw-vdw-df2-ice` |
| Kernel \(E_{\mathrm{nl}}\) synthetic 16³ [Ha] | matches unit test | libvdwxc `VDWXC_DF2` \(+1.1428161716669384\times 10^{-1}\) | \(1\times 10^{-6}\) Ha | Fixed Gaussian blob, L=8 Bohr |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | — | — | Context only; OT/grid/DIIS |

**Tolerance rationale.** Unit tests lock the RPS/DF2 kernel to libvdwxc at
μHa on a fixed density. The product gate uses a looser \(5\times 10^{-5}\) Ha
tolerance on post-SCF \(E_c^{\mathrm{nl}}\) because the host density comes from
a finite SCF (mix/tol/cutoff) and the RPS path uses float32 kernel tables.
Positive \(E_c^{\mathrm{nl}}\) is expected for this isolated-dimer density;
compare DRSLL (`gpw-vdw-df-ice`) only via interaction differences, not raw
\(E_{\mathrm{disp}}\).

## Performance

Source: `REFERENCE.md` (queue / handoff wave 4: **perf_exempt**).

RPS evaluation is \(O(N_g \log N_g \times n_\alpha)\) with \(n_\alpha=20\),
dominated by the electronic SCF on the 6-atom cell. A fair full Quickstep twin
wall vs CP2K is **not available** on the pin host (CP2K+libvdwxc QS not
measured this wave). Same policy as `gpw-vdw-df-ice` / `gpw-ts-ice` /
`gpw-mbd-ice` / `gpw-rvv10-ice`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (not fair-twinned) | — | **perf_exempt** | — | No CP2K QS wall on pin host |
| E_disp product gate | n/a | short SCF + RPS (reg) | **perf_exempt** | — | Hard gate is E_disp only |
| Kernel micro (synthetic) | libvdwxc pin host | unit test | **perf_exempt** | — | Accuracy gate, not wall ratio |

**perf_exempt = true.** Reason (REFERENCE / handoff): primary deliverable is
E_disp pin parity against libvdwxc-aligned kernel + dual-frozen full-GPW value;
full QS wall is out of scope for this wave. Re-measure a strict full-SCF twin
when a host provides CP2K linked with libvdwxc if product policy requires
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic
  nonlocal LMKLL parity; not a structural ice benchmark.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS is
  not asserted.
- **Post-SCF only** — nonlocal potential is not fed back into the SCF.
- **PBE host only** on this pin; other semi-local hosts share the same
  `vdw_df2` flag but are not e2e-gated here.
- **Exclusive** with `d3` / `d3bj` / `d4` / `ts` / `mbd` / `vdw_df` / `rvv10`
  — any two true throws at run.
- **Periodic single-grid** product path; nonperiodic MT + nonlocal not the
  regression surface.
- Wall **perf_exempt** for this wave; not a speed optimization target.
- Related productized siblings: DRSLL `gpw-vdw-df-ice`, rVV10 `gpw-rvv10-ice`,
  pairwise/many-body `gpw-pbe-d3bj-ice` / `gpw-ts-ice` / `gpw-mbd-*`.

## References

- Offline pin: `regressions/gpw/gpw-vdw-df2-ice/REFERENCE.md`
- Oracle inputs: `tools/cp2k_oracle/inputs/gpw-vdw-df2-ice/`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion (vdW-DF2 ✓)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Sibling: DRSLL `docs/tutorial/methods/` / `regressions/gpw/gpw-vdw-df-ice`
- Closed spec (git history): `.claude/specs/gpw-vdw-df2-ice/` (deleted at done)
- K. Lee, É. D. Murray, L. Kong, B. I. Lundqvist, D. C. Langreth,
  *Phys. Rev. B* **82**, 081101(R) (2010) (vdW-DF2 / LMKLL)
- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) (vdW-DF / DRSLL skeleton)
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)
  (RPS FFT evaluation)
