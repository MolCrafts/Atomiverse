# Finite-\(q\) inter-\(k\) RI-GPW dRPA — diamond Si (bct 4-atom, MP 2×1×1)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft::direct_rpa()` → `cpu::gpw::RpaResult` (`cpu/gpw/rpa.h`); multi-k members `k_mesh[]`, `k_time_reversal`, `mp2_ri`, `mp2_n_virt` |
| Roadmap | `docs/roadmap/dft.md` §10 Periodic RPA (finite-\(q\) inter-\(k\)) |

## Summary

Closed-shell **finite-\(q\) inter-\(k\) direct RPA (dRPA)** correlation on a
converged multi-\(k\) **GPW PBE** reference. For every crystal-momentum
transfer \(q\) on the same closed Monkhorst–Pack mesh as \(k\), Atomiverse
builds the Coulomb-only Casida spectrum from momentum-conserving
transitions \(i_k\to a_{k+q}\) with **RI-GPW** Coulomb at transfer \(q\),
reduces each block with the **Furche plasmon formula**, and Brillouin-zone
averages. This is the **product multi-\(k\) path** for periodic RPA: it
supersedes the multi-\(k\) **k-diagonal** foothold of [periodic-rpa-kpoints](periodic-rpa-kpoints.md)
and sits above the Γ-only product [periodic-rpa-si](periodic-rpa-si.md). Total energy is
**dRPA@KS** \(E_{\mathrm{SCF}}+E_c\) (no EXX — periodic HF blocked).
Public path is `GpwDft` members → `build` → `run` → `direct_rpa` — no
`*Config` bag, no factory. No standing regression.

## Method

**Level.** Direct random-phase approximation as a **standalone correlation
energy** on KS orbitals (roadmap Periodic RPA). Not TDHF excitation RPA, not
TDDFT Casida, not RPAX/SOSEX, not CP2K’s frequency-integrated RI-RPA + EXX
total-energy assembly.

**Spin.** Closed-shell spatial / complex Bloch MOs only (RKS GPW). UKS /
URPA is out of scope.

**Reference.** Multi-\(k\) GPW RKS (PBE) on diamond Si in the **bct 4-atom**
cell used by `gpw-kpoints-si` / [periodic-rpa-kpoints](periodic-rpa-kpoints.md): a = 5.43 Å,
\(a_t=a/\sqrt{2}\), \(c=a\); DZVP-MOLOPT-GTH + GTH-PBE-q4; cutoff 80 Ry;
MP \(2\times1\times1\) with `k_time_reversal=false` → \(N_k=2\), equal
weights \(w_k=1/2\). Virtual space capped with `mp2_n_virt=2` for the
e2e ≤ 60 s budget (full virtual space is the production path).

**Kernel (per \(q\)).** Coulomb-only Casida matrices for transitions
\(i_{\mathbf k}\to a_{\mathbf{k}+\mathbf q}\). RI-GPW Poisson at transfer
\(q\): complex Bloch pair densities, linear real Poisson on Re/Im;
\(G=0\) dropped only at \(q=0\). Same RI product as multi-\(k\) MP2
(`mp2_ri=true`).

**Correlation (Furche plasmon, multi-\(q\)).**

\[
E_c=\sum_q w_q\,\tfrac12\Bigl(\sum_s\Omega_{q,s}-\mathrm{Tr}\,A_q\Bigr).
\]

At Γ (\(N_k=1\)) this collapses to the real plasmon formula used by
[periodic-rpa-si](periodic-rpa-si.md) and molecular `DirectRPA<RHF>::run`. Unit tests pin
\(N_k=1\) multi-\(k\) complex MOs ≡ real `direct_rpa_rhf_energy`.

**Spectrum size.** With \(q\)-mesh \(=\) \(k\)-mesh,

\[
n_\Omega = N_q\,N_k\,n_{\mathrm{occ}}\,n_{\mathrm{virt}}
         = N_k^2\,n_{\mathrm{ov}}.
\]

For the regression deck: \(N_k=2\), \(n_{\mathrm{occ}}=8\),
\(n_{\mathrm{virt}}=2\) → \(n_\Omega=64\). All \(\Omega_s>0\).

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E_c\) | BZ-averaged plasmon correlation | Ha (internal); public Frame energy remains eV when written |
| \(E_{\mathrm{SCF}}\) | converged multi-\(k\) GPW PBE total | Ha |
| \(E_{\mathrm{tot}}\) | \(E_{\mathrm{SCF}}+E_c\) (dRPA@KS) | Ha |
| \(\mathrm{Tr}\,A\) | \(\sum_q w_q\,\mathrm{Tr}\,A_q\) | Ha |
| \(\Omega_s\) | all finite-\(q\) roots, concatenated then sorted | Ha |
| `ri`, `n_kpts` | path flags on `RpaResult` | — |

**Key exclusions**

- Periodic exact exchange / hybrid RPA total energies (blocked with periodic HF)
- Frequency-integrated RI-RPA + Gaussian `RI_AUX` as in stock CP2K
  `&RI_RPA` (near-twin diagnostic only — not a hard \(E_c\) pin)
- Open-shell / UKS dRPA; CUDA GPW RPA
- Analytic RPA forces / stress
- Production cutoffs and uncapped virtuals (e2e deliberately tiny)

## Implementation

Public surface only (`GpwDft` members; no `*Config`, no `make_*` / `build_*`).
Headers: `gpw_solver.h`, `rpa.h`; kernels under
`src/cpu/gpw/rpa.cpp` + `src/cpu/gpw/internal/mp2_kernels.cpp` (shared
RI-GPW \((ia|jb)\) / finite-\(q\) path).

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/rpa.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using namespace ATV::cpu::gpw;

// bct diamond Si4: a = 5.43 Å, a_t = a/√2, c = a; fractional coords
// (0,0,0), (½,½,½), (½,0,¼), (0,½,¾); Z = 14 in kAtom_type.
constexpr double a = 5.43;                   // [Å]
const double at = a / std::sqrt(2.0);
const double frac[4][3] = {{0, 0, 0}, {0.5, 0.5, 0.5}, {0.5, 0, 0.25}, {0, 0.5, 0.75}};
cpu::Frame frame(4);
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_type)[i] = 14;
    frame.span(kAtom_x)[i] = frac[i][0] * at;
    frame.span(kAtom_y)[i] = frac[i][1] * at;
    frame.span(kAtom_z)[i] = frac[i][2] * a;
}
frame.box = cpu::Box::orthorhombic(at, at, a);

GpwDft m{{Kind{molrs::Element::Si,
               GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
               GthPotential(molrs::Element::Si, "GTH-PBE-q4")}}, 80.0, XcKind::PBE};
m.n_occ = 8;                 // 16 valence e⁻
m.max_iter = 60;
m.tol = 1e-7;
m.mix = 0.3;
m.level_shift = 0.0;
m.eps_default = 1e-7;
m.mp2_ri = true;             // RI-GPW product path (required multi-k)
m.mp2_n_virt = 2;            // e2e virt cap
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = false;   // FULL_GRID / no symmetry twin
m.diis_depth = 6;

m.build(frame);
const GpwResult& scf = m.run(frame);   // multi-k RKS
const RpaResult r = m.direct_rpa();
// r.e_corr, r.e_scf, r.e_total [Ha]; r.omega_ha; r.tr_A; r.n_kpts; r.ri
```

Build and run the regression (scoped — never bare full-tree ctest for this
slug):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse
'
```

Unit recovery (no full SCF): `tests/cpu/gpw/test_cpu_gpw_rpa` — multi-\(k\)
kernel at \(N_k=1\) ≡ real plasmon.

## Validation (vs offline CP2K)

Source: the pin notes recorded when this page was written (no standing regression; offline pin;
**never** live CP2K in ctest). Oracle input twin:
CP2K oracle.

**Oracle status.** No CP2K binary on the pin host at first green
(2026-07-26). Absolute CP2K \(E_c\) digits are **not invented**. CP2K’s
default path is frequency-integrated RI-RPA with Gaussian `RI_AUX` and EXX
total-energy assembly — **not** the exact \(q\)-resolved plasmon formula on
PW-RI \((ia|jb)\). Printed CP2K correlation is a near-twin diagnostic only.

### Accuracy table (hard-coded pins)

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k PBE, 80 Ry) | **−15.5751782413** Ha (7 iters) | CP2K twin setup (no abs pin) | SCF gates | multi-\(k\) DIIS; first green 2026-07-26 |
| \(E_c\) (finite-\(q\), \(n_{\mathrm{virt}}=2\)) | **−0.0736273682** Ha | first-green ATV band (not CP2K digit twin) | **5e-3 Ha** | setup band covers OMP orbital scatter |
| \(E_{\mathrm{tot}}=E_{\mathrm{SCF}}+E_c\) | identity | — | **1e-12** abs | regression hard gate |
| \(E_c\) sign / finite | \(E_c<0\), finite | — | hard | |
| Spectrum size \(n_\Omega\) | **64** (\(=2\cdot2\cdot8\cdot2\)) | product formula | exact | finite-\(q\) product gate |
| All \(\Omega_s\) | \(>0\), finite | — | hard | |
| Flags | `ri=true`, `n_kpts=2`, \(n_{\mathrm{occ}}=8\), \(n_{\mathrm{virt}}=2\) | — | exact | |
| Γ recovery (unit) | multi-\(k\) \(N_k=1\) ≡ real plasmon | internal | unit tol | `test_cpu_gpw_rpa` |

Geometry / cell / basis / PP / XC / k-mesh: see REFERENCE system table and
`periodic-rpa-finite-q.inp` (`MONKHORST-PACK 2 1 1`, `FULL_GRID`,
`SYMMETRY .FALSE.`, `ADDED_MOS 2` ↔ `mp2_n_virt=2`).

## Performance

Source: the pin notes recorded when this page was written. Fair-wall target when a comparable CP2K binary and
algorithm twin exist: \(r=T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\le 10\).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| finite-\(q\) plasmon vs CP2K RI-RPA | n/a (no fair twin / no binary) | ~3 (SCF+RPA, OMP=4) | **perf_exempt** | — | CP2K RI-RPA ≠ \(q\)-resolved plasmon on PW-RI; never live CP2K in ctest |

Regression wall budget: **≤ 60 s** (hard fail above).

## Limitations

- **e2e deck is a capability pin**, not a materials-quality RPA number:
  80 Ry, \(n_{\mathrm{virt}}=2\), MP \(2\times1\times1\).
- Absolute \(E_c\) is a **first-green ATV band** until a fair offline CP2K
  (or literature) plasmon twin is available; do not treat −0.0736273682 Ha
  as a CP2K digit.
- No EXX / hybrid dRPA@HF total energy (periodic HF blocked).
- Closed-shell only; no RPA forces/stress; no CUDA.
- Lineage: Γ product [periodic-rpa-si](periodic-rpa-si.md) → multi-\(k\) k-diagonal residual
  [periodic-rpa-kpoints](periodic-rpa-kpoints.md) → **this finite-\(q\) product**. Shared multi-\(k\)
  path; k-diagonal foothold is superseded for the product surface.

## References

- Oracle input: CP2K oracle
- Oracle rules: CP2K oracle (offline only; never in ctest)
- Roadmap: `docs/roadmap/dft.md` §10
- Related: [periodic-rpa-si](periodic-rpa-si.md), [periodic-rpa-kpoints](periodic-rpa-kpoints.md), [periodic-mp2-kpoints](periodic-mp2-kpoints.md),
  [periodic-gw-kpoints](periodic-gw-kpoints.md)
- Furche, *J. Chem. Phys.* **114**, 5982 (2001)
- Ren et al., *New J. Phys.* **14**, 053020 (2012)
- Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012)
  (GPW \((ia|jb)\) / RI-GPW route)
