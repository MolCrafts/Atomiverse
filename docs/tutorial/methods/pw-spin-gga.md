# PW true collinear spin-GGA PBE — O₂ triplet (Γ)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-spin-gga` |
| Regression | `regressions/pw/pw-spin-gga/` |
| Public API | `cpu::pw::PW` (`spin_polarized`, `multiplicity`, `xc = XCType::PBE`) → `PWResult::{energy_ha, magnetization}` |
| Track | CP2K capability wave 4 () |
| Roadmap | PW true collinear spin-GGA (`docs/roadmap/dft.md` §10; productized `reg_pw-spin-gga`) |

## Summary

**True collinear spin-GGA** on the plane-wave FFT grid: separate
\(\rho_\uparrow/\rho_\downarrow\) channels with spectral density gradients enter
the reduced-gradient invariants \(\sigma_{\sigma\sigma'}\) in
`cpu::scf::evaluate_xc`, and the spin potentials include the full divergence
term (same form as GPW `xc_grid_gga_spin`). This productizes
`XCBuilder::step_spin` beyond the historical PBE@\(\sigma=0\) LSDA-limit
foothold. Target system is gas-phase **O₂ triplet** \(^3\Sigma_g^-\)
(multiplicity 3) with an SG15 ONCV oxygen UPF. Gates: fixed-occupation
magnetization \(M=2\), offline \(E_{\mathrm{tot}}\) [Ha] pin, and an internal
PBE–LSDA energy-gap activation pin. Public surface: concrete `cpu::pw::PW`
members only (no `*Config`, no factories). Regression:
`reg_pw-spin-gga`. **No live CP2K/QE in ctest** — offline pins only. Perf is
**`perf_exempt`**.

## Method

**Level.** Collinear spin-polarized plane-wave Kohn–Sham DFT with **true GGA**
exchange–correlation (PBE) on the real-space FFT grid:

- Separate \(\alpha/\beta\) KS orbitals and densities \(\rho_\uparrow\),
  \(\rho_\downarrow\).
- Hartree from total density \(\rho = \rho_\uparrow + \rho_\downarrow\).
- Spectral \(\nabla\rho_\sigma\) from \(\rho_\sigma(G)\); reduced gradients
  \[
  \sigma_{\sigma\sigma'}
  =
  (\nabla\rho_\sigma)\cdot(\nabla\rho_{\sigma'}).
  \]
- Pointwise \(e_{\mathrm{xc}}(\rho_\uparrow,\rho_\downarrow,\{\sigma\})\) via
  shared `cpu::scf::evaluate_xc` (libxc-parity tree).
- Spin potentials (collinear spin-GGA; `XCBuilder::step_spin`):

\[
v_\sigma
=
v_{\rho_\sigma}
-
\nabla\cdot\Big(
  2\,v_{\sigma_{\sigma\sigma}}\,\nabla\rho_\sigma
  +
  v_{\sigma_{\uparrow\downarrow}}\,\nabla\rho_{\bar\sigma}
\Big).
\]

This is **not** calling PBE with forced \(\sigma=0\) per spin channel (that
LSDA-limit path was the earlier spin foothold and is superseded as the
production spin-GGA path when `xc = XCType::PBE`).

**Spin / occupations.** Non-metallic fixed fillings from multiplicity \(2S+1\):

\[
N_\uparrow - N_\downarrow = 2S = m - 1,\qquad
N_\uparrow + N_\downarrow = N_{\mathrm{val}}.
\]

Oxygen NC-PP \(Z_{\mathrm{val}} = 6\) → \(N_{\mathrm{val}} = 12\) for O₂;
multiplicity 3 → \(N_\uparrow = 7\), \(N_\downarrow = 5\),

\[
M = N_\uparrow - N_\downarrow = 2.
\]

**Basis / PP / sampling.** Plane-wave cutoff \(E_{\mathrm{cut}} = 12\) Ry;
Γ-only Monkhorst–Pack (`k_mesh = 1`); cubic cell \(L = 8.0\) Å; UPF
`tests/data/pw/O_ONCV_PBE-1.2.upf` (Hamann ONCVPSP / SG15 library, PBE).

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E\) | Total KS energy (`PWResult::energy`) | Ry |
| \(E_{\mathrm{Ha}}\) | \(E/2\) (`PWResult::energy_ha`) | Ha |
| \(M\) | \(N_\uparrow - N_\downarrow\) (`PWResult::magnetization`) | e⁻ |
| \(\lvert E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}\rvert\) | GGA activation dual run | Ha |
| — | SCF convergence flag / iteration count | — |

**Excluded (this slug).** Metallic smearing + spin (`pw-uks-fe-smearing` class);
noncollinear / SOC; spin-polarized analytic forces or stress acceptance; hybrid
/ meta-GGA XC; CUDA PW spin; absolute digit match to CP2K Quickstep (GPW/GTH ≠
NC-UPF PW — prefer QE `pw.x` offline re-pin). GPW spin-GGA is a separate
product surface (`gpw-uks-pbe`, `gpw-uks-blyp`).

## Implementation

Public surface only: construct `cpu::pw::PW`, set members (no `*Config` bag,
no `make_*` / `build_*` factory), `build(frame)` → `run(frame)`. Header:
`include/atomiverse/cpu/planewave/pw_solver.h`. Internals
(`XCBuilder::step_spin`, density spin buffers) stay under
`cpu/planewave/internal/`.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 8;       // Z(O)
    frame.span(kAtom_mass)[i] = 15.999;
    frame.span(kAtom_x)[i] = 0.0;
    frame.span(kAtom_y)[i] = 0.0;
}
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_z)[1] = 1.2075;         // R(O–O) [Å]
frame.box = cpu::Box::cubic(8.0);

PW pw;
pw.upf_path = "tests/data/pw/O_ONCV_PBE-1.2.upf";
pw.ecut_ry = 12.0;
pw.k_mesh = 1;                           // Γ
pw.tol = 5e-4;                           // density L2
pw.max_iter = 60;
pw.mix_beta = 0.3;
pw.pulay_depth = 5;
pw.spin_polarized = true;
pw.multiplicity = 3;                     // triplet → M = 2
pw.xc = cpu::scf::XCType::PBE;           // true collinear spin-GGA

pw.build(frame);
const PWResult& r = pw.run(frame);
// r.energy_ha [Ha], r.energy [Ry], r.magnetization, r.converged
```

GGA activation twin (same geometry/occupations, `xc = XCType::LSDA`) is part
of the regression dual gate, not a separate public API. `PWEvaluator` can own
a moved `PW` for Driver binding; this regression calls `PW` directly.
Forces/stress APIs are **not** part of the spin-GGA acceptance.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_pw-spin-gga -j$(nproc)
  ./build/regressions/reg_pw-spin-gga
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/planewave/pw_solver.h` |
| SCF / spin branch | `src/cpu/planewave/pw_solver.cpp` |
| Spin-GGA XC | `include/atomiverse/cpu/planewave/internal/xc_builder.h`, `src/cpu/planewave/internal/xc_builder.cpp` |
| Density spin | `src/cpu/planewave/internal/density_builder.cpp` |
| Unit tests | `tests/cpu/planewave/` (spin / XC components as present) |
| Regression | `regressions/pw/pw-spin-gga/reg_pw-spin-gga.cpp` |
| Offline pin | `regressions/pw/pw-spin-gga/REFERENCE.md` |
| Oracle note | `tools/cp2k_oracle/inputs/pw-spin-gga.note` |

## Validation (CP2K)

Source:
[`regressions/pw/pw-spin-gga/REFERENCE.md`](../../../regressions/pw/pw-spin-gga/REFERENCE.md)
(offline only; **no live CP2K or Quantum ESPRESSO in ctest**).

| Setting | Value |
| ------- | ----- |
| System | O₂, \(R(\mathrm{O{-}O}) = 1.2075\) Å |
| Geometry | O \((0,0,0)\); O \((0,0,1.2075)\) [Å] |
| Cell | cubic \(L = 8.0\) Å |
| PP | SG15 ONCV PBE O, `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| \(E_{\mathrm{cut}}\) | 12 Ry |
| **k**-mesh | \(1\times1\times1\) (Γ) |
| Multiplicity | 3 (\(^3\Sigma_g^-\)) |
| Occupations | fixed \(N_\uparrow=7\), \(N_\downarrow=5\) |
| SCF | density L2 tol \(5\times10^{-4}\), `max_iter=60`, `mix_beta=0.3` |
| XC | PBE true collinear spin-GGA (`XCBuilder::step_spin`) |
| Twin note | CP2K Quickstep is **GPW/GTH**, not NC-UPF PW; prefer QE `pw.x` + this UPF for absolute \(e_{\mathrm{tot}}\) re-pin when available (`tools/cp2k_oracle/inputs/pw-spin-gga.note`) |

### Magnetization (literature + fixed occupations)

O₂ ground state is the triplet \(^3\Sigma_g^-\) (Herzberg, *Molecular Spectra
and Molecular Structure* I). With fixed integer occupations and multiplicity 3,
\(M = 2\) is an **identity** of the occupation assignment (hard literature gate).

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(M\) [e⁻] | **2** | 2 (fixed occupations / Herzberg multiplet) | \(1\times10^{-9}\) | `PWResult::magnetization` |

### Total energy (offline method pin)

| Quantity | Atomiverse / pin | Twin | Tol | Notes |
| -------- | ---------------: | ---- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−28.7585210635** | offline method pin (same PP/ecut/cell/geom; true spin-GGA) | \(1\times10^{-3}\) Ha | Pin 2026-07-26 (`wf/pw-spin-gga`); no loadable CP2K/`pw.x` on pin host |
| SCF | must converge | — | — | regression fails if `!converged` |

True spin-GGA lowers the energy relative to the historical spin foothold
(PBE@\(\sigma=0\) / LSDA-like, near −28.63 Ha) by ~0.13 Ha at these settings.
Absolute PW total energies shift with cutoff and PP family; the regression
gate is **stability under the frozen method string**, not a literature
chemical-accuracy claim.

### GGA activation pin

Same geometry/occupations with `xc = XCType::LSDA`:

| Quantity | Atomiverse / pin | Twin | Bound | Notes |
| -------- | ---------------: | ---- | ----- | ----- |
| \(\lvert E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}\rvert\) [Ha] | \(\approx 0.129\) | internal dual run | \(> 10^{-4}\) Ha | proves spin path is not stuck in density-only LDA limit of one functional |
| \(E_{\mathrm{LSDA}}\) [Ha] | \(\approx -28.62946347\) | dual run | finite + converged | informational |

Units: public evaluator energy is eV on `frame.meta`; `PWResult` reports Ha/Ry
as elsewhere in PW.

## Performance

Source: `REFERENCE.md` § Perf; queue/handoff mark this slug **perf_exempt**.

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No fair CP2K/QE NC-UPF wall twin on pin host (module dep / no `pw.x`); CP2K Quickstep is GPW/GTH (different basis/PP class) |
| \(T_{\mathrm{ATV}}\) (order of magnitude, single rank, PBE+LSDA) | ~60 s (not for ratio) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a** |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| O₂ PW collinear spin-GGA PBE (NC-UPF) | n/a | ~60 (order) | **n/a** | exempt | No fair CP2K/QE wall twin at pin host |

Dual gate: (1) mag + \(e_{\mathrm{tot}}\) + GGA activation + SCF converge;
(2) perf exempt with reason above. Queue row: ratio **perf_exempt**
(2026-07-26 wave 4).

## Limitations

- **Γ-only molecular cell** on this pin (`k_mesh = 1`); multi-k collinear UKS
  is a sibling product (`pw-uks-kmesh-fe`), not re-audited here for spin-GGA
  activation digits.
- **Fixed occupations only** — metallic smearing is orthogonal
  (`pw-smearing-al`); combined spin + smearing metals are not the acceptance
  of this page.
- **Energy twin debt:** absolute \(E_{\mathrm{tot}}\) is a frozen offline method
  pin (tol \(10^{-3}\) Ha), not a CP2K/QE re-pin — re-pin when `pw.x` or a fair
  NC-UPF twin is available (`tools/cp2k_oracle/inputs/pw-spin-gga.note`).
- **No spin-polarized forces/stress** in acceptance; analytic force/stress APIs
  are not dual-gated for nspin=2 + GGA on this slug.
- **Modest ecut (12 Ry)** for wall budget; production O₂/ONCV needs higher
  cutoff for basis completeness. Absolute energies are not transferable without
  retuning.
- **Sibling vs this slug:** `pw-spin-polarized-o2` was the collinear spin
  foothold; with production `xc=PBE` it now rides true spin-GGA. This page is
  the **dedicated GGA-activation product** (dual LSDA run + energy pin).
  Closed-shell true GGA: `pw-pbe-gga-si`. GPW UKS GGA: `gpw-uks-pbe` /
  `gpw-uks-blyp`.
- **Out of scope forever for this page:** noncollinear spin, SOC, hybrid PW,
  meta-GGA (\(\tau\) channel blocked on PW/GPW engines), CUDA PW spin.

## References

- Offline pin: [`regressions/pw/pw-spin-gga/REFERENCE.md`](../../../regressions/pw/pw-spin-gga/REFERENCE.md)
- Regression consumer: `regressions/pw/pw-spin-gga/reg_pw-spin-gga.cpp`
- Oracle note: `tools/cp2k_oracle/inputs/pw-spin-gga.note`
- Oracle rules: `tools/cp2k_oracle/README.md` (never live in ctest)
- Roadmap: `docs/roadmap/dft.md` (CPU true collinear spin-GGA ✓)
- Narrative PW / GPW chapters: [ch08-plane-wave.md](../ch08-plane-wave.md),
  [ch09-gpw.md](../ch09-gpw.md)
- Related: `pw-spin-polarized-o2` (spin foothold), `pw-pbe-gga-si` (closed-shell
  true GGA), `pw-uks-kmesh-fe` (k-mesh UKS), GPW UKS GGA
  (`gpw-uks-pbe`, `gpw-uks-blyp`)
- Closed spec (history): `.claude/specs/pw-spin-gga/spec.md` (deleted at done;
  git log)
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  — PBE
- Herzberg G., *Molecular Spectra and Molecular Structure* I (O₂ \(^3\Sigma_g^-\)
  ground multiplet)
- D. R. Hamann, optimized norm-conserving Vanderbilt pseudopotentials; SG15
  ONCV library (UPF used offline)
