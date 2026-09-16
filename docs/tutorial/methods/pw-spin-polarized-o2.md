# PW collinear spin-polarized KS — O₂ triplet (Γ)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-spin-polarized-o2` |
| Regression | `regressions/pw/pw-spin-polarized-o2/` |
| Public API | `cpu::pw::PW` (`spin_polarized`, `multiplicity`) → `PWResult::magnetization` |
| Track | CP2K capability wave 1 () |
| Roadmap | PW collinear spin (Γ) — enables later `pw-uks-kmesh-fe` |

## Summary

**Collinear spin-polarized plane-wave Kohn–Sham** (nspin=2, fixed occupations)
on the concrete method `cpu::pw::PW`. Target system is gas-phase **O₂ triplet**
\(^3\Sigma_g^-\) (multiplicity 3) in a cubic cell with a **norm-conserving** SG15
ONCV oxygen UPF. Primary observables: collinear magnetization
\(M = N_\uparrow - N_\downarrow = 2\) and total energy \(E_{\mathrm{tot}}\) [Ha]
at fixed settings (ecut, cell, SCF tol). Restricted default
(`spin_polarized = false`) is unchanged. Regression binary:
`reg_pw-spin-polarized-o2`. **No live CP2K/QE in ctest** — pins are offline
only.

## Method

**Level.** Collinear spin-DFT (von Barth–Hedin / LSD channel on the PW grid):

- Separate \(\alpha/\beta\) KS orbitals and densities \(\rho_\uparrow\),
  \(\rho_\downarrow\).
- Hartree from total density \(\rho = \rho_\uparrow + \rho_\downarrow\).
- Spin-dependent XC potentials \(V_{\mathrm{xc}}^\uparrow(\rho_\uparrow,\rho_\downarrow)\),
  \(V_{\mathrm{xc}}^\downarrow\) via the existing PW XC path
  (`cpu::scf::evaluate_xc`; this pin uses the **PBE-\(\sigma=0\)** LDA-limit
  channel, same as restricted PW when gradients are zeroed on the molecular
  cell).
- Dual Davidson diagonalizations per **k**-point with spin occupations of 1
  (no closed-shell factor of 2).

**Spin / occupations.** Non-metallic fixed fillings from multiplicity \(2S+1\):

\[
N_\uparrow - N_\downarrow = 2S = m - 1,\qquad
N_\uparrow + N_\downarrow = N_{\mathrm{val}}.
\]

Oxygen NC-PP \(Z_{\mathrm{val}} = 6\) → \(N_{\mathrm{val}} = 12\) for O₂;
multiplicity 3 → \(N_\uparrow = 7\), \(N_\downarrow = 5\), magnetization

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
| — | SCF convergence flag / iteration count | — |

**Excluded (this slug).** k-point UKS metals / smearing (`pw-smearing-al`,
`pw-uks-kmesh-fe`); noncollinear / SOC; spin-polarized analytic forces or
stress; GPW UKS (`gpw-uks-o2-gamma`); hybrid / meta-GGA XC; CUDA PW spin.

## Implementation

Public surface only: construct `cpu::pw::PW`, set members (no `*Config` bag,
no `make_*` / `build_*` factory), `build(frame)` → `run(frame)`. Header:
`include/atomiverse/cpu/planewave/pw_solver.h`. Internals (density / XC /
workspace spin buffers) stay under `cpu/planewave/internal/`.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

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

pw.build(frame);
const PWResult& r = pw.run(frame);
// r.energy_ha [Ha], r.energy [Ry], r.magnetization, r.converged
```

`PWEvaluator` can own a moved `PW` for Driver binding; this regression calls
`PW` directly. Forces/stress APIs exist on the restricted path but are **not**
part of the spin-polarized acceptance.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_pw-spin-polarized-o2 -j$(nproc)
  ./build/regressions/reg_pw-spin-polarized-o2
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/planewave/pw_solver.h` |
| SCF / spin branch | `src/cpu/planewave/pw_solver.cpp` |
| Density / XC spin | `src/cpu/planewave/internal/{density,xc}_builder.cpp` |
| Unit test | `tests/cpu/planewave/test_pw_spin.cpp` |
| Regression | `regressions/pw/pw-spin-polarized-o2/reg_pw-spin-polarized-o2.cpp` |
| Offline pin | `regressions/pw/pw-spin-polarized-o2/REFERENCE.md` |
| Oracle note | `tools/cp2k_oracle/inputs/pw-spin-polarized-o2.note` |

## Validation (CP2K)

Source: [`regressions/pw/pw-spin-polarized-o2/REFERENCE.md`](../../../regressions/pw/pw-spin-polarized-o2/REFERENCE.md)
(offline only; **no live CP2K or QE in ctest**).

| Setting | Value |
| ------- | ----- |
| System | O₂, \(R(\mathrm{O{-}O}) = 1.2075\) Å (same as molecular O₂ SCF pins) |
| Geometry | O \((0,0,0)\); O \((0,0,1.2075)\) [Å] |
| Cell | cubic \(L = 8.0\) Å |
| PP | SG15 ONCV PBE O, `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| \(E_{\mathrm{cut}}\) | 12 Ry |
| **k**-mesh | \(1\times1\times1\) (Γ) |
| Multiplicity | 3 (\(^3\Sigma_g^-\)) |
| Occupations | fixed \(N_\uparrow=7\), \(N_\downarrow=5\) |
| SCF | density L2 tol \(5\times10^{-4}\), `max_iter=60`, `mix_beta=0.3` |
| XC | PW collinear path; PBE-\(\sigma=0\) (LDA-limit) as restricted PW |
| Twin note | CP2K Quickstep is **GPW/GTH**, not NC-UPF PW; prefer QE `pw.x` + this UPF for absolute \(e_{\mathrm{tot}}\) re-pin when available |

### Magnetization (literature + fixed occupations)

O₂ ground state is the triplet \(^3\Sigma_g^-\) (Herzberg, *Molecular Spectra
and Molecular Structure* I; standard open-shell computational test). With fixed
integer occupations and multiplicity 3, \(M = 2\) is an **identity** of the
occupation assignment (not a floating SCF observable).

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(M\) [e⁻] | **2** | 2 (fixed occupations / Herzberg multiplet) | \(1\times10^{-9}\) | `PWResult::magnetization` |

### Total energy (offline method pin)

| Quantity | Atomiverse / pin | Twin | Tol | Notes |
| -------- | ---------------: | ---- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−28.3392565801** | offline method pin (same PP/ecut/cell/geom) | \(1\times10^{-3}\) Ha | Host CP2K 2026.1 module unloadable (`hwloc`); no `pw.x` on PATH at pin time |
| SCF | must converge | — | — | regression fails if `!converged` |

Absolute PW total energies shift with cutoff and PP family; the regression
gate is **stability under the frozen method string**, not a literature
chemical accuracy claim. Oracle note:
`tools/cp2k_oracle/inputs/pw-spin-polarized-o2.note` — re-pin against QE with
this UPF when an offline twin exists; magnetization remains the hard literature
gate.

## Performance

Source: `REFERENCE.md` § Perf; queue/handoff mark this slug **perf_exempt**.

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No loadable CP2K binary on pin host for a fair single-rank CP2K PW/NC-UPF twin wall; CP2K Quickstep is GPW/GTH (different basis/PP class) |
| \(T_{\mathrm{ATV}}\) (order of magnitude, single rank) | ~40 s (not for ratio) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a** |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| O₂ PW collinear LSD (NC-UPF) | n/a | ~40 (order) | **n/a** | exempt | No fair CP2K/QE wall twin at pin host |

Dual gate: (1) mag + \(e_{\mathrm{tot}}\) + SCF converge; (2) perf exempt with
reason above. Queue row: ratio **perf_exempt** (2026-07-23 wave 1).

## Limitations

- **Γ-only molecular cell** on this pin (`k_mesh = 1`); periodic k-point UKS is
  a later slug (`pw-uks-kmesh-fe`).
- **Fixed occupations only** here — metallic smearing is orthogonal
  (`pw-smearing-al`); spin + smearing together is not validated on this case.
- **XC channel:** pin path is collinear density with PBE-\(\sigma=0\)
  (LDA-limit) through the shared `evaluate_xc` builder, not a full production
  spin-GGA audit against libxc/QE.
- **No spin-polarized forces/stress** in acceptance; analytic force/stress APIs
  are not dual-gated for nspin=2 on this slug.
- **Energy twin debt:** absolute \(E_{\mathrm{tot}}\) is a frozen offline method
  pin (tol \(10^{-3}\) Ha), not a CP2K/QE re-pin — re-pin when `pw.x` or a
  fair NC-UPF twin is available.
- **Modest ecut (12 Ry)** for wall budget; production O₂/ONCV needs higher
  cutoff for basis completeness.
- **Out of scope forever for this page:** noncollinear spin, SOC, hybrid PW,
  CUDA, GPW UKS (see `gpw-uks-o2-gamma`).

## References

- Offline pin: [`regressions/pw/pw-spin-polarized-o2/REFERENCE.md`](../../../regressions/pw/pw-spin-polarized-o2/REFERENCE.md)
- Regression consumer: `regressions/pw/pw-spin-polarized-o2/reg_pw-spin-polarized-o2.cpp`
- Oracle note: `tools/cp2k_oracle/inputs/pw-spin-polarized-o2.note`
- Narrative PW / GPW chapters: [ch08-plane-wave.md](../ch08-plane-wave.md),
  [ch09-gpw.md](../ch09-gpw.md)
- Related Γ-UKS GPW (different basis/PP): `regressions/gpw/gpw-uks-o2-gamma`
- Herzberg G., *Molecular Spectra and Molecular Structure* I (O₂ \(^3\Sigma_g^-\)
  ground multiplet)
- Hamann D. R., optimized norm-conserving Vanderbilt pseudopotentials; SG15
  ONCV library (UPF used offline)
