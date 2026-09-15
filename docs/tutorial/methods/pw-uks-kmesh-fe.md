# PW k-point collinear UKS — O₂ multi-k (Monkhorst–Pack)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-uks-kmesh-fe` |
| Regression | `regressions/pw/pw-uks-kmesh-fe/` |
| Public API | `cpu::pw::PW` (`spin_polarized`, `multiplicity`, `k_mesh ≥ 2`) → `PWResult::{magnetization, energy_ha}` |
| Track | CP2K capability wave 2 () |
| Roadmap | [dft.md](../../roadmap/dft.md) § k-point periodic UKS; queue `pw-uks-kmesh-fe` (done) |

## Summary

**Collinear spin-polarized plane-wave Kohn–Sham on a multi-k Monkhorst–Pack
mesh** — the ladder step after Γ-only UKS (`pw-spin-polarized-o2`) and restricted
metallic smearing (`pw-smearing-al`). Concrete method `cpu::pw::PW` with
`spin_polarized = true`, `multiplicity = 3`, and **`k_mesh = 2`** (not Γ-only).
Target system is gas-phase **O₂ triplet** \(^3\Sigma_g^-\) in a cubic cell with
SG15 ONCV oxygen NC-UPF. Primary gates: collinear magnetization
\(M = N_\uparrow - N_\downarrow = 2\) and total energy \(E_{\mathrm{tot}}\) [Ha]
at fixed settings. Queue name allows Fe or O₂; **bulk Fe metal is deferred**
(needs spin + metallic smearing together — not co-supported on `PW::run` yet).
Regression binary: `reg_pw-uks-kmesh-fe`. **No live CP2K/QE in ctest** —
offline pins only.

## Method

**Level.** Collinear spin-DFT on the plane-wave grid (same LSD channel as
`pw-spin-polarized-o2`), with **Brillouin-zone sampling**:

- Separate \(\alpha/\beta\) KS orbitals and densities \(\rho_\uparrow\),
  \(\rho_\downarrow\) at **every** \(\mathbf{k}\) of the MP mesh.
- Hartree from total density \(\rho = \rho_\uparrow + \rho_\downarrow\).
- Spin-dependent XC \(V_{\mathrm{xc}}^\uparrow(\rho_\uparrow,\rho_\downarrow)\),
  \(V_{\mathrm{xc}}^\downarrow\) via the existing PW XC path (`step_spin` /
  `cpu::scf::evaluate_xc`).
- Dual Davidson diagonalizations per **k**-point and spin channel; k-weights
  from `generate_mp_mesh`.

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

Multi-k does **not** change fixed integer occupations; \(M = 2\) remains an
identity of the fill assignment (same as Γ UKS).

**Basis / PP / sampling.** Plane-wave cutoff \(E_{\mathrm{cut}} = 5\) Ry (modest
tutorial ecut for multi-k × dual-spin wall budget); Monkhorst–Pack
\(2\times 2\times 2\) (`k_mesh = 2`); cubic cell \(L = 5.5\) Å; UPF
`tests/data/pw/O_ONCV_PBE-1.2.upf` (Hamann ONCVPSP / SG15 library, PBE).

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E\) | Total KS energy (`PWResult::energy`) | Ry |
| \(E_{\mathrm{Ha}}\) | \(E/2\) (`PWResult::energy_ha`) | Ha |
| \(M\) | \(N_\uparrow - N_\downarrow\) (`PWResult::magnetization`) | e⁻ |
| — | SCF convergence flag / iteration count | — |

**Excluded (this slug).** Spin + metallic smearing / bulk Fe metal pin; Fe
NC-UPF product path; noncollinear / SOC; spin-polarized analytic forces or
stress; GPW k-points (`gpw-kpoints-si`); CUDA PW spin; hybrid / meta-GGA XC.
Γ-only UKS remains `pw-spin-polarized-o2`.

## Implementation

Public surface only: construct `cpu::pw::PW`, set members (no `*Config` bag,
no `make_*` / `build_*` factory), `build(frame)` → `run(frame)`. **No new
public members** relative to Γ UKS — this slug gates existing
`spin_polarized` / `multiplicity` / `k_mesh` with `k_mesh ≥ 2`. Header:
`include/atomiverse/cpu/planewave/pw_solver.h`. Internals stay under
`cpu/planewave/internal/`.

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
frame.box = cpu::Box::cubic(5.5);

PW pw;
pw.upf_path = "tests/data/pw/O_ONCV_PBE-1.2.upf";
pw.ecut_ry = 5.0;
pw.k_mesh = 2;                           // multi-k UKS (this slug)
pw.tol = 2e-3;                           // density L2
pw.max_iter = 20;
pw.mix_beta = 0.5;
pw.pulay_depth = 3;
pw.spin_polarized = true;
pw.multiplicity = 3;                     // triplet → M = 2

pw.build(frame);
const PWResult& r = pw.run(frame);
// r.energy_ha [Ha], r.energy [Ry], r.magnetization, r.converged
```

`PWEvaluator` can own a moved `PW` for Driver binding; this regression calls
`PW` directly. Forces/stress APIs are **not** part of the multi-k UKS
acceptance.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_pw-uks-kmesh-fe -j$(nproc)
  ./build/regressions/reg_pw-uks-kmesh-fe
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/planewave/pw_solver.h` |
| SCF / spin × k branch | `src/cpu/planewave/pw_solver.cpp` |
| Unit (multi-k mag) | `tests/cpu/planewave/test_pw_spin.cpp` |
| Regression | `regressions/pw/pw-uks-kmesh-fe/reg_pw-uks-kmesh-fe.cpp` |
| Offline pin | `regressions/pw/pw-uks-kmesh-fe/REFERENCE.md` |
| Oracle note | `tools/cp2k_oracle/inputs/pw-uks-kmesh-fe.note` |

## Validation (CP2K)

Source: [`regressions/pw/pw-uks-kmesh-fe/REFERENCE.md`](../../../regressions/pw/pw-uks-kmesh-fe/REFERENCE.md)
(offline only; **no live CP2K or QE in ctest**). Closed acceptance
(git `pw-uks-kmesh-fe` / wave 2): multi-k collinear UKS, mag + \(e_{\mathrm{tot}}\),
`perf_exempt`.

| Setting | Value |
| ------- | ----- |
| System | O₂, \(R(\mathrm{O{-}O}) = 1.2075\) Å |
| Geometry | O \((0,0,0)\); O \((0,0,1.2075)\) [Å] |
| Cell | cubic \(L = 5.5\) Å |
| PP | SG15 ONCV PBE O, `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| \(E_{\mathrm{cut}}\) | 5 Ry |
| **k**-mesh | \(2\times 2\times 2\) (`k_mesh = 2`) |
| Multiplicity | 3 (\(^3\Sigma_g^-\)) |
| Occupations | fixed \(N_\uparrow=7\), \(N_\downarrow=5\) |
| SCF | density L2 tol \(2\times10^{-3}\), `max_iter=20`, `mix_beta=0.5`, `pulay_depth=3` |
| Twin note | CP2K Quickstep is **GPW/GTH**, not NC-UPF PW; absolute \(e_{\mathrm{tot}}\) is a frozen method pin |

### Magnetization (literature + fixed occupations)

O₂ ground state is the triplet \(^3\Sigma_g^-\) (Herzberg, *Molecular Spectra
and Molecular Structure* I). With fixed integer occupations and multiplicity 3,
\(M = 2\) is an **identity** of the occupation assignment — independent of
k-mesh size.

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(M\) [e⁻] | **2** | 2 (fixed occupations / Herzberg multiplet) | \(1\times10^{-9}\) | `PWResult::magnetization` |

### Total energy (offline method pin)

| Quantity | Atomiverse / pin | Twin | Tol | Notes |
| -------- | ---------------: | ---- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−24.2395602747** | offline method pin (same PP/ecut/cell/geom/k-mesh) | \(5\times10^{-3}\) Ha | Recorded 2026-07-24 Arrhenius x86_64; ~5 SCF iters |
| SCF | must converge | — | — | regression fails if `!converged` |

Absolute PW total energies shift with cutoff, k-mesh, and PP family; the
regression gate is **stability under the frozen method string**, not a
literature chemical-accuracy claim. Optional CP2K documentation twin
(algorithm only, not energy-compared): Quickstep UKS + `KPOINTS`
Monkhorst–Pack \(2\times2\times2\) on the same O₂ cell with GTH + MOLOPT —
see `tools/cp2k_oracle/inputs/pw-uks-kmesh-fe.note`. Magnetization remains the
hard literature gate.

## Performance

Source: `REFERENCE.md` § Perf; queue/handoff mark this slug **perf_exempt**.

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K binary not used as fair single-rank PW/NC-UPF twin wall; GTH-GPW ≠ NC-UPF PW; no loadable comparable twin on pin host |
| \(T_{\mathrm{ATV}}\) (this host, OMP=8, single rank) | **~31 s** |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a** |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| O₂ PW multi-k collinear UKS (NC-UPF) | n/a | ~31 | **n/a** | exempt | No fair CP2K/QE wall twin at pin host |

Dual gate: (1) mag + \(e_{\mathrm{tot}}\) + SCF converge with `k_mesh = 2`;
(2) perf exempt with reason above. Queue row: ratio **perf_exempt**
(2026-07-24 wave 2). Wall budget: e2e ≤ 60 s single-rank CPU.

## Limitations

- **O₂ cell, not bulk Fe.** Queue name `pw-uks-kmesh-fe` reflects the intended
  Fe metal use-case; product pin uses O₂ because spin + metallic smearing is
  not co-supported yet on `PW::run`. Fe metal + spin+smearing is a follow-on.
- **Fixed occupations only** — no metallic smearing on this dual-spin path;
  restricted smearing is `pw-smearing-al`.
- **Modest ecut (5 Ry)** and small cell for multi-k × dual-spin wall budget;
  production ONCV O₂ needs higher cutoff for basis completeness. Absolute
  energy is therefore not chemically converged.
- **No spin-polarized forces/stress** in acceptance.
- **Energy twin debt:** absolute \(E_{\mathrm{tot}}\) is a frozen offline method
  pin (tol \(5\times10^{-3}\) Ha), not a CP2K/QE re-pin — re-pin when `pw.x` or
  a fair NC-UPF twin is available.
- **Out of scope forever for this page:** noncollinear spin, SOC, hybrid PW,
  CUDA, GPW k-points (`gpw-kpoints-si`), Γ-only UKS (see
  `pw-spin-polarized-o2`).

## References

- Offline pin: [`regressions/pw/pw-uks-kmesh-fe/REFERENCE.md`](../../../regressions/pw/pw-uks-kmesh-fe/REFERENCE.md)
- Regression consumer: `regressions/pw/pw-uks-kmesh-fe/reg_pw-uks-kmesh-fe.cpp`
- Oracle note: `tools/cp2k_oracle/inputs/pw-uks-kmesh-fe.note`
- Closed acceptance (git history): `.claude/specs/pw-uks-kmesh-fe/` (deleted at
  done; wave 2 2026-07-24)
- Narrative PW / GPW chapters: [ch08-plane-wave.md](../ch08-plane-wave.md),
  [ch09-gpw.md](../ch09-gpw.md)
- Related Γ UKS PW: [`pw-spin-polarized-o2`](pw-spin-polarized-o2.md)
- Related restricted multi-k: [`pw-lda-si`](pw-lda-si.md),
  [`pw-smearing-al`](pw-smearing-al.md)
- Related Γ-UKS GPW (different basis/PP): `regressions/gpw/gpw-uks-o2-gamma`
- Herzberg G., *Molecular Spectra and Molecular Structure* I (O₂ \(^3\Sigma_g^-\)
  ground multiplet)
- Hamann D. R., optimized norm-conserving Vanderbilt pseudopotentials; SG15
  ONCV library (UPF used offline)
