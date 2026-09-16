# Plane-wave true PBE GGA — Si diamond

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-pbe-gga-si` |
| Regression | `regressions/pw/pw-pbe-gga-si/` |
| Public API | `cpu::pw::PW` (`xc = XCType::PBE`); `pw_forces_analytic` [eV/Å] |
| Roadmap | `docs/roadmap/dft.md` / CP2K queue `pw-pbe-gga-si` (wave 1 ✓) |

## Summary

Spin-unpolarized plane-wave Kohn–Sham DFT with the **true GGA PBE** exchange–correlation
functional on the FFT grid: density gradients enter both \(E_{\mathrm{xc}}\) and
\(v_{\mathrm{xc}}\), not the historical LDA-limit path that called PBE with
\(\sigma = 0\). The product system is bulk silicon (diamond FCC primitive cell, 2 atoms)
at a modest cutoff and \(2\times2\times2\) k-mesh so the e2e gate finishes in seconds.
Public surface: concrete `cpu::pw::PW` with members (`upf_path`, `ecut_ry`, `k_mesh`,
`xc`, …); Frame in / Frame out; analytic ionic forces via `pw_forces_analytic`.
Product gate: `regressions/pw/pw-pbe-gga-si` (`reg_pw-pbe-gga-si`). Offline absolute
CP2K/QE energy twin was **unavailable** on the pin host; correctness uses crystal-symmetry
force zeros plus a PBE–LSDA energy-gap activation pin. Performance is **`perf_exempt`**.

## Method

**Plane-wave KS.** Bloch states expand in plane waves truncated by kinetic cutoff
\(E_{\mathrm{cut}}\):

\[
\psi_{n\mathbf{k}}(\mathbf{r})
=
\sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G})\,
e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}},
\qquad
\tfrac12|\mathbf{k}+\mathbf{G}|^2 \le E_{\mathrm{cut}}.
\]

The SCF loop (owned by concrete `cpu::pw::PW`, not a Plan component graph) is:
Poisson \(V_{\mathrm{H}}(G)\) → XC on the FFT grid → \(V_{\mathrm{eff}}\) → Davidson
diagonalization → density + Pulay mix → residual. Ion–ion energy is 3D Ewald. Total
energy is reported in Rydberg on `PWResult`; the evaluator path writes eV to
`frame.meta.get(kEnergy)`.

**True GGA PBE** (Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)).
When `is_gga(xc)` (e.g. `XCType::PBE`), spectral \(\nabla\rho\) is built from
\(\rho(G)\); pointwise \(e_{\mathrm{xc}}(\rho,\sigma)\) and potential contributions
go through shared `cpu::scf::evaluate_xc` (libxc-parity tree). For closed shell:

\[
v_{\mathrm{xc}}
=
v_\rho
-
\nabla\cdot\Big[
\big(v_{\sigma_{\alpha\alpha}}+\tfrac12 v_{\sigma_{\alpha\beta}}\big)
\nabla\rho
\Big].
\]

Pure LDA is selected with `XCType::LSDA` (pointwise density-only path). Historical
bug fixed by this slug: calling PBE with forced \(\sigma=0\) is **not** true GGA.

**Forces.** Analytic Hellmann–Feynman (plus PP projectors) via
`pw_forces_analytic(frame, pw)` in eV/Å. On ideal diamond sites, lattice symmetry
requires \(\mathbf{F}=\mathbf{0}\) for every atom (product residual bound below).

**Key exclusions (this slug / regression).**

- Spin-polarized PW UKS (`pw-spin-polarized-o2`).
- Metallic smearing (`pw-smearing-al`).
- Analytic Nielsen–Martin stress (`pw-analytic-stress-si`).
- Live CP2K / Quantum ESPRESSO from ctest (offline pins only).
- CUDA PW GGA (CUDA path not this product).
- Absolute total-energy digit match to CP2K/QE (no binary on pin host; see Validation).

## Implementation

Public surface only (`cpu::Frame` + concrete method members; no `*Config`, no factories):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

// Si diamond FCC primitive: a = 5.43 Å; basis (0,0,0), (a/4,a/4,a/4);
// FCC cell matrix as in reg_pw-pbe-gga-si.cpp
ATV::cpu::Frame frame = /* 2 Si, Z=14, mass 28.0855, box = FCC cell */;

ATV::cpu::pw::PW pbe;
pbe.upf_path = "tests/data/pw/Si_lda_nc.upf";  // in-tree ONCVPSP NC UPF
pbe.ecut_ry  = 15.0;                          // wavefunction cutoff [Ry]
pbe.k_mesh   = 2;                             // 2×2×2 Monkhorst–Pack
pbe.tol      = 1e-5;
pbe.max_iter = 40;
pbe.mix_beta = 0.5;
pbe.xc       = ATV::cpu::scf::XCType::PBE;    // true GGA (spectral ∇ρ)

pbe.build(frame);
const auto& r = pbe.run(frame);               // r.energy [Ry]
const auto F = ATV::cpu::pw::pw_forces_analytic(frame, pbe); // [eV/Å]
```

Optional single-point via evaluator (writes `kEnergy` [eV]):

```cpp
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"

ATV::cpu::pw::PWEvaluator pwe{std::move(pbe)};
ATV::cpu::single_point(frame, pwe);
// frame.meta.get(ATV::kEnergy) [eV]; pwe.last_result() [Ry/Ha]
```

Build and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_pw-pbe-gga-si -j$(nproc)
  ./build/regressions/reg_pw-pbe-gga-si
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/planewave/pw_solver.h` | Public `PW`, `PWResult`, `pw_forces_analytic` |
| `include/atomiverse/cpu/planewave/pw_evaluator.h` | Driver-bound `PWEvaluator` |
| `include/atomiverse/cpu/planewave/internal/xc_builder.h` | Internal GGA XC (not public) |
| `src/cpu/planewave/internal/xc_builder.cpp` | Spectral \(\nabla\rho\), divergence \(v_{\mathrm{xc}}\) |
| `tests/cpu/planewave/test_pw_components.cpp` | Unit: uniform \(\sigma=0\) limit; nonuniform GGA gap |
| `regressions/pw/pw-pbe-gga-si/reg_pw-pbe-gga-si.cpp` | Product gate |
| `regressions/pw/pw-pbe-gga-si/REFERENCE.md` | Offline pins + perf_exempt |

## Validation (CP2K)

Source:
[`regressions/pw/pw-pbe-gga-si/REFERENCE.md`](../../../regressions/pw/pw-pbe-gga-si/REFERENCE.md).
**Oracle status:** no CP2K Quickstep or Quantum ESPRESSO binary on the pin host
(`module avail cp2k` / `qe` empty; no `cp2k.psmp` / `pw.x` on `PATH`). Absolute total-energy
digits are **not** invented and **not** taken from a live third-party process at ctest time.
When a twin becomes available offline, extend REFERENCE with version + inputs; do not change
the public API.

**Setup (from REFERENCE):**

| Item | Value |
| ---- | ----- |
| System | Si diamond, FCC primitive (2 atoms) |
| Lattice | \(a = 5.43\) Å (experimental Si) |
| Basis | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Pseudopotential | in-tree ONCVPSP NC UPF `tests/data/pw/Si_lda_nc.upf` (LDA-generated NC; fixed valence PP for GGA path smoke) |
| Cutoff | \(E_{\mathrm{cut}} = 15\) Ry |
| k-mesh | \(2\times2\times2\) Monkhorst–Pack |
| Mixing | Pulay, `mix_beta = 0.5`, `tol = 1e-5`, `max_iter = 40` |
| Spin | unpolarized |

**Product accuracy pins** (hard-coded in the regression binary):

| Quantity | Atomiverse / pin | Twin | Tol / bound | Notes |
| -------- | ---------------: | ---- | ----------- | ----- |
| \(E_{\mathrm{PBE}}\) | finite (SCF preferred converged) | — | finite | no absolute CP2K digit |
| \(\lvert F_i\rvert\) (both Si) | symmetry residual | crystal symmetry \(\mathbf{F}=\mathbf{0}\) | \(< 0.5\) eV/Å | ideal diamond sites |
| \(\lvert E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}\rvert\) | GGA activation | internal dual run | \(> 10^{-4}\) Ry | proves not stuck at density-only LDA limit of one functional |

**Observed Atomiverse numbers (informational only — not a self-golden product pin).**
Worker run 2026-07-23 (`wf/pw-pbe-gga-si`, Release, single rank), from REFERENCE:

| Quantity | Value |
| -------- | ----: |
| \(E_{\mathrm{PBE}}\) | −15.70481672 Ry (converged, 6 iters) |
| \(E_{\mathrm{LSDA}}\) | −15.67175235 Ry (converged, 5 iters) |
| \(\lvert E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}\rvert\) | 0.0330644 Ry |
| \(\lvert F_0\rvert\), \(\lvert F_1\rvert\) | \(\sim 0.0015\) eV/Å |
| Wall (PBE + forces + LSDA dual) | \(\sim 27\) s |

Unit tests (not e2e): uniform density → PBE GGA \(E_{\mathrm{xc}}\) matches PBE@\(\sigma=0\)
within \(10^{-10}\) Ha-scale grid integral; nonuniform density → GGA differs from
\(\sigma=0\) by a meaningful threshold.

## Performance

**`perf_exempt = true`.**

Reason (REFERENCE + queue): no offline CP2K/QE binary on the pin host for a fair wall twin
(\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) undefined). Correctness pins above still apply.
Re-run fair bench when an oracle is installed; target ratio \(\le 10\).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | --------------------------------------: | ---- | ------ |
| PW PBE GGA Si (fair wall) | n/a | n/a | **n/a** | n/a | `perf_exempt` — no CP2K/QE twin on pin host |

Informational ATV wall only (REFERENCE, not a ratio gate): dual PBE+LSDA+forces package
\(\sim 27\) s on the 2026-07-23 worker.

Oracle notes when a twin is later available: `tools/cp2k_oracle/README.md`.
Never spawn CP2K from ctest.

## Limitations

- **No absolute energy oracle yet.** Pins are symmetry forces + internal GGA–LDA gap, not
  a cited CP2K/QE total energy at matched cutoff/PP.
- **PP inconsistency.** The shipped Si UPF is LDA-generated ONCVPSP; it is used only as a
  fixed NC valence potential for the GGA path smoke — not a production PBE-generated PP pair.
- **Modest basis.** \(E_{\mathrm{cut}}=15\) Ry and \(2\times2\times2\) k-mesh are tutorial /
  gate settings; production Si typically wants \(\sim 30\) Ry and denser meshes. Absolute
  energies are not transferable to PseudoDojo deltafactor tables without retuning.
- **Scope gaps (other slugs):** collinear spin, metallic smearing, analytic cell stress,
  CUDA GGA, hybrid XC on PW.
- **Unit vs e2e:** spectral GGA math is unit-tested under `tests/cpu/planewave/`; full SCF
  product lives only under `regressions/pw/pw-pbe-gga-si/` (no e2e in ctest unit suite).

## References

- Offline pin: `regressions/pw/pw-pbe-gga-si/REFERENCE.md`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Narrative PW chapter: `docs/tutorial/ch08-plane-wave.md`
- Roadmap: `docs/roadmap/dft.md` (CPU true GGA PBE ✓)
- Closed spec (history): `.claude/specs/pw-pbe-gga-si/spec.md` (deleted at done; git log)
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013) — ONCVPSP (class of shipped UPF).
- Related regression: `regressions/pw/pw-lda-si` (LDA Si k-mesh energy foothold).
