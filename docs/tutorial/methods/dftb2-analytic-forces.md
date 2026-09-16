# SCC-DFTB2 analytic nuclear forces — H₂O (MioSketch)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb2-analytic-forces` |
| Regression | `regressions/semiempirical/dftb2-analytic-forces/` |
| Public API | `cpu::semiempirical::DFTB2` (`with_forces = true`, restricted MioSketch); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; DFTB is sibling family) |
| Queue | TB method loop — wave 1 done |
| Roadmap | `docs/roadmap/semiempirical.md` § DFTB analytic gradient (restricted) ✓ |

## Summary

Closed-shell **SCC-DFTB2** (Elstner) nuclear forces on off-equilibrium water
using Atomiverse’s static **mio-style sketch** SK tables (H/C/N/O). After one
SCC solve, forces are the **closed-form** nuclear gradient at the SCC density —
Hellmann–Feynman on \(\partial H_0\), Pulay energy-weighted density on
\(\partial S\), fixed-\(\Delta q\) SCC \(\gamma\) geometry terms, plus the
pair repulsive derivative — written to `frame.span(kAtom_f*)` in eV/Å. This is
the **DFTB+ force-module boundary** (Hourahine et al.), **not** a \(6N\) re-SCC
finite-difference of \(E\). Public surface: concrete method
`cpu::semiempirical::DFTB2` with members (`with_forces`, `scc_tol`, `sk_set`, …);
no public `*Config`, no `make_*` / `build_*`. Product gate:
`reg_dftb2-analytic-forces`. Oracle family: **DFTB+** (formula-level module
boundary; offline note only — no live DFTB+ in ctest). `perf_exempt: true`
(no DFTB+ binary / fair wall on pin host).

## Method

**SCC-DFTB2 energy** (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998)):

\[
E = \sum_i n_i \varepsilon_i
  - \tfrac12 \sum_{AB} \Delta q_A \,\gamma_{AB}\,\Delta q_B
  + E_{\mathrm{rep}}.
\]

Parameter path: static SK / Hubbard / repulsive tables → \(H_0\), \(S\) →
generalized eigenproblem → Mulliken \(\Delta q\) → SCC mix of \(\gamma\) shifts
→ band + SCC + repulsive energy. In ATV this is
`ParameterSet → SK → H/S → eigen → P → SCC(Hubbard, γ) → repulsive → E/F`
with parameters as members of `DFTB2` and tables in
`basis_data/dftb_mio_tables.h` for `DftbSkSet::MioSketch`.

**Ionic forces (restricted, closed-form)** — `analytic_forces_restricted` in
`src/cpu/semiempirical/dftb2.cpp`, DFTB+ force structure:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial \mathbf{R}_A}
=
\mathbf{F}_A^{H_0}
+ \mathbf{F}_A^{\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{SCC}\,\gamma}
+ \mathbf{F}_A^{\mathrm{rep}},
\]

with pairwise accumulation (antisymmetrized bond contributions):

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H_0\) | \(\mathrm{Tr}\!\bigl(P\,\partial H_0/\partial R\bigr)\) |
| Pulay + shift on \(S\) | \(-\mathrm{Tr}\!\bigl(W\,\partial S/\partial R\bigr) + \mathrm{Tr}\!\bigl(P\,(\tfrac12(V_A+V_B)\,\partial S/\partial R)\bigr)\) with \(V_A=\sum_B\gamma_{AB}\Delta q_B\), \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| SCC \(\gamma\) (fixed \(\Delta q\)) | \(-\Delta q_A\Delta q_B\,(\partial\gamma_{AB}/\partial R)\) |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) from static pair \(A e^{-b r}\) form |

SK block derivatives \(\partial H_0/\partial R\), \(\partial S/\partial R\) come
from analytic radial derivatives of the exponential hoppings / overlaps in the
mio sketch plus directional factors for \(s\)/\(p\) blocks
(`fill_sk_block_deriv`).

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** so \(\|F\|\neq 0\) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | SCC-DFTB2, closed shell (`multiplicity = 1`, `charge = 0`) |
| SK set | `DftbSkSet::MioSketch` (default) — **not** DFTB.org mio-1-1 bit-identical |
| SCC | `scc_tol = 1\times 10^{-10}` (regression); unit tests use \(10^{-12}\) |
| Forces | `with_forces = true` → closed-form path (not re-SCC FD) |
| Units | \(E\) [eV], \(\mathbf{R}\) [Å], \(\mathbf{F}\) [eV/Å] |
| Observable | Finite \(E\); net \(\sum_A\mathbf{F}_A\approx 0\); analytic vs re-SCC central FD |

**Geometry [Å]:**

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | 1.05 | 0.00 | 0.00 |
| H2 | 1 | −0.26 | 1.00 | 0.00 |

**Key exclusions (this page / acceptance):**

- Absolute \(E\)/\(F\) vs DFTB+ + DFTB.org **mio-1-1** — different slug
  (`dftb2-dftbplus-energy-pin`); sketch SK is not interchangeable for pin identity.
- Production **3ob-3-1** SK forces — out of scope for *this* page, and since
  **shipped**: closed-form ThreeOb F (`dftb2-analytic-forces-threeob`,
  2026-07-28) and cell stress (`dftb2-analytic-stress-threeob`, 2026-07-29).
- Open-shell / collinear spin forces — re-SCC central FD
  (`spin-dftb2-o2` is energy + Mulliken \(m\); analytic open-shell F open).
- Cell **stress**, PBC/k-DFTB, DFTB0/DFTB3, D3, TD-DFTB — other queue rows
  (`dftb2-analytic-stress`, `dftb2-periodic-gamma`, …).
- Live DFTB+ / OpenMOPAC / xtb from ctest — **forbidden**; offline pins only.
- Using OpenMOPAC as a DFTB force reference — family lock forbids it.

**Why FD residual is not machine zero on H₂O.** The analytic path is the
**fixed-\(\Delta q\)** nuclear gradient at the converged SCC density (DFTB+
force shape). The regression FD recomputes a **full re-SCC** energy at
\(\mathbf{R}\pm h\hat{e}\). Charge-response beyond fixed \(\Delta q\) remains
in the multi-element residual: H₂ matches to \(5\times 10^{-3}\) eV/Å in unit
tests; H₂O gate is **1.0 eV/Å** on selected components (same dual-component
gate as the regression).

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::DFTB2` members; no
public `*Config`, no `make_*` / `build_*`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

cpu::Frame frame(3);
// Off-eq water (Å) so |F| ≠ 0
frame.span(kAtom_type)[0] = 8;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_type)[1] = 1;
frame.span(kAtom_x)[1] = 1.05;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
frame.span(kAtom_type)[2] = 1;
frame.span(kAtom_x)[2] = -0.26;
frame.span(kAtom_y)[2] = 1.0;
frame.span(kAtom_z)[2] = 0.0;
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
}
frame.box = cpu::Box::cubic(20.0);
frame.meta.get(kEnergy) = 0;

cpu::semiempirical::DFTB2 dftb;
// dftb.sk_set = DftbSkSet::MioSketch;  // default — analytic F path
dftb.with_forces = true;
dftb.scc_tol = 1e-10;

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
eval.build();
eval.run(frame);

// Energy: frame.meta.get(kEnergy) [eV]
// Forces: kAtom_fx / fy / fz [eV/Å]
```

SE-only build and regression (binding for this loop — never bare
`cmake --build build`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON \
        -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
        -DATV_BUILD_VIB=OFF -DATV_BUILD_PATH=OFF -DATV_BUILD_POTENTIALS=OFF \
        -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF \
        -DATV_BUILD_GPW=OFF -DATV_BUILD_KSPACE=OFF -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse test_cpu_dftb2 \
        reg_dftb2-analytic-forces
  ctest --test-dir build -R "^(test_cpu_dftb2)$" --output-on-failure
  cmake --build build --target run_reg_dftb2-analytic-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/dftb2.h` | Public `DFTB2` / `DftbSkSet` |
| `include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h` | Static mio-style SK / Hubbard / rep |
| `src/cpu/semiempirical/dftb2.cpp` | SCC + `analytic_forces_restricted` + SK derivs |
| `regressions/semiempirical/dftb2-analytic-forces/reg_dftb2-analytic-forces.cpp` | Product gate |
| `regressions/semiempirical/dftb2-analytic-forces/REFERENCE.md` | Offline gates + oracle line |
| `tools/tb_oracle/inputs/dftbplus/dftb2_analytic_forces_note.md` | Why formula-level pin |
| `tests/cpu/semiempirical/test_dftb2.cpp` | Unit analytic-vs-FD (H₂ tight; H₂O dual-comp) |

## Validation (DFTB+)

Source:
[`regressions/semiempirical/dftb2-analytic-forces/REFERENCE.md`](../../../regressions/semiempirical/dftb2-analytic-forces/REFERENCE.md).
Oracle: **`dftb+`** (formula-level / force module boundary — Hourahine et al.,
*J. Chem. Phys.* **152**, 124101 (2020); Elstner 1998).
`oracle_version`: formula-level (no DFTB+ binary on pin host at wave-1).
Hard-coded gates in the regression binary — **never** live DFTB+ from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV SCC-DFTB2 MioSketch closed shell |
| Oracle family | DFTB+ force structure (not OpenMOPAC, not xtb) |
| Offline note | `tools/tb_oracle/inputs/dftbplus/dftb2_analytic_forces_note.md` |
| Atomiverse path | `DFTB2` + `Evaluate`, `with_forces=true`, `scc_tol=1e-10` |
| FD protocol | Central re-SCC energy-only, \(h=10^{-4}\) Å |

**Hard-coded gates (regression):**

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E\) [eV] | finite, \(\|E\| > 10^{-3}\) | hard | Sketch SK — not absolute mio-1-1 DFTB+ \(E\) |
| \(\|F\|^2\) [eV²/Å²] | \(> 10^{-8}\) | hard | Off-eq geometry must produce force |
| \(\|\sum_i F_{i,d}\|\) [eV/Å] | \(\le 10^{-6}\) | hard | Translation invariance, each Cartesian |
| \(F_{\mathrm{O},x}\) vs re-SCC FD [eV/Å] | identity | **1.0** | Charge-response residual class |
| \(F_{\mathrm{H1},y}\) vs re-SCC FD [eV/Å] | identity | **1.0** | Same dual-component product gate |
| H₂ analytic vs re-SCC FD [eV/Å] | unit test | \(5\times 10^{-3}\) | `TEST(DFTB2, AnalyticForceMatchesFdH2)` |

**Tolerance rationale.** Absolute DFTB+ + mio-1-1 forces are **not** the gate
here (SK sketch). The product claim is: (1) closed-form path implements the
DFTB+ force module contractions; (2) net force vanishes; (3) selected
components track re-SCC central FD of ATV’s own \(E\) within the documented
multi-element residual. Tight H₂ unit coverage anchors the same formulas at
\(5\times 10^{-3}\) eV/Å.

## Performance

Source: `REFERENCE.md` — **`perf_exempt: true`**.

No DFTB+ binary / fair wall on the pin host at wave-1. Analytic path replaces
\(6N\) full re-SCC force evaluations with **one** post-SCC contraction of
\(P\), \(W\), \(\partial H_0\), \(\partial S\), \(\partial\gamma\), and
\(\partial E_{\mathrm{rep}}\).

| Twin | \(T_{\mathrm{DFTB+}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| H₂O SCC-DFTB2 ENERGY_FORCE | — | — | — | exempt | No DFTB+ Lmod module / fair bench; formula pin only |

Gate: fair \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin
exists. Handoff records wave-1 TB greens as `perf_exempt`.

## Limitations

- **MioSketch SK is what *this page* gates.** The analytic path has since been
  generalized to the 3ob radial-table derivatives: ThreeOb closed shell now has
  closed-form F **and** a closed-form pair-virial cell stress
  (`reg_dftb2-analytic-forces-threeob`, `reg_dftb2-analytic-stress-threeob`).
- **Restricted closed shell only** — open-shell collinear path still uses
  re-SCC central FD of \(E\) when `with_forces` is set.
- **Fixed-\(\Delta q\) force identity** — full re-SCC FD residual on multi-
  element systems is larger than H₂; product tol is intentionally 1 eV/Å.
- **Not absolute DFTB+ mio-1-1** — do not cite sketch forces as DFTB.org
  numerical twins; deepen via `dftb2-dftbplus-energy-pin` / production SK.
- **No stress / PBC / DFTB3 / D3 / TD** on this page — see Phase A queue.
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure.
- Family lock: **DFTB+ only** for DFTB rows — never OpenMOPAC as DFTB oracle;
  never treat GFN-xTB as DFTB3 successor.

## References

- Offline pin: `regressions/semiempirical/dftb2-analytic-forces/REFERENCE.md`
- Oracle note: `tools/tb_oracle/inputs/dftbplus/dftb2_analytic_forces_note.md`
- Oracle README: `tools/tb_oracle/README.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/dftb2-analytic-forces` /
  wave-1 harness (`.claude/specs/dftb2-analytic-forces/` removed at done)
- M. Elstner et al., *Phys. Rev. B* **58**, 7260 (1998)
- B. Hourahine et al., *J. Chem. Phys.* **152**, 124101 (2020) (DFTB+)
