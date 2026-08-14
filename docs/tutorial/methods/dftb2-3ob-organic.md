# SCC-DFTB2 — CH₄ / 3ob-3-1 organic SK

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb2-3ob-organic` |
| Regression | `regressions/semiempirical/dftb2-3ob-organic/` |
| Public API | `cpu::semiempirical::DFTB2` + `DftbSkSet::ThreeOb`; `cpu::Evaluate` |
| Roadmap / queue | `docs/roadmap/semiempirical.md`; `docs/developer/tb-method-queue.md` → **done** (wave 1, 2026-07-27) |
| Oracle | **DFTB+ 24.1** (family-locked; never OpenMOPAC / xtb for this pin) |

## Summary

Self-consistent-charge **DFTB2** total energy for isolated methane using the
DFTB.org **3ob-3-1** organic H/C/N/O Slater–Koster tables, transcribed as
static C++ data (`basis_data/dftb_3ob_tables.h`). Public path is concrete
`cpu::semiempirical::DFTB2` with member `sk_set = DftbSkSet::ThreeOb` (no
`*Config` bag, no `make_*` / `build_*`), bound into `cpu::Evaluate` at
`evaluate::Phase::evaluate`. Product gate `reg_dftb2-3ob-organic` asserts

\[
\bigl|E_{\mathrm{ATV}} - (-87.86856831\,\mathrm{eV})\bigr| \le 0.05\,\mathrm{eV}
\]

against an offline DFTB+ 24.1 pin (pure SCC-DFTB2, **no** ThirdOrder). This
slug is **SK fidelity under DFTB2** for production 3ob tables — not a DFTB3
product claim (3ob was trained with third-order corrections; the twin is
deliberately DFTB2). Fair wall vs DFTB+ is **`perf_exempt`**.

## Method

**SCC-DFTB2** (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998); DOI
[10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)). Minimal
valence LCAO with tabulated two-centre hoppings and overlaps; charge
self-consistency on Mulliken fluctuations \(\Delta q_A\).

Conceptual pipeline (binding TB template):

```text
ParameterSet → SK → H/S → eigen → P → SCC(Hubbard, γ) → repulsive → E
```

1. **SK tables.** Ordered pair \((Z_A,Z_B)\) radial grids for \(H_{\mu\nu}(r)\),
   \(S_{\mu\nu}(r)\) plus a short-range repulsive spline (static 3ob-3-1
   transcription; length [Bohr], energy [Ha] in-header; public surface eV/Å).
2. **Electronic structure.** Assemble \(H^0\), \(S\); solve the generalized
   eigenproblem; build density matrix \(P\); Mulliken \(\Delta q_A\).
3. **SCC.** Second-order charge correction via DFTB+ Elstner \(\gamma_{AB}\)`
   and atomic Hubbard \(U_A\) until \(\max|\Delta q|\) drops below `scc_tol`.
4. **Total energy.** Electronic + SCC + repulsive (public
   `frame.meta.get(kEnergy)` [eV]).

**SK set — 3ob-3-1 organic CHNO** (Gaus, Goez, Elstner, *J. Chem. Theory
Comput.* **9**, 338 (2013); DOI
[10.1021/ct300849w](https://doi.org/10.1021/ct300849w)). Elements H, C, N, O
only. In-tree header is CC-BY-SA 4.0 transcription of DFTB.org `.skf`; **no**
runtime SKF I/O.

**Pin Hamiltonian twin (DFTB+):**

| Item | Setting |
| ---- | ------- |
| Spin | Closed-shell (default `multiplicity = 1`) |
| SCC | Yes |
| ThirdOrder | **No** (pure DFTB2) |
| SK | 3ob-3-1 (`H-H`, `C-C`, `C-H`, `H-C`) |
| System | CH₄, \(r(\mathrm{C{-}H})=1.09\) Å, tetrahedral |
| Cell | cubic 20 Å (isolated molecular path) |
| Oracle | DFTB+ **24.1** |

**What this pin includes**

| Item | Value |
| ---- | ----- |
| Method | SCC-DFTB2 energy only |
| SK | `DftbSkSet::ThreeOb` |
| Forces | off (`with_forces = false`) |
| SCC | `scc_tol = 1e-10`, `max_scc = 80` |
| Charge | 0 |

**Key exclusions (non-goals of this slug)**

- **DFTB3** third-order / \(\Gamma\) damping (queue: `dftb3-water-mio`, …).
- Analytic nuclear forces on ThreeOb — **superseded**: closed-form F landed
  2026-07-28 (`dftb2-analytic-forces-threeob`) and the closed-form cell stress
  landed 2026-07-29 (`dftb2-analytic-stress-threeob`). This slug remains an
  energy / SK-fidelity gate.
- Spin-polarized / open-shell (collinear foothold: `spin-dftb2-o2`, MioSketch).
- Dispersion (D3/D4), periodic Γ / k-points, stress, TD-DFTB.
- Full mio-1-1 SK dump fidelity (separate sketch + `dftb2-dftbplus-energy-pin`).
- Live DFTB+ (or any third-party oracle) inside ctest.
- OpenMOPAC / xtb as reference for this family (**forbidden** cross-oracle).
- Public `*Config` bags / `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `DFTB2` members; parameters are members /
ctor defaults — iron law):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

// CH4 tetrahedral, r(C–H) = 1.09 Å; type = Z.
cpu::Frame frame(5);
frame.span(kAtom_type)[0] = 6;
frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] = 0.0;
const double s = 1.09 / std::sqrt(3.0);
const double hpos[4][3] = {
    {s, s, s}, {s, -s, -s}, {-s, s, -s}, {-s, -s, s},
};
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_type)[i + 1] = 1;
    frame.span(kAtom_x)[i + 1] = hpos[i][0];
    frame.span(kAtom_y)[i + 1] = hpos[i][1];
    frame.span(kAtom_z)[i + 1] = hpos[i][2];
}
frame.box = cpu::Box::cubic(20.0);

cpu::Evaluate eval;
cpu::semiempirical::DFTB2 dftb;
dftb.sk_set = cpu::semiempirical::DftbSkSet::ThreeOb;
dftb.with_forces = false;
dftb.scc_tol = 1e-10;
dftb.max_scc = 80;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
eval.build();
eval.run(frame);

const double e_eV = frame.meta.get(kEnergy);  // [eV]
```

**SE-only configure + scoped build/run** (never bare `cmake --build build`;
never unfiltered `ctest`):

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
  cmake --build build -j$(nproc) --target atomiverse reg_dftb2-3ob-organic
  ./build/regressions/reg_dftb2-3ob-organic
  # or: cmake --build build --target run_reg_dftb2-3ob-organic
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/dftb2.h` | Public `DFTB2`, `DftbSkSet` |
| `include/atomiverse/cpu/semiempirical/basis_data/dftb_3ob_tables.h` | Static 3ob-3-1 CHNO tables |
| `include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h` | MioSketch foothold (other slugs) |
| `src/cpu/semiempirical/dftb2.cpp` | SK assemble / SCC / energy (private) |
| `regressions/semiempirical/dftb2-3ob-organic/reg_dftb2-3ob-organic.cpp` | Product gate |
| `tools/tb_oracle/inputs/dftbplus/ch4_3ob_dftb2/` | Offline DFTB+ twin notes |

## Validation (vs DFTB+)

Source:
[`regressions/semiempirical/dftb2-3ob-organic/REFERENCE.md`](../../../regressions/semiempirical/dftb2-3ob-organic/REFERENCE.md).

```text
oracle: dftb+
oracle_version: 24.1
```

Offline pin only — **never** launch DFTB+ from ctest. Twin protocol:
`tools/tb_oracle/README.md`, inputs under
`tools/tb_oracle/inputs/dftbplus/ch4_3ob_dftb2/`.

**Setup:**

| Item | Value |
| ---- | ----- |
| Molecule | CH₄, charge 0, closed shell |
| Geometry [Å] | C at origin; H at \(\pm s,\pm s,\pm s\) with \(s=1.09/\sqrt{3}\) |
| \(r(\mathrm{C{-}H})\) | 1.09 Å |
| Cell | cubic \(L = 20\) Å (isolated) |
| Hamiltonian | SCC-DFTB2, no ThirdOrder |
| SK | 3ob-3-1 H/C |
| Provenance | DFTB+ 24.1 total energy −3.2291103259 Ha |

**DFTB+ energy decomposition (documentation; Ha and eV):**

| Term | Ha | eV |
| ---- | --: | --: |
| Electronic | −3.1685644194 | −86.2210 |
| SCC | 0.0011039499 | 0.0300 |
| Repulsive | −0.0605459065 | −1.6475 |
| **Total** | **−3.2291103259** | **−87.8686** |

Conversion: \(E[\mathrm{eV}] = E[\mathrm{Ha}] \times 27.211386245988\).

**Accuracy gate (product):**

| Quantity | Atomiverse / pin | DFTB+ 24.1 | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | **−87.86856831 eV** | **−87.86856831 eV** (−3.2291103259 Ha) | **0.05 eV** | Product gate `\|E − pin\| ≤ 0.05` |
| Finite energy | required | — | finite | Product gate |

Tolerance rationale (REFERENCE): residual budget covers linear SK
interpolation and Elstner \(\gamma\) vs DFTB+ poly-interp / \(\gamma\)
implementation differences. A **50 meV** e2e band fails gross SK-set / geometry
/ method mistakes while accepting small table-interp residuals.

Offline re-pin (human only; never ctest):

```bash
# when DFTB+ 24.1 is available; see tools/tb_oracle/inputs/dftbplus/ch4_3ob_dftb2/
# SCC = Yes, no ThirdOrder, 3ob-3-1 SK, CH4 r_CH=1.09 Å
# record Total Energy [H] into REFERENCE.md → eV via 27.211386245988
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/semiempirical/dftb2-3ob-organic/REFERENCE.md),
[`docs/developer/tb-method-handoff.md`](../../developer/tb-method-handoff.md)
(wave 1: all three greens `perf_exempt`).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No fair DFTB+ wall on the pin host for ratio at wave-1 harness; energy pin only. |
| DFTB+ wall (REFERENCE, optional note) | ~0.12 s (single core, release, this molecule) |
| \(T_{\mathrm{ATV}}\) | expected well under 10× (small dense eigenproblem) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{DFTB+}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

## Limitations

- **DFTB2 + 3ob SK**, not DFTB3 product accuracy for 3ob training set.
- Energy-only gate on **this** slug. ThreeOb forces and cell stress are now
  closed-form on the product path (`reg_dftb2-analytic-forces-threeob`,
  `reg_dftb2-analytic-stress-threeob`) — there is no finite-difference ThreeOb
  path left in `DFTB2`.
- Organic CHNO only; other 3ob extensions (S, P, …) not in-tree.
- Cross-oracle misuse forbidden: do not re-pin this energy with OpenMOPAC or
  xtb.
- Follow-ons (queue): `dftb2-dftbplus-energy-pin` (mio hygiene), DFTB0/3,
  D3, Γ-PBC, analytic stress, TD-DFTB; GFN-xTB last on the assign ladder.

## References

- Offline pin: `regressions/semiempirical/dftb2-3ob-organic/REFERENCE.md`
- Oracle notes: `tools/tb_oracle/inputs/dftbplus/ch4_3ob_dftb2/README.md`
- Oracle rules: `tools/tb_oracle/README.md` (DFTB* → DFTB+ only)
- Loop design: `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Queue / handoff: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Literature:
  - M. Elstner et al., *Phys. Rev. B* **58**, 7260 (1998)
  - M. Gaus, A. Goez, M. Elstner, *J. Chem. Theory Comput.* **9**, 338 (2013)
- Related narrative: [ch17-semiempirical-am1](../ch17-semiempirical-am1.md)
  (NDDO/AM1; different family — OpenMOPAC oracle)
