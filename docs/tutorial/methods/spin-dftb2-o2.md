# Collinear spin-polarized SCC-DFTB2 — O₂ triplet

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `spin-dftb2-o2` |
| Regression | `regressions/semiempirical/spin-dftb2-o2/` |
| Public API | `cpu::semiempirical::DFTB2` (`multiplicity`, Mulliken `m`) |
| Track | TB / SE method loop wave 1 (`tb-method-loop`) |
| Roadmap | Spin-polarized DFTB (collinear) — ✓ foothold |
| Family oracle | **DFTB+** (never OpenMOPAC / xtb) |
| Pin oracle | `literature` + ATV mio-style Hamiltonian twin |

## Summary

**Collinear unrestricted SCC-DFTB2** on gas-phase **O₂** in the triplet
ground multiplet \(^3\Sigma_g^-\) (multiplicity 3). The concrete method
`cpu::semiempirical::DFTB2` owns the SCC loop, dual \(\alpha/\beta\) Fock
matrices, and Mulliken charge/spin populations as members — Frame in,
energy on `frame.meta` (`kEnergy` [eV]), no public `*Config` bag and no
`make_*` / `build_*` factory. Static in-tree **mio-style** SK tables
(`DftbSkSet::MioSketch`) drive this pin; absolute energy identity with a
live DFTB+ binary is **not** claimed (tables are sketches, not DFTB.org
mio-1-1). Primary gates: total energy twin + total Mulliken spin
\(\sum_A m_A \approx 2\). Regression binary: `reg_spin-dftb2-o2`.
**No live DFTB+ / OpenMOPAC / xtb in ctest** — pins are offline only.

## Method

**Level.** Self-consistent-charge DFTB2 (Elstner et al.) with a **collinear**
spin channel (Köhler / Frauenheim shape; DFTB+ spin module analogue):

- Minimal valence LCAO from Slater–Koster \(H^0(R)\), \(S(R)\) tables.
- Charge SCC: DFTB+ Elstner \(\gamma_{AB}\) + onsite Hubbard \(U\).
- Spin: atom-averaged constant \(W_A\) shifts the spin-dependent potential;
  dual \(\alpha/\beta\) generalized eigenproblems per SCC cycle.
- Repulsive pair potential \(E_{\mathrm{rep}}\) from the same SK set.

**Charge / spin potentials** (conceptual; eV):

\[
\varepsilon_A^\sigma
  = \sum_B \gamma_{AB}\,\Delta q_B
  + \sigma\, W_A\, m_A,
\qquad
\sigma \in \{+1,-1\}
\quad(\alpha/\beta).
\]

**Energy** (after SCC; public surface reports eV):

\[
E
  = \sum_{\sigma i} f_{i\sigma}\,\varepsilon_{i\sigma}
  - \tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\,\Delta q_B
  - \tfrac12\sum_A W_A\, m_A^2
  + E_{\mathrm{rep}}.
\]

**Occupations.** Fixed integer fillings from net charge and multiplicity
\(2S+1\):

\[
N_\alpha - N_\beta = 2S = m-1,\qquad
N_\alpha + N_\beta = N_{\mathrm{val}} - q.
\]

Oxygen valence \(Z_{\mathrm{val}}=6\) → \(N_{\mathrm{val}}=12\) for neutral O₂;
multiplicity 3 → \(N_\alpha=7\), \(N_\beta=5\), target total spin population

\[
\sum_A m_A = N_\alpha - N_\beta = 2.
\]

**Observables**

| Symbol | Definition | Unit | Access |
| ------ | ---------- | ---- | ------ |
| \(E\) | Electronic + repulsive total energy | eV | `frame.meta.get(kEnergy)` / `last_energy()` |
| \(\Delta q_A\) | Mulliken charge fluctuation | e | `last_mulliken_dq()` |
| \(m_A\) | Mulliken spin \(N_{\alpha,A}-N_{\beta,A}\) | e | `last_mulliken_spin()` |

**Excluded (this slug).** Noncollinear / SOC spin-DFTB; DFTB3 \(\Gamma\)
correction; D3/PBC/stress/TD-DFTB; ThreeOb open-shell (spin channel is
**MioSketch-only**); absolute DFTB.org mio-1-1 / DFTB+ energy identity
(`dftb2-dftbplus-energy-pin`); open-shell **analytic** nuclear forces
(open shell uses re-SCC central FD when `with_forces`); NDDO / GFN-xTB
paths (different family oracles).

## Implementation

Public surface only: construct `cpu::semiempirical::DFTB2`, set members
(constructor args / members — **no** `*Config`), bind into `cpu::Evaluate`
at `evaluate::Phase::evaluate`. Header:
`include/atomiverse/cpu/semiempirical/dftb2.h`. SK data:
`basis_data/dftb_mio_tables.h` (includes `spin_w`). Internals stay private
to the method object.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

cpu::Frame frame(2);
frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 8;  // Z(O)
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = 1.21;   // R(O–O) [Å]
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_mass)[i] = 15.999;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::cubic(30.0);

cpu::semiempirical::DFTB2 dftb;
dftb.charge = 0;
dftb.multiplicity = 3;           // triplet → Nα=7, Nβ=5
dftb.with_forces = false;        // energy + spin gate only
dftb.scc_tol = 1e-10;
dftb.max_scc = 80;
// default sk_set = DftbSkSet::MioSketch (required for spin channel)

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(dftb);
eval.build();
eval.run(frame);

const double e = frame.meta.get(kEnergy);          // [eV]
const auto& m = dftb.last_mulliken_spin();         // m_A [e]
const auto& dq = dftb.last_mulliken_dq();          // Δq_A [e]
```

Ownership: use `add_ref(lvalue)` when the caller must keep the component
after `run` (Mulliken accessors); `add(rvalue)` transfers ownership into
the plan.

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
  # SCOPED — never bare cmake --build build
  cmake --build build -j$(nproc) --target atomiverse reg_spin-dftb2-o2
  ./build/regressions/reg_spin-dftb2-o2
  # or: cmake --build build --target run_reg_spin-dftb2-o2
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/semiempirical/dftb2.h` |
| Implementation | `src/cpu/semiempirical/dftb2.cpp` |
| Mio-style SK (+ `spin_w`) | `include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h` |
| Regression | `regressions/semiempirical/spin-dftb2-o2/reg_spin-dftb2-o2.cpp` |
| Offline pin | `regressions/semiempirical/spin-dftb2-o2/REFERENCE.md` |
| Family oracle tooling | `tools/tb_oracle/README.md` |

## Validation (vs offline pin / DFTB+ family)

Source:
[`regressions/semiempirical/spin-dftb2-o2/REFERENCE.md`](../../../regressions/semiempirical/spin-dftb2-o2/REFERENCE.md)
(offline only; **no live DFTB+ in ctest**).

```text
oracle: literature
oracle_version: Elstner1998+Kohler2001-method; ATV-mio-style-twin-2026-07-27
family_oracle: dftb+
```

| Setting | Value |
| ------- | ----- |
| System | O₂, \(R(\mathrm{O{-}O}) = 1.21\) Å (near experimental) |
| Geometry | O \((0,0,0)\); O \((1.21,0,0)\) [Å] |
| Cell | cubic \(L = 30\) Å (isolated molecule) |
| SK set | `DftbSkSet::MioSketch` (in-tree exponential sketch + \(W\)) |
| Charge / multiplicity | \(0\) / **3** (\(^3\Sigma_g^-\)) |
| Valence occupations | \(N_\alpha=7\), \(N_\beta=5\) (12 valence e⁻) |
| SCC | `scc_tol = 1e-10`, `max_scc = 80` |
| Forces | off on this pin (`with_forces = false`) |

### Accuracy table

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) [eV] | **−142.1382039484** | ATV mio-style Hamiltonian twin | **0.05** | Not DFTB.org mio-1-1 / DFTB+ absolute identity |
| \(\sum_A m_A\) [e] | **2.0** (observed \([1.0, 1.0]\)) | 2 (triplet occupations / multiplet) | **0.25** | Mulliken spin sum |
| Per-atom \(\lvert m_A\rvert\) | \(> 0.3\) each | both atoms spin-polarized | — | asymmetry gate \(\lvert m_0-m_1\rvert < 0.2\) |
| \(\Delta q_A\) | observed \([0, 0]\) | homonuclear | finite | reported; not hard-pinned beyond finiteness |

**Family oracle note.** Numerical family oracle for all DFTB rows is **DFTB+**.
Absolute energy re-pin against DFTB+ lands when SK fidelity ships
(`dftb2-3ob-organic` offline pin path; deepen slug
`dftb2-dftbplus-energy-pin` for mio). This page’s energy gate is the
**deterministic ATV twin** on the published sketch tables; the spin
population gates are literature multiplet / occupation physics.

## Performance

Source: `REFERENCE.md` § Perf; handoff marks wave-1 TB slugs **perf_exempt**.

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No DFTB+ binary on pin host for a fair wall; absolute \(E\) is ATV SK twin, not a fair DFTB+ energy twin |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{DFTB+}}\) | **n/a** |

### Speed table

| Twin | \(T_{\mathrm{DFTB+}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| O₂ collinear spin-DFTB2 (MioSketch) | n/a | n/a (exempt) | **n/a** | exempt | No fair DFTB+ wall twin at pin host; SK class not mio-1-1 |

Dual gate: (1) energy twin + Mulliken spin physics; (2) perf exempt with
reason above. Queue/handoff: **perf_exempt** (2026-07-27 TB wave 1).

## Limitations

- **MioSketch only for spin.** `DftbSkSet::ThreeOb` is closed-shell energy
  on this product surface; collinear open shell is MioSketch (+ `spin_w`).
- **SK fidelity.** In-tree mio tables are exponential **sketches**, not
  bit-identical DFTB.org mio-1-1 — do not quote absolute \(E\) vs a stock
  DFTB+ mio-1-1 input as an acceptance failure of the spin channel.
- **Open-shell forces** remain re-SCC finite difference (restricted
  analytic F is a separate foothold: `dftb2-analytic-forces`).
- **No noncollinear / SOC** (`noncollinear-spin-dftb` remains blocked /
  separate product).
- **No DFTB3 / D3 / PBC / TD** on this page.
- **Oracle discipline.** Family oracle is DFTB+ only — never OpenMOPAC as
  a DFTB energy reference; never xtb (GFN is a different ladder end).
- **Foothold ≠ closed ladder.** Collinear spin-DFTB is ✓ on the roadmap
  table; Phase A (DFTB0/3, dispersion, PBC, stress, TD, mio absolute pin)
  remains open — see `docs/developer/tb-method-queue.md`.

## References

- Offline pin:
  [`regressions/semiempirical/spin-dftb2-o2/REFERENCE.md`](../../../regressions/semiempirical/spin-dftb2-o2/REFERENCE.md)
- Regression consumer:
  `regressions/semiempirical/spin-dftb2-o2/reg_spin-dftb2-o2.cpp`
- Queue / handoff: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`
- Loop design: `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md` (spin-polarized DFTB ✓ collinear)
- Offline oracle tooling: `tools/tb_oracle/README.md`
- Related product pages (same wave): [dftb2-3ob-organic](dftb2-3ob-organic.md),
  [dftb2-analytic-forces](dftb2-analytic-forces.md)
- Elstner M. et al., *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)
  (SCC-DFTB2)
- Köhler C., Frauenheim T. et al. — collinear spin-DFTB / DFTB+ spin module
  shape (atom-averaged \(W\))
- Herzberg G., *Molecular Spectra and Molecular Structure* I
  (O₂ \(^3\Sigma_g^-\) ground multiplet)
