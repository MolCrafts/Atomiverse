# Collinear spin-polarized SCC-DFTB2 — O₂ triplet

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `spin-dftb2-o2` |
| Regression | `regressions/semiempirical/spin-dftb2-o2/` |
| Public API | `cpu::semiempirical::DFTB2` (`multiplicity`, `spin_w_ha`, Mulliken `m`) |
| Track | TB / SE method loop wave 1 () |
| Roadmap | Spin-polarized DFTB (collinear) — ✓ foothold |
| Family oracle | **DFTB+** (never OpenMOPAC / xtb) |
| Pin oracle | `dftb+` 24.1 (transcribed mio-1-1 + published PBE \(W\)) |

## Summary

**Collinear unrestricted SCC-DFTB2** on gas-phase **O₂** in the triplet
ground multiplet \(^3\Sigma_g^-\) (multiplicity 3). The concrete method
`cpu::semiempirical::DFTB2` owns the SCC loop, dual \(\alpha/\beta\) Fock
matrices, and Mulliken charge/spin populations as members — Frame in,
energy on `frame.meta` (`kEnergy` [eV]), no public `*Config` bag and no
`make_*` / `build_*` factory. Static in-tree **mio-1-1** SK tables
(`DftbSkSet::MioSketch`) drive this pin; collinear spin constants \(W\)
are **not** in those tables — they are explicit `spin_w_ha` input
[Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV),
assigned from `basis_data::kDftbSpinConstantsPbeHa`. Leaving
`multiplicity = 3` with empty `spin_w_ha` throws. Primary gates: total
energy vs DFTB+ 24.1 and total Mulliken spin \(\sum_A m_A \approx 2\).
Regression binary: `reg_spin-dftb2-o2`.
**No live DFTB+ / OpenMOPAC / xtb in ctest** — pins are offline only.

## Method

**Level.** Self-consistent-charge DFTB2 (Elstner et al.) with a **collinear**
spin channel (Köhler / Frauenheim shape; DFTB+ spin module analogue):

- Minimal valence LCAO from Slater–Koster \(H^0(R)\), \(S(R)\) tables.
- Charge SCC: DFTB+ Elstner \(\gamma_{AB}\) + onsite Hubbard \(U\).
- Spin: explicit `spin_w_ha` table. `launch()` converts Ha→eV and collapses
  the shell matrix to the highest-occupied atomic orbital (HOAO) diagonal
  (H \(W_{ss}\), O \(W_{pp}\)), matching DFTB+ `ShellResolvedSpin = No`.
  Dual \(\alpha/\beta\) generalized eigenproblems per SCC cycle.
- Repulsive pair potential \(E_{\mathrm{rep}}\) from the same SK set.

**Charge / spin potentials** (conceptual; eV — after the Ha→eV conversion):

\[
\varepsilon_A^\sigma
  = \sum_B \gamma_{AB}\,\Delta q_B
  + \sigma\, W_A\, m_A,
\qquad
\sigma \in \{+1,-1\}
\quad(\alpha/\beta).
\]

**Energy** (after SCC; public surface reports eV). ATV assembles
\(E=\mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}\)
in `src/cpu/semiempirical/dftb2.cpp`, which is equivalent to the band-sum
form below once the spin/SCC shifts in \(\varepsilon_{i\sigma}\) are
double-count subtracted:

\[
E
  = \sum_{\sigma i} f_{i\sigma}\,\varepsilon_{i\sigma}
  - \tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\Delta q_B
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
**MioSketch-only**); nuclear forces (this pin is energy + Mulliken spin;
open-shell analytic \(F\) is `dftb2-spin-analytic-forces`); NDDO / GFN-xTB
paths (different family oracles).

## Implementation

Public surface only: construct `cpu::semiempirical::DFTB2`, set members
(constructor args / members — **no** `*Config`), bind into `cpu::Evaluate`
at `evaluate::Phase::evaluate`. Header:
`include/atomiverse/cpu/semiempirical/dftb2.h`. That header already
includes `basis_data/dftb_spin_constants.h`. SK data:
`basis_data/dftb_mio_sk_tables.h` (transcribed mio-1-1). Internals stay
private to the method object.

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
dftb.boundary = cpu::semiempirical::DftbBoundary::cluster; // DFTB+ `C` geometry
dftb.charge = 0;
dftb.multiplicity = 3;
dftb.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;
dftb.with_forces = false;        // energy + spin gate only
dftb.scc_tol = 1e-10;
dftb.max_scc = 80;
// default sk_set = DftbSkSet::MioSketch (required for spin channel)
// default shell_resolved_spin = false (DFTB+ ShellResolvedSpin = No)

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
| Mio-1-1 SK (no \(W\)) | `include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_sk_tables.h` |
| PBE spin constants [Ha] | `include/atomiverse/cpu/semiempirical/basis_data/dftb_spin_constants.h` |
| Regression | `regressions/semiempirical/spin-dftb2-o2/reg_spin-dftb2-o2.cpp` |
| Offline pin | `regressions/semiempirical/spin-dftb2-o2/REFERENCE.md` |
| Family oracle tooling | `tools/tb_oracle/README.md` |

## Validation (vs offline pin / DFTB+ family)

Source:
[`regressions/semiempirical/spin-dftb2-o2/REFERENCE.md`](../../../regressions/semiempirical/spin-dftb2-o2/REFERENCE.md)
(offline only; **no live DFTB+ in ctest**).

```text
oracle: dftb+
oracle_version: 24.1
family_oracle: dftb+
```

| Setting | Value |
| ------- | ----- |
| System | O₂, \(R(\mathrm{O{-}O}) = 1.21\) Å (near experimental) |
| Geometry | O \((0,0,0)\); O \((1.21,0,0)\) [Å] |
| Cell | cubic \(L = 30\) Å, `DftbBoundary::cluster` (DFTB+ `C`) |
| SK set | `DftbSkSet::MioSketch` (transcribed mio-1-1) |
| Spin \(W\) | `spin_w_ha = kDftbSpinConstantsPbeHa`; HOAO \(W_{pp}(\mathrm{O})=-0.028\) Ha |
| Charge / multiplicity | \(0\) / **3** (\(^3\Sigma_g^-\)) |
| Valence occupations | \(N_\alpha=7\), \(N_\beta=5\) (12 valence e⁻) |
| SCC | `scc_tol = 1e-10`, `max_scc = 80` |
| Forces | off on this pin (`with_forces = false`) |

### Accuracy table

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) [eV] | **−177.004501172479** | DFTB+ 24.1 **−177.004501133607** | **\(10^{-4}\)** | \(\lvert\Delta\rvert=3.9\times10^{-8}\) eV |
| \(E_{\mathrm{spin}}\) [eV] | **−0.761918814888** | DFTB+ Energy SPIN \(=-0.028\) Ha | **\(10^{-9}\)** | \(W_{pp}(\mathrm{O},\mathrm{PBE})\); \(m_A=1\) |
| \(\sum_A m_A\) [e] | **2.0** (observed \([1.0, 1.0]\)) | 2 (triplet occupations / multiplet) | **0.25** | Mulliken spin sum |
| Per-atom \(\lvert m_A\rvert\) | \(> 0.3\) each | both atoms spin-polarized | — | asymmetry gate \(\lvert m_0-m_1\rvert < 0.2\) |
| \(\Delta q_A\) | observed \([0, 0]\) | homonuclear | finite | reported; not hard-pinned beyond finiteness |

**Family oracle note.** Numerical family oracle for all DFTB rows is **DFTB+**.
This page’s energy gate is the retaken ATV / DFTB+ 24.1 mio-1-1 twin
(published PBE \(W\), `ShellResolvedSpin = No`). The spin-population gates
are literature multiplet / occupation physics. A previous sketch-\(W=-1\) eV
pin sat at −177.2425823576 eV; the shift
\(\Delta E_{\mathrm{spin}}=+0.238081185112\) eV is an identity for this
homonuclear \(\Delta q\equiv 0\) case (see `REFERENCE.md`).

## Performance

Source: `REFERENCE.md` § Perf; handoff marks wave-1 TB slugs **perf_exempt**.

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | DFTB+ 24.1 wall is a cold process + I/O (~0.3 s); ATV Evaluate+DFTB2 is ~0.02 s. Not a fair kernel-vs-kernel wall. |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{DFTB+}}\) | **n/a** |

### Speed table

| Twin | \(T_{\mathrm{DFTB+}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| O₂ collinear spin-DFTB2 (MioSketch) | n/a | n/a (exempt) | **n/a** | exempt | Cold DFTB+ process+I/O vs in-process ATV; not a kernel wall |

Dual gate: (1) energy twin + Mulliken spin physics; (2) perf exempt with
reason above. Queue/handoff: **perf_exempt** (wall twin, not energy-class).

## Limitations

- **MioSketch only for spin.** `DftbSkSet::ThreeOb` is closed-shell energy
  on this product surface; collinear open shell is MioSketch + explicit
  `spin_w_ha`.
- **`spin_w_ha` is required.** `multiplicity != 1` with empty `spin_w_ha`
  throws; there is no silent sketch-\(W\) default and \(W\) does not live
  in `dftb_mio_tables.h`.
- **HOAO collapse only.** `shell_resolved_spin = true` throws
  (`dftb-shell-resolved-spin`).
- **This slug is energy + Mulliken spin.** Open-shell analytic nuclear
  forces are a separate foothold: `dftb2-spin-analytic-forces`.
- **No noncollinear / SOC** (`noncollinear-spin-dftb` remains blocked /
  separate product).
- **No DFTB3 / D3 / PBC / TD** on this page.
- **Oracle discipline.** Family oracle is DFTB+ only — never OpenMOPAC as
  a DFTB energy reference; never xtb (GFN is a different ladder end).
- **Foothold ≠ closed ladder.** Collinear spin-DFTB is ✓ on the roadmap
  table; remaining Phase A rows are tracked in
## References

- Offline pin:
  [`regressions/semiempirical/spin-dftb2-o2/REFERENCE.md`](../../../regressions/semiempirical/spin-dftb2-o2/REFERENCE.md)
- Regression consumer:
  `regressions/semiempirical/spin-dftb2-o2/reg_spin-dftb2-o2.cpp`
- Roadmap: `docs/roadmap/semiempirical.md` (spin-polarized DFTB ✓ collinear)
- Offline oracle tooling: `tools/tb_oracle/README.md`
- Related product pages (same family): [dftb2-3ob-organic](dftb2-3ob-organic.md),
  [dftb2-analytic-forces](dftb2-analytic-forces.md),
  [dftb2-spin-analytic-forces](dftb2-spin-analytic-forces.md)
- Elstner M. et al., *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)
  (SCC-DFTB2)
- Köhler C., Frauenheim T., Hourahine B., Seifert G., Sternberg M.,
  *J. Phys. Chem. A* **111**, 5622 (2007),
  DOI [10.1021/jp068802p](https://doi.org/10.1021/jp068802p)
  (collinear spin-DFTB / DFTB+ spin constants)
- Herzberg G., *Molecular Spectra and Molecular Structure* I
  (O₂ \(^3\Sigma_g^-\) ground multiplet)
