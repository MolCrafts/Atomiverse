# Gaussian electrode charge kernels — Reed / MetalWalls form (E2)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gaussian-electrode-charge` |
| Regression | `regressions/md/gaussian-electrode-charge/` |
| Public API | `cpu::PairGaussianCoul` (Potential on `cpu::Evaluate` / MD force slot); pure math `cpu::GaussianChargeKernel` |
| Map / queue | MetalWalls capability map **E2**; `docs/developer/metalwalls-method-queue.md` → **done** (wave 1, 2026-07-23) |

## Summary

Electrode atoms in constant-potential MD are modelled as **normalized 3D
Gaussian charge densities** of width \(\eta\) [Å⁻¹], not classical point
charges. This slug delivers the **closed-form Coulomb Green function**
(Reed–Lanning–Madden; MetalWalls electrode Gaussians) as a real-space pair
potential and a header-only kernel used later by the constant-potential BO
matrix \(A\). Public path: construct `cpu::PairGaussianCoul` with a hard
cutoff and a per-`kAtom_type` \(\eta\) table (\(\eta\le 0\) ⇒ point ion),
bind via `Evaluate::set_evaluator` or an MD force phase — **no** public
`*Config` bag, **no** factory. Architecture: a **Component** on the existing
Driver stack, never a new Driver category and never a charge solve inside
`KSpace*`. Product gate: `reg_gaussian-electrode-charge` asserts fixed-\(Q\)
energy and a toy BO \(Q\) against **literature** analytic pins (not live
MetalWalls or LAMMPS). Fair wall is **`perf_exempt`**.

## Method

**Gaussian charge density** (normalized 3D Gaussian; metal units on the public
surface):

\[
\rho_i(\mathbf{r}) = q_i \frac{\eta_i^3}{\pi^{3/2}}
  e^{-\eta_i^2|\mathbf{r}-\mathbf{R}_i|^2}.
\]

**Effective pair width** and **pair Green function**:

\[
\eta_{ij}=\frac{\eta_i\eta_j}{\sqrt{\eta_i^2+\eta_j^2}},
\quad
G_{ij}(r)=\frac{\mathrm{erf}(\eta_{ij} r)}{r},
\quad
G_{ii}=\eta_i\sqrt{\frac{2}{\pi}}.
\]

**Energy** with ATV metal Coulomb prefactor
(`units::coulomb = 14.399645` eV·Å/e²):

\[
U=\texttt{coulomb}\Big[
  \sum_{i<j} q_i q_j G_{ij}(r_{ij})
  +\tfrac12\sum_i q_i^2 G_{ii}\Big].
\]

| Pair kind | Kernel |
| --------- | ------ |
| Gaussian–Gaussian | \(G=\mathrm{erf}(\eta_{ij} r)/r\); self \(G_{ii}\) when enabled |
| Gaussian–point | \(\eta_j\le 0\) flag ⇒ \(\eta_{ij}=\eta_i\) |
| Point–point | \(G=1/r\); **no** self term |

**Forces** use the analytic radial derivative

\[
\frac{dG}{dr}
  = \frac{2\eta_{\mathrm{eff}}}{\sqrt{\pi}}
    \frac{e^{-(\eta_{\mathrm{eff}} r)^2}}{r}
    - \frac{\mathrm{erf}(\eta_{\mathrm{eff}} r)}{r^2}
\]

(point–point: \(-1/r^2\)).

**Why not point-charge \(A\) for constant potential.** Constant-potential BO
solves \(A Q = \Psi\) (plus neutrality). Replacing the Gaussian kernel by
point-charge \(A_{ij}=\texttt{coulomb}/r_{ij}\) (no self diagonal) drops
\(G_{ii}\) and replaces \(\mathrm{erf}(\eta r)/r\) by \(1/r\). On the two-site
toy below, \(Q\) changes by \(O(1)\) and **flips sign**. Production electrode
MD must use the Gaussian kernel for \(A\); this feature is the foothold for
that \(A\).

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Physics | Reed / MetalWalls Gaussian–Gaussian (+ Gaussian–point, point–point) Coulomb |
| Self energy | Optional on `PairGaussianCoul` (default **on**) |
| Geometry (energy) | 5-atom parallel-plate toy, \(L_{\mathrm{box}}=100\) Å, cutoff 40 Å |
| Geometry (BO toy) | Two sites at \(z=0,10\) Å, \(\eta=1.0\) Å⁻¹, \(\Psi=(\pm 0.5)\) eV/e |
| Units | eV, Å, e on the public surface |
| Hosting | `cpu::Evaluate` single-pass energy (forces available from the same potential) |

**Key exclusions (non-goals of this slug)**

- Full constant-potential BO loop / electrode neutrality solve (other queue
  slugs: `const-pot-bo-parallel-plate`, `const-pot-md-nvt-short`, …).
- Reciprocal-space / slab 2D Ewald for Gaussian electrodes
  (`slab-2d-ewald-foothold` is a separate foothold).
- Neighbor-listed \(O(N)\) path — foothold is all-pairs with hard cutoff
  (same pattern as small-cell `PairYukawa`).
- Live MetalWalls or LAMMPS `ELECTRODE` inside ctest (none available on the
  pin host).
- CUDA path; public Config bags / `make_*` factories.

## Implementation

Public surface only (`cpu::Frame` + constructor args; no `*Config`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/gaussian_charge.h"
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"

using namespace ATV;

// Fixed-Q 5-atom toy (REFERENCE.md): electrodes type=1, η=1.0; ion type=0 point.
cpu::Frame f(5);
// L0,L1,R0,R1,ion positions / charges / types — see REFERENCE table
// ...
f.box = cpu::Box::cubic(100.0);

cpu::Evaluate eval;
eval.set_evaluator(cpu::PairGaussianCoul{
    /*cutoff [Å]=*/40.0,
    /*eta_by_type=*/{0.0, 1.0},  // type 0 point, type 1 Gaussian
    /*include_self=*/true});
eval.build();
eval.run(f);
const double E = f.meta.get(kEnergy);  // [eV]

// Math-only (BO matrix elements; no Frame):
const double Gii = cpu::GaussianChargeKernel::self_greens(/*eta=*/1.0);
const double Gij = cpu::GaussianChargeKernel::greens(/*r=*/10.0, 1.0, 1.0);
// A_ij = units::coulomb * G_ij  [eV/e²]
```

Build and run the regression (MD-only configure; modules as in `AGENTS.md`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_gaussian-electrode-charge -j$(nproc)
  ./build/regressions/reg_gaussian-electrode-charge
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/potentials/gaussian_charge.h` | Header-only `GaussianChargeKernel` |
| `include/atomiverse/cpu/potentials/pair_gaussian_coul.h` | `PairGaussianCoul` Potential |
| `src/cpu/potentials/pair_gaussian_coul.cpp` | All-pairs energy/force launch |
| `tests/cpu/potentials/test_gaussian_charge.cpp` | Unit tests (kernel + pair) |
| `regressions/md/gaussian-electrode-charge/reg_gaussian-electrode-charge.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/metalwalls/gaussian-electrode-charge/` | Offline oracle notes (no live twin yet) |

## Validation (oracle: literature)

Source:
[`regressions/md/gaussian-electrode-charge/REFERENCE.md`](../../../regressions/md/gaussian-electrode-charge/REFERENCE.md).

| Field | Value |
| ----- | ----- |
| **oracle** | **literature** (not live LAMMPS; not live MetalWalls) |
| **oracle_version** | Reed *J. Chem. Phys.* **126**, 084704 (2007) Gaussian-charge Coulomb; MetalWalls *JOSS* **5**, 2373 (2020) / Scalfi constant-potential electrode form |
| Why not LAMMPS | Feature is MW-style Gaussian electrode kernels; LAMMPS `ELECTRODE` not on agent host |
| Why not live MetalWalls | No MetalWalls binary on pin host; pin = closed-form evaluation of the **same** published Green function MW uses |
| Live engine in ctest | **Never** |

Offline oracle notes:
`tools/metalwalls_oracle/inputs/metalwalls/gaussian-electrode-charge/README.md`.

### Geometry — fixed-\(Q\) energy pin

Cubic box \(L=100\) Å (min-image irrelevant at this spacing), cutoff **40** Å.

| Atom | type | \(\eta\) [Å⁻¹] | \(q\) [e] | \((x,y,z)\) [Å] |
| ---- | ---- | -------------- | --------- | --------------- |
| L0 | 1 | 1.0 | +0.25 | (0, 0, 0) |
| L1 | 1 | 1.0 | +0.25 | (2, 0, 0) |
| R0 | 1 | 1.0 | −0.25 | (0, 0, 10) |
| R1 | 1 | 1.0 | −0.25 | (2, 0, 10) |
| ion | 0 | point (\(\le 0\)) | +1 | (1, 0, 5) |

**API under test:** `cpu::Evaluate` +
`cpu::PairGaussianCoul{40, {0.0, 1.0}, /*self=*/true}`.

### Accuracy gate (product)

| Quantity | Atomiverse / pin | Oracle (literature) | Tol | Notes |
| -------- | ---------------: | ------------------: | --- | ----- |
| \(E\) [eV] (fixed \(Q\)) | **1.938689655050537** | same closed-form | \(\lvert\Delta E\rvert < 10^{-9}\) eV | Direct eval of Reed/MW \(G\) |
| \(Q_1\) [e] (Gaussian \(A\)) | **+0.0497547591576** | same closed-form | \(\lvert\Delta Q\rvert < 10^{-9}\) e | Two-site BO + neutrality |
| \(\lvert Q_1^{\mathrm{gauss}}-Q_1^{\mathrm{point}}\rvert\) | \(> 10^{-6}\) e (required) | documents point-\(A\) error | — | Point \(A\): \(Q_1=-0.347230782426\) e |
| \(\lim_{r\to 0} G(r)\) vs \(G_{ii}\) | matched in regression | analytic limit | \(10^{-6}\) | Self-consistency sanity |

**Toy BO setup:** two Gaussian sites at \(z=0\) and \(z=L=10\) Å,
\(\eta=1.0\) Å⁻¹, \(\Psi=(+0.5,-0.5)\) eV/e, neutrality \(Q_1+Q_2=0\):

\[
Q_1=\frac{\Psi_1-\Psi_2}{2(A_{11}-A_{12})},\quad
A_{11}=\texttt{coulomb}\,G_{ii},\quad
A_{12}=\texttt{coulomb}\,G(L).
\]

| Model | \(A_{11}\) | \(A_{12}\) | \(Q_1\) [e] |
| ----- | ---------- | ---------- | ----------- |
| Gaussian (this feature) | \(\texttt{coulomb}\,\eta\sqrt{2/\pi}\) | \(\texttt{coulomb}\,\mathrm{erf}(\eta_{\mathrm{eff}} L)/L\) | **+0.0497547591576** |
| Point-charge \(A\) (approx) | 0 (no self) | \(\texttt{coulomb}/L\) | −0.347230782426 |

Pins are **not** goldens of Atomiverse’s own prior output: they are re-evaluable
from the published formulae with `units::coulomb`.

## Performance

Source:
[`REFERENCE.md`](../../../regressions/md/gaussian-electrode-charge/REFERENCE.md)
and queue row in `docs/developer/metalwalls-method-queue.md` (2026-07-23).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No MetalWalls binary and no LAMMPS `ELECTRODE` package on the agent host; pin is an analytic closed-form kernel (not a wall-time twin). |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |
| Foothold complexity | \(O(N^2)\) all-pairs; neighbor-listed path is future work |

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| Fixed-\(Q\) Gaussian Coulomb (5 atoms) | — | — | **exempt** | n/a | No MW / ELECTRODE binary for fair wall; remeasure when available |

Campaign queue records the same: ratio **perf_exempt**. When a MetalWalls or
LAMMPS ELECTRODE twin exists, fair wall should use that binary on the same
host/core count; do **not** invent a ratio.

## Limitations

- **Kernel + pair foothold only.** Full const-pot BO, charge observers,
  charging work, and short NVT electrode MD are separate slugs
  (`const-pot-bo-parallel-plate`, `const-pot-charge-observer`,
  `charging-work-energy`, `const-pot-md-nvt-short`).
- **All-pairs** real-space cut — not suitable as production \(O(N)\) MD for
  large electrodes until a neighbor-listed path lands.
- **Oracle is literature**, not a live engine twin. Physics matches MetalWalls
  Gaussian electrodes by formula identity; bit-parity vs a MW binary is not
  asserted here.
- **`perf_exempt`:** no fair \(r\) until offline MW or LAMMPS ELECTRODE is timed;
  do not invent a speed ratio.
- **CPU only** in this loop; no CUDA electrode Gaussian pair.
- Point-charge \(A\) remains available as a **documented approximation** for
  pedagogy; production electrode \(A\) must use this Gaussian kernel.

## References

- Offline pin: `regressions/md/gaussian-electrode-charge/REFERENCE.md`
- Regression consumer:
  `regressions/md/gaussian-electrode-charge/reg_gaussian-electrode-charge.cpp`
- Oracle notes:
  `tools/metalwalls_oracle/inputs/metalwalls/gaussian-electrode-charge/README.md`
- Oracle tree: `tools/metalwalls_oracle/README.md`
- Queue / loop / handoff: `docs/developer/metalwalls-method-queue.md`,
  `docs/developer/metalwalls-method-loop.md`,
  `docs/developer/metalwalls-method-handoff.md`
- Public API: `include/atomiverse/cpu/potentials/gaussian_charge.h`,
  `include/atomiverse/cpu/potentials/pair_gaussian_coul.h`
- S. K. Reed, O. J. Lanning, P. A. Madden, *J. Chem. Phys.* **126**, 084704
  (2007) — Gaussian-charge electrode Coulomb.
- A. Coretti et al., MetalWalls, *J. Open Source Softw.* **5**, 2373 (2020) —
  constant-potential electrode MD with Gaussian charges.
- L. Scalfi et al., constant-potential electrode / electrochemical MD reviews
  (form of electrode Gaussians and BO \(A\) used by MW).
