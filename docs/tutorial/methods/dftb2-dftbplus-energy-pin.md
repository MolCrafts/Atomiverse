# SCC-DFTB2 H₂ total energy — offline DFTB+ / mio-1-1 pin

*(Technical-report page; TB method-loop wave 2. Narrative house style TBD.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb2-dftbplus-energy-pin` |
| Regression | `regressions/semiempirical/dftb2-dftbplus-energy-pin/` |
| Public API | `cpu::Evaluate` + `cpu::semiempirical::DFTB2` (`sk_set = MioSketch`, forces off) |
| Roadmap / queue | `docs/roadmap/semiempirical.md` §2 SCC-DFTB2; queue done 2026-07-27 |
| Oracle | **DFTB+ 24.1** only (family-locked; never OpenMOPAC / xtb) |

## Summary

Self-consistent-charge DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260
(1998)) on isolated H₂ at \(r(\mathrm{H–H}) = 0.74\) Å is pinned to an offline
**DFTB+ 24.1** total energy obtained with the published **mio-1-1** H–H
Slater–Koster file. Atomiverse does **not** parse `.skf` at run time: the H–H
grids (hopping + overlap + repulsive spline + H onsite / Hubbard \(U\)) are
transcribed once into the static header
`include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_hh_sk.h` and selected
by the default `DftbSkSet::MioSketch` path. The regression asserts absolute
energy identity on the public Evaluate surface (eV on `frame.meta[kEnergy]`).
This is the absolute-energy hygiene pin for H–H; multi-element sketch tables
and 3ob organic SK are separate slugs.

## Method

### SCC-DFTB2 energy

For a closed-shell molecular calculation the DFTB2 total energy is

\[
E = E_{\mathrm{band}} + E_{\mathrm{SCC}} + E_{\mathrm{rep}},
\]

with

\[
E_{\mathrm{band}} = \sum_{i\in\mathrm{occ}} n_i\,\varepsilon_i
  \quad\text{(or equivalently }\operatorname{Tr}(P H^0)\text{ form)},
\]

\[
E_{\mathrm{SCC}} = \tfrac12\sum_{A,B}\Delta q_A\,\gamma_{AB}(R_{AB})\,\Delta q_B,
\]

\[
E_{\mathrm{rep}} = \sum_{A<B} V_{\mathrm{rep}}^{AB}(R_{AB}).
\]

Here \(H^0_{μν}\) and \(S_{μν}\) are assembled from two-centre SK radial tables
(H \(s\) only for H₂), \(\gamma_{AB}\) is the screened Hubbard kernel (DFTB+ Elstner
form with atomic \(U_A\)), and \(\Delta q_A\) are Mulliken charge fluctuations
from the neutral valence reference. At homonuclear H₂ symmetry,
\(\Delta q_A = 0\), so \(E_{\mathrm{SCC}} = 0\) and the pin reduces to band +
repulsive.

### SK set (binding for this pin)

| Piece | Source in ATV | Oracle twin |
| ----- | ------------- | ----------- |
| H–H \(H_{ss}\), \(S_{ss}\) radial grids | `dftb_mio_hh_sk.h` (mio-1-1) | DFTB+ `H-H.skf` |
| H onsite \(E_s\), Hubbard \(U\) | same header | mio-1-1 |
| Repulsive | mio-1-1 spline (static) | DFTB+ spline |
| Other H/C/N/O pairs | exponential **sketch** (`dftb_mio_tables.h`) | **not** bit-identical to mio-1-1 |

Only H–H is mio-1-1-identical on the MioSketch path. Absolute multi-element
identity vs DFTB.org full mio dumps is **out of scope** here (see
`h2o_dftb2_mio`, `dftb2-3ob-organic`).

### Geometry and units

| Setting | Value |
| ------- | ----- |
| System | H₂, \(Z=1\), closed shell, charge 0 |
| \(r(\mathrm{H–H})\) | **0.74 Å** |
| Box | cubic 30 Å (isolation only; molecular min-image) |
| Public energy unit | **eV** on `frame.meta.get(kEnergy)` |
| Internal SK unit | Hartree / Bohr (DFTB+ convention; converted at use site) |
| Ha → eV | \(27.211386245988\) (matches DFTB+ printout / CODATA) |

### Explicit exclusions

- **Not** DFTB0 (non-SCC), **not** DFTB3 (\(E^\gamma\) third-order), **not**
  spin-DFTB, **not** PBC / k-mesh (Γ lattice sum is wave-2 sibling
  `dftb2-periodic-gamma`).
- **Not** analytic forces (sibling `dftb2-analytic-forces`; this reg forces
  `with_forces = false`).
- **Not** OpenMOPAC or xtb as reference — family lock is **DFTB+ only**.
- **Not** live DFTB+ in ctest: hard-coded pin only.

## Implementation

Public surface only: construct `DFTB2`, set members, bind into `Evaluate`.
No `*Config` bag, no `make_*` / `build_*` factory.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

// H2 @ 0.74 Å, type = Z, cubic isolation box
cpu::Frame frame(2);
frame.span(kAtom_x)[0] = 0;  frame.span(kAtom_x)[1] = 0.74;
frame.span(kAtom_y)[0] = frame.span(kAtom_y)[1] = 0;
frame.span(kAtom_z)[0] = frame.span(kAtom_z)[1] = 0;
frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 1;
frame.span(kAtom_mass)[0] = frame.span(kAtom_mass)[1] = 1.008;
frame.span(kAtom_tag)[0] = 1; frame.span(kAtom_tag)[1] = 2;
frame.box = cpu::Box::cubic(30.0);

cpu::Evaluate eval;
cpu::semiempirical::DFTB2 dftb;   // sk_set = MioSketch (default) → mio-1-1 H–H
dftb.with_forces = false;
dftb.scc_tol = 1e-8;
dftb.max_scc = 50;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
eval.build();
eval.run(frame);

const double E_eV = frame.meta.get(kEnergy);
// gate: |E_eV − (−18.36635052)| ≤ 1e−3 eV
```

### SE-only build and run

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
  cmake --build build -j"$(nproc)" --target atomiverse reg_dftb2-dftbplus-energy-pin
  cmake --build build --target run_reg_dftb2-dftbplus-energy-pin
'
```

Relevant paths:

| Role | Path |
| ---- | ---- |
| Method | `include/atomiverse/cpu/semiempirical/dftb2.h` |
| Impl | `src/cpu/semiempirical/dftb2.cpp` |
| Static mio-1-1 H–H | `…/basis_data/dftb_mio_hh_sk.h` |
| Regression | `regressions/semiempirical/dftb2-dftbplus-energy-pin/` |
| Offline twin | `tools/tb_oracle/inputs/dftbplus/h2_mio_scc/` |

## Validation (vs DFTB+)

Source: `regressions/semiempirical/dftb2-dftbplus-energy-pin/REFERENCE.md`
(offline pin; **never** live DFTB+ in ctest).

```text
oracle: dftb+
oracle_version: 24.1
```

### Oracle decomposition (DFTB+ 24.1, mio-1-1, SCC on)

| Quantity | Value | Unit |
| -------- | ----: | ---- |
| Total energy | −0.6749509324 | Ha |
| Total energy (pin) | **−18.36635052** | eV |
| Band energy | −0.6807883427 | Ha |
| Repulsive | +0.0058374104 | Ha |
| SCC energy | 0 (\(\Delta q = 0\)) | Ha |

DFTB+ inputs (reproducible offline only):

- `tools/tb_oracle/inputs/dftbplus/h2_mio_scc/geo.gen` — H₂ at 0.74 Å
- `tools/tb_oracle/inputs/dftbplus/h2_mio_scc/dftb_in.hsd` — `SCC = Yes`,
  `SCCTolerance = 1e-8`, `MaxAngularMomentum { H = "s" }`, Type2 SK names
- SK: dftbparams/mio `H-H.skf` (operator host; not vendored as binary)

### Accuracy gate (ctest)

| Quantity | Atomiverse / pin | DFTB+ 24.1 | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | `kEnergy` | **−18.36635052 eV** | **1×10⁻³ eV** | hard-coded in `reg_*.cpp` |
| Host residual | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert\) | — | — | ~**3×10⁻⁹ eV** on pin host |

Tolerance is intentionally loose vs residual (mHa-scale hygiene for SK
interpolation / unit conversion drift); measured residual is near machine
identity for the transcribed tables + DFTB+ “new” SK interpolation + spline
repulsive (see `tools/tb_oracle/pin_notes.md`).

## Performance

Informative only — **not** asserted in ctest. Fair-wall style numbers from
REFERENCE on the pin host:

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----------------------------------------: | ---- | ------ |
| H₂ SCC-DFTB2 Evaluate (forces off) | ~0.15 (DFTB+ 24.1 cold, incl. I/O) | ~0.06 | **≈ 0.4** | ✓ | Cold DFTB+ includes SK/I/O; ATV tables are static |

Gate convention (TB loop): \(r \le 10\) unless `perf_exempt`. This case is
well inside the wall.

## Limitations

| Item | Status |
| ---- | ------ |
| Absolute \(E\) on H–H mio-1-1 | **closed** (this slug) |
| Multi-element mio **sketch** vs full mio-1-1 dump | not bit-identical; product energy path only |
| 3ob-3-1 organic CHNO absolute pin | sibling `dftb2-3ob-organic` |
| Analytic nuclear forces (restricted MioSketch) | sibling `dftb2-analytic-forces` |
| ThreeOb / open-shell forces | still re-SCC FD (deepen slugs) |
| DFTB0 / DFTB3 / D3 / k-mesh / stress / TD-DFTB | open or incomplete tips |
| Live oracle in CI | **forbidden** — offline pins only |

Foothold ≠ closed: SCC-DFTB2 product energy was already ✓ before this pin;
the pin closes **absolute oracle identity for H–H**, not the full DFTB ladder.

## References

- Offline pin: [`regressions/semiempirical/dftb2-dftbplus-energy-pin/REFERENCE.md`](../../../regressions/semiempirical/dftb2-dftbplus-energy-pin/REFERENCE.md)
- Oracle twin: [`tools/tb_oracle/inputs/dftbplus/h2_mio_scc/`](../../../tools/tb_oracle/inputs/dftbplus/h2_mio_scc/)
- Family rules: [`tools/tb_oracle/README.md`](../../../tools/tb_oracle/README.md)
- Roadmap: [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md) §2
- M. Elstner et al., *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)
- DFTB.org / dftbparams mio-1-1 `H-H.skf`; DFTB+ 24.1
