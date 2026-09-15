# SCC-DFTB3 — H₂O / mio-style SK (third-order energy foothold)

*(Technical-report page; TB method-loop harness wave 3. Narrative house style TBD.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb3-water-mio` |
| Regression | `regressions/semiempirical/dftb3-water-mio/` |
| Public API | `cpu::Evaluate` + `cpu::semiempirical::DFTB3` (forces off) |
| Roadmap / queue | `docs/roadmap/semiempirical.md` § DFTB3; queue **done** 2026-07-28 (`reg_dftb3-water-mio`) |
| Oracle | **DFTB+** family only (formula-Gaus2011 module boundary; never OpenMOPAC / xtb) |

## Summary

Self-consistent-charge **DFTB3** (Gaus, Cui, Elstner, *J. Chem. Theory Comput.*
**7**, 931 (2011)) extends Elstner SCC-DFTB2 by a third-order charge-fluctuation
term \(E^{(3)}\) built from the charge derivative of the Elstner \(\gamma\) kernel
(\(\Gamma_{AB}=\partial\gamma_{AB}/\partial q_A\), on-site \(U^{\mathrm{d}}\)).
Atomiverse ships a concrete method `cpu::semiempirical::DFTB3` with parameters as
public members (no `*Config` bag, no `make_*` / `build_*`), bound into
`cpu::Evaluate` at `evaluate::Phase::evaluate`. Integrals and Hubbard data come
from **static in-tree mio-style** tables (`basis_data/dftb_mio_tables.h` + shared
mio-1-1 H–H grid via `internal::MioHHTable`); there is no runtime `.skf` parser.

The product gate `reg_dftb3-water-mio` pins an isolated, slightly off-equilibrium
**H₂O** total energy on the public surface (eV on `frame.meta[kEnergy]`), asserts
SCC **convergence** (not budget exhaustion), and dual-gates against DFTB2 so that
\(|E_{\mathrm{DFTB3}}-E_{\mathrm{DFTB2}}|\) is **commensurate with** \(|E^{(3)}|\).
Absolute energy identity vs DFTB+ + full DFTB.org mio-1-1 is **deferred**
(Hamiltonian-twin / formula pin; host may lack a DFTB+ binary). Homonuclear
reduction (H₂) recovers the DFTB+ 24.1 / mio-1-1 H–H pin of
`dftb2-dftbplus-energy-pin` exactly.

## Method

### Energy (closed shell, molecular)

\[
E = \mathrm{Tr}(\rho H^{0}) + E^{(2)} + E^{(3)} + E_{\mathrm{rep}},
\]

with second- and third-order SCC terms (Gaus/Cui/Elstner 2011 form)

\[
E^{(2)}=\tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\Delta q_B,\qquad
E^{(3)}=\tfrac13\sum_{AB}\Delta q_A^{2}\,\Delta q_B\,\Gamma_{AB}.
\]

Here \(\gamma_{AB}\) is the DFTB+ Elstner charge kernel (on-site
\(\gamma_{AA}=U_A\)), \(\Gamma_{AB}=\partial\gamma_{AB}/\partial q_A\)
(\(\Gamma_{AB}\neq\Gamma_{BA}\); on-site \(\Gamma_{AA}=U^{\mathrm{d}}_A\)),
and \(\Delta q_A\) are Mulliken charge fluctuations from the neutral valence
reference. The SCC atomic potential is the **exact** derivative

\[
V^{(3)}_{C}=\tfrac23\,\Delta q_C\sum_B\Gamma_{CB}\Delta q_B
         +\tfrac13\sum_B\Delta q_B^{2}\,\Gamma_{BC},
\]

so the Euler identity \(\sum_A\Delta q_A V_A=2E^{(2)}+3E^{(3)}\) holds and a
band-structure energy built from \(V\) double-counts by \(E^{(2)}+2E^{(3)}\).
Both \(E^{(2/3)}\) and \(V\) live in private workspace
`internal/dftb3_scc.h` (unit-tested: Euler identity, finite differences,
\(\Delta q\equiv 0\) reduction).

**Correct assembly route (binding).** Both DFTB2 and DFTB3 assemble

\[
E = \mathrm{Tr}(\rho H^{0}) + E^{(2)} + E^{(3)} + E_{\mathrm{rep}}
\]

(not \(E_{\mathrm{band}}-E_{\mathrm{dc}}\) with \(\sum_A V_A\Delta q_A\) only).
Using gross Mulliken populations incorrectly as \(\sum_A V_A q_A\) while
subtracting only \(E^{(2)}+2E^{(3)}\) injects a spurious linear term
\(\sum_A V_A q^{0}_A\) — fixed 2026-07-28; see REFERENCE.

### Conceptual pipeline (TB template)

```text
ParameterSet → SK → H/S → eigen → P → SCC(Hubbard, γ, Γ, Uᵈ) → repulsive → E
```

1. **SK tables.** Ordered pair \((Z_A,Z_B)\) radial grids for \(H_{\mu\nu}(r)\),
   \(S_{\mu\nu}(r)\) plus short-range repulsive (static; length Bohr / energy Ha
   in-header; public surface eV/Å). H–H uses the transcribed mio-1-1 grid shared
   with DFTB0/DFTB2; other H/C/N/O pairs are mio-**style sketches**.
2. **Electronic structure.** Assemble \(H^{0}\), \(S\); generalized eigenproblem;
   density \(P\); Mulliken \(\Delta q_A\).
3. **SCC.** Build \(V^{(2)}+V^{(3)}\); Anderson charge mixing (`mix_beta`, default
   0.3) until \(\max|\Delta q_{\mathrm{out}}-\Delta q_{\mathrm{in}}|<\texttt{scc_tol}\).
   Exhausting `max_scc` **throws** (no silent budget-exhausted energy).
4. **Total energy.** Trace + \(E^{(2)}\) + \(E^{(3)}\) + repulsive →
   `frame.meta.get(kEnergy)` [eV]. Optional forces are central FD of \(E\)
   when `with_forces = true` (this slug: off).

### Geometry and units

| Setting | Value |
| ------- | ----- |
| System | H₂O, \(Z=(8,1,1)\), closed shell, charge 0 |
| Geometry (Å) | O \((0,0,0)\); H \((0.96,0,0)\); H \((-0.24,0.93,0)\) — slightly off-eq |
| Box | cubic **20 Å** (isolation; molecular path) |
| Public energy unit | **eV** on `frame.meta.get(kEnergy)` |
| SCC | `scc_tol = 1e-10`, `max_scc = 50`, `mix_beta = 0.3` (default) |
| Forces | **off** (`with_forces = false`) |

### SK set (binding for this pin)

| Piece | Source in ATV | Oracle twin |
| ----- | ------------- | ----------- |
| H–H \(H_{ss}\), \(S_{ss}\), repulsive, onsite | `dftb_mio_hh_sk.h` via `MioHHTable` (mio-1-1) | DFTB+ `H-H.skf` (shared with DFTB2 pin) |
| O–O, O–H, H–O hoppings / \(U\), \(U^{\mathrm{d}}\) | mio-**style sketch** (`dftb_mio_tables.h`) | **not** bit-identical to mio-1-1 |
| Elements | Z ∈ {1, 6, 7, 8} only; other Z throws | — |

Absolute multi-element identity vs DFTB.org full mio-1-1 dumps is **out of
scope** here (product energy path + formula lineage). Homonuclear H₂ reduction
is the absolute-energy bridge to DFTB+ 24.1.

### Explicit exclusions

- **Not** DFTB0 (non-SCC), **not** pure DFTB2 product claim (DFTB2 is dual-gate only).
- **Not** analytic DFTB3 nuclear forces (pending sibling `dftb3-analytic-forces`).
- **Not** spin-DFTB, **not** PBC / k-mesh, **not** TD-DFTB, **not** D3 dispersion.
- **Not** OpenMOPAC or xtb as reference — family lock is **DFTB+ only**.
- **Not** live DFTB+ in ctest: hard-coded pin only.
- **Not** GFN-xTB (xtb family; not a DFTB3 successor).

## Implementation

Public surface only: construct `DFTB3`, set members, bind into `Evaluate`.
No `*Config` bag, no `make_*` / `build_*` factory.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/semiempirical/dftb3.h"

using namespace ATV;

// H2O slightly off-eq (Å); type = Z; cubic isolation box
cpu::Frame frame(3);
frame.span(kAtom_type)[0] = 8;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_type)[1] = 1;
frame.span(kAtom_x)[1] = 0.96;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
frame.span(kAtom_type)[2] = 1;
frame.span(kAtom_x)[2] = -0.24;
frame.span(kAtom_y)[2] = 0.93;
frame.span(kAtom_z)[2] = 0.0;
// masses, tags, zero forces as in reg_dftb3-water-mio.cpp …
frame.box = cpu::Box::cubic(20.0);

cpu::Evaluate eval;
cpu::semiempirical::DFTB3 dftb3;
dftb3.with_forces = false;
dftb3.scc_tol = 1e-10;
// max_scc = 50, mix_beta = 0.3 defaults
eval.on(evaluate::Phase::evaluate).add_ref(dftb3);
eval.build();
eval.run(frame);

const double E_eV = frame.meta.get(kEnergy);
// gates: SCC residual < 1e-10; |E − (−94.273801189131)| ≤ 1e−6 eV;
//        |E − E_DFTB2| commensurate with |E3|
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
  cmake --build build -j"$(nproc)" --target atomiverse reg_dftb3-water-mio
  cmake --build build --target run_reg_dftb3-water-mio
'
```

Relevant paths:

| Role | Path |
| ---- | ---- |
| Method | `include/atomiverse/cpu/semiempirical/dftb3.h` |
| Impl | `src/cpu/semiempirical/dftb3.cpp` |
| SCC kernel (private) | `include/atomiverse/cpu/semiempirical/internal/dftb3_scc.h` |
| Static mio-style SK | `…/basis_data/dftb_mio_tables.h` |
| mio-1-1 H–H | `…/basis_data/dftb_mio_hh_sk.h` |
| Regression | `regressions/semiempirical/dftb3-water-mio/` |
| Offline twin note | `tools/tb_oracle/inputs/dftbplus/dftb3_h2o_mio_note.md` |
| Unit gates | `tests/cpu/semiempirical/test_dftb3.cpp`, `test_dftb3_scc.cpp` |

## Validation (vs DFTB+ family)

Source: `regressions/semiempirical/dftb3-water-mio/REFERENCE.md`
(offline pin; **never** live DFTB+ in ctest).

```text
oracle: dftb+
oracle_version: formula-Gaus2011 (DFTB+ module boundary; no host DFTB+ binary)
```

### Accuracy gate (ctest) — hard-coded pins

| Quantity | Atomiverse / pin | Oracle / check | Tol | Notes |
| -------- | ---------------: | -------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) | **−94.273801189131 eV** | ATV Hamiltonian twin (provisional) | **1×10⁻⁶ eV** | `kPinEnergyEV` in `reg_*.cpp` |
| SCC residual | \(\lt 1\times10^{-10}\,e\) | must converge | — | residual ~**2.3×10⁻¹⁴** \(e\) in **6** iters |
| \(|E_{\mathrm{DFTB3}}-E_{\mathrm{DFTB2}}|\) | \(\gt 10^{-6}\) eV | third-order active | — | dual Evaluate on same geometry |
| \(\|E_3\|\) ratio | \(0.5\le |dE|/|E^{(3)}| \le 2\) | formula magnitude | — | pin: **0.973** |

### Informational decomposition (at pin geometry)

| Quantity | Value | Unit |
| -------- | ----: | ---- |
| \(E_{\mathrm{tot}}\) (pin) | **−94.273801189131** | eV |
| \(E^{(2)}\) | 0.134894744531 | eV |
| \(E^{(3)}\) | −2.240219×10⁻³ | eV |
| \(|E_{\mathrm{DFTB3}}-E_{\mathrm{DFTB2}}|\) | 2.179184×10⁻³ | eV |
| ratio \(|dE|/|E^{(3)}|\) | **0.973** | — |

Sketch SK + \(U^{\mathrm{d}}\) only: \(\Delta q\) sign pattern on O/H is a
property of the sketch rows, not a claim about production mio-1-1.

### Reduction oracle (absolute, not a snapshot)

For homonuclear H₂ at \(r=0.74\) Å, \(\Delta q\equiv 0\) so
\(E^{(2)}=E^{(3)}=0\) and DFTB3 **must** equal DFTB2. Measured:

| Twin | \(E\) [eV] | Notes |
| ---- | ---------: | ----- |
| DFTB3 H₂ | **−18.366350522958** | after MioHHTable rewire |
| DFTB2 H₂ / DFTB+ 24.1 pin | **−18.36635052** | residual ~3×10⁻⁹ eV |
| Pre-fix DFTB3 H₂ | −18.137397986146 | sketch-vs-mio H–H mismatch (0.229 eV) |

Gated by `tests/cpu/semiempirical/test_dftb3.cpp::ReducesToDftb2ForHomonuclearH2`.

### What invalidated the previous water pin (−107.618793920072 eV)

Three defects, all fixed 2026-07-28 (see REFERENCE for numbers):

1. **DFTB3 read the sketch H–H row** while DFTB0/2 used mio-1-1 → different
   Hamiltonians. Now shares `internal::MioHHTable`.
2. **Undamped Picard SCC** was a period-2 limit cycle (residual ≈3.6 \(e\));
   energy depended on `max_scc` parity. Fixed by Anderson mixing + throw on
   budget exhaustion.
3. **Wrong double-counting** (\(E_{\mathrm{band}}-E_{\mathrm{dc}}\) with
   gross-population confusion) carried spurious \(\sum_A V_A q^{0}_A\).

### DFTB+ absolute oracle still owed

No DFTB+ binary on the pin host (`module spider dftb` → none). Twin note:
`tools/tb_oracle/inputs/dftbplus/dftb3_h2o_mio_note.md`. When DFTB+ + matching
SK land offline: `ThirdOrderFull = Yes` (or diagonal DFTB3), same geometry and
SCC tol → replace `kPinEnergyEV` only after SK fidelity matches.

## Performance

`perf_exempt: true` — host has no DFTB+ binary for fair wall ratio; formula /
Hamiltonian-twin oracle only. Revisit when offline DFTB+ pin infrastructure
lands under `tools/tb_oracle/inputs/dftbplus/`.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O SCC-DFTB3 Evaluate (forces off) | n/a (no DFTB+ binary) | n/a | **perf_exempt** | — | formula pin only |

Gate convention (TB loop): \(r \le 10\) unless `perf_exempt`.

## Limitations

| Item | Status |
| ---- | ------ |
| Third-order SCC energy + exact \(\partial E^{(3)}/\partial\Delta q\) | **closed** (this slug) |
| Anderson mix + throw-on-nonconvergence | **closed** |
| Homonuclear reduction to DFTB2 / DFTB+ H–H pin | **closed** |
| Absolute \(E\) vs DFTB+ + full mio-1-1 for H₂O | **owed** (SK fidelity + host binary) |
| Analytic DFTB3 nuclear forces | open → `dftb3-analytic-forces` |
| Multi-element mio sketch vs full dump | not bit-identical |
| Spin / PBC / k-mesh / TD-DFTB on DFTB3 | out of scope |
| Live oracle in CI | **forbidden** — offline pins only |

Foothold ≠ closed: this slug closes **product DFTB3 energy** on the Evaluate
surface with a dual DFTB2 magnitude gate; it does **not** close absolute
DFTB+/mio-1-1 water identity or analytic forces.

## References

- Offline pin: [`regressions/semiempirical/dftb3-water-mio/REFERENCE.md`](../../../regressions/semiempirical/dftb3-water-mio/REFERENCE.md)
- Twin note: [`tools/tb_oracle/inputs/dftbplus/dftb3_h2o_mio_note.md`](../../../tools/tb_oracle/inputs/dftbplus/dftb3_h2o_mio_note.md)
- Family rules: [`tools/tb_oracle/README.md`](../../../tools/tb_oracle/README.md)
- Roadmap: [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md)
- H–H absolute bridge: [`dftb2-dftbplus-energy-pin`](dftb2-dftbplus-energy-pin.md)
- M. Gaus, Q. Cui, M. Elstner, *J. Chem. Theory Comput.* **7**, 931 (2011),
  DOI [10.1021/ct100684s](https://doi.org/10.1021/ct100684s)
- M. Elstner et al., *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)
- DFTB.org / dftbparams mio-1-1 (H–H fidelity path); DFTB+ family surface
  (`ThirdOrderFull` / diagonal DFTB3) for future absolute pin
