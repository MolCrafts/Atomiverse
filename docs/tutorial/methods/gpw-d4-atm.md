# GPW PBE-D4 + ATM — periodic water dimer (three-body dispersion)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d4-atm` |
| Regression | `regressions/gpw/gpw-d4-atm/` |
| Public API | `cpu::gpw::GpwDft::d4` + `d4_atm`; `cpu::gpw::GPWEvaluator::d4` + `d4_atm`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

Grimme / Caldeweyher **DFT-D4** with Becke–Johnson (rational) damping **plus**
the Axilrod–Teller–Muto **three-body** term at \(s_9 = 1\) (dftd4
`pbe-D4-ATM`) is wired into the **Γ-point GPW** path as an additive London
correction on top of PBE. This product pin reuses the same **periodic
water-dimer cell** as `gpw-d4-water` (12 Å cubic, ice-relevant H-bonded pair)
and gates the **dispersion energy** against offline **dftd4 3.7.0**
(`-f pbe`, default ATM scale).

**Why dftd4 rather than live CP2K QS?** CP2K 2026.1 on Arrhenius was built
**without** the DFTD4 library (`TYPE DFTD4` aborts:
`CP2K build without DFTD4`). The twin geometry / QS input family is retained
under `tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts that link libdftd4;
ATM-specific pin JSON lives under `tools/cp2k_oracle/inputs/gpw-d4-atm/`. The
hard pin is **libdftd4 / dftd4 3.7.0** — the same backend CP2K calls when DFTD4
is enabled.

Public surface: bool members `d4` and `d4_atm` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). `d4_atm` requires `d4` and sets
\(s_9 = 1\); default `d4_atm = false` keeps the two-body product
(`gpw-d4-water`). Dispersion is reported on `GpwResult::e_disp` [Hartree] and
folded into `e_total`. Mutually exclusive with zero-damping `d3`, BJ `d3bj`,
and other free-atom dispersion flags (any two families true throws at run).
Product gate: `regressions/gpw/gpw-d4-atm` (`reg_gpw-d4-atm`). Wall ratio is
**perf_exempt** (E_disp pin; full QS twin wall unfair / unavailable — see
Performance).

## Method

**Total D4 energy** is two-body BJ plus optional ATM:

\[
E_{\mathrm{disp}} = E_{\mathrm{disp}}^{(2)} + s_9\, E_{\mathrm{ATM}},
\]

with \(s_9 = 1\) for `pbe-D4-ATM` and \(s_9 = 0\) for the two-body product.

**Two-body DFT-D4 with BJ damping** (Caldeweyher et al., *J. Chem. Phys.*
**150**, 154122 (2019)), with **periodic** lattice sums over image vectors
\(\mathbf{T}\):

\[
E_{\mathrm{disp}}^{(2)} = -\sum_{A\le B}\sum_{\mathbf{T}}{}^{\prime}
\left[
  \frac{s_6\,C_6^{AB}}{R_{AB,T}^6 + f^6}
  + \frac{s_8\,C_8^{AB}}{R_{AB,T}^8 + f^8}
\right],
\quad
f = a_1\sqrt{3\,r_4^{A} r_4^{B}} + a_2,
\]

where the prime excludes the singular \(R=0\) self term.
\(C_8^{AB} = 3\,C_6^{AB}\,r_4^{A} r_4^{B}\).

**Three-body ATM.** Axilrod–Teller–Muto C9 triples over atoms \(A,B,C\) and
lattice images (dftd4 / simple-dftd4 `disp.f90` form; cutoff **disp3 = 40 Bohr**).
ATM reweights C6 with **neutral-charge** partial charges (\(q = 0\)), matching
simple-dftd4; CN and two-body \(C_6\) still use the EEQ2019 charge model.
ATM damping exponent **alp = 16**. On this dimer cell the ATM shift is small
(\(\approx -9.56\times 10^{-7}\) Ha) but finite and pin-gated.

**Charge / CN model (periodic).** Partial charges from **EEQ2019**;
coordination numbers are EN-weighted erf CNs; \(C_6\) from reference
polarizability interpolation (in-tree tables). Periodic EEQ uses an
**Ewald-split A-matrix** (multicharge-style). Default cutoffs match dftd4:
CN **30 Bohr**, two-body disp **60 Bohr**, three-body **40 Bohr**.

**PBE-D4 parameters** (SAW190103 BJ set; dftd4 `pbe-D4-ATM`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.95948085 | dimensionless |
| \(s_9\) | **1.0** | dimensionless (ATM on) |
| \(a_1\) | 0.38574991 | dimensionless |
| \(a_2\) | 4.80688534 | Bohr |
| alp (ATM) | 16 | dimensionless |

**Electronic host.** Dispersion is geometry-only; the regression still runs a
short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`) so `e_disp` is
materialized on a real `GpwResult` and included in `e_total`. Absolute QS
\(E_{\mathrm{tot}}\) micro-Ha parity is **out of scope** (OT/grid/DIIS
differences; CP2K DFTD4 build unavailable on the pin host).

**Relation to two-body GPW D4.** Slug `gpw-d4-water` is the same geometry with
`d4 = true`, `d4_atm = false` (\(s_9 = 0\)):
**−9.9458842745846774×10⁻⁴ Ha**. This ATM slug must **not** match that pin
(regression asserts \(\lvert E - E_{\mathrm{2body}}\rvert > 10^{-10}\) Ha).

**Relation to molecular D4.** Non-periodic `cpu::scf::D4` (slug
`dft-d4-h2o-pbe`) remains the isolated-molecule / PySCF path (default
`s9 = 0`). GPW `d4` + `d4_atm` adds periodic lattice CN/EEQ/dispersion with
ATM on the same in-tree parameter tables. Isolated ATM context on the dimer
nuclei: **−9.8819741533958208×10⁻⁴ Ha** (dftd4 3.7.0).

**Key exclusions.**

- Two-body-only product (covered by `gpw-d4-water`; \(s_9 = 0\))
- D3 ATM / C9 (`d3_atm`; separate slug `gpw-d3-atm`)
- D4 ATM as a separate forces/stress product gate (energy pin here; forces
  may exist on the free-function path but are not this slug’s e2e gate)
- Full ice Ih supercell / phonons
- Molecular `cpu::scf::D4` / PySCF pin changes
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS
- k-points, UKS, cell stress as part of this slug
- CUDA GPW

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no `*Config`, no
`make_*` / `build_*` factories). Regression uses the concrete solver:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

// Geometry [Å] → Bohr — water dimer (REFERENCE.md / same as gpw-d4-water).
constexpr double kA2B = 1.0 / 0.52917720859;
// … fill 6 GpwAtom rows (2×O + 4×H) with DZVP-MOLOPT-SR-GTH + GTH-PBE-q* …

GpwDft gpw;
const double L = 12.0 * kA2B;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0; // e_disp is geometry-only; SCF only needs to run
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.d4 = true;         // exclusive vs gpw.d3 / gpw.d3bj / …
gpw.d4_atm = true;     // s9 = 1 (pbe-D4-ATM)

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-8 of offline dftd4 3.7.0 ATM pin
// res.e_total includes e_disp
// res.e_disp must differ from the two-body-only pin
```

Driver / Frame path (component; same members):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.d4 = true;
eval.d4_atm = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
```

Build and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_gpw-d4-atm -j$(nproc)
  ./build/regressions/reg_gpw-d4-atm
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `d4`, `d4_atm` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::d4`, `d4_atm` |
| `include/atomiverse/cpu/scf/dftd4.h` / tables | Free-function `dftd4_energy(..., s9)`, molecular `D4` |
| `src/cpu/gpw/gpw_solver.cpp` | `s9 = cfg.d4_atm ? 1.0 : 0.0` into `dftd4_energy` |
| `tests/cpu/gpw/test_cpu_gpw_dftd4.cpp` | Unit gate (isolated + periodic ATM free-function) |
| `regressions/gpw/gpw-d4-atm/reg_gpw-d4-atm.cpp` | Product gate |
| `regressions/gpw/gpw-d4-atm/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-d4-atm/` | POSCAR + pin JSON (ATM + s9=0 context) |

## Validation (CP2K / dftd4)

Source:
[`regressions/gpw/gpw-d4-atm/REFERENCE.md`](../../../regressions/gpw/gpw-d4-atm/REFERENCE.md).
Oracle: offline **dftd4 3.7.0** (`/software/.../dftd4/3.7.0-gfbf-2025b` on
Arrhenius), CLI twin on `tools/cp2k_oracle/inputs/gpw-d4-atm/POSCAR`.
External engine pin — not a golden of Atomiverse’s own output; **never**
launch live dftd4 or CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) (E_disp is basis-independent) |
| XC + disp | PBE host + DFT-D4 BJ + ATM (`s9=1`) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Pin command | `dftd4 -f pbe --json pin_periodic_atm.json POSCAR` |
| Pin line | `Dispersion energy: -9.9554475834240E-04 Eh` |

| Quantity | Atomiverse / pin | dftd4 3.7.0 (offline) | Tol | Notes |
| -------- | ---------------: | --------------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) D4+ATM [Ha] | −0.00099554475834240288 | −0.00099554475834240288 | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) two-body only [Ha] | (must differ) | −9.9458842745846774×10⁻⁴ | \(\Delta > 10^{-10}\) Ha | Context / anti-pin |
| \(E_{\mathrm{disp}}\) isolated ATM [Ha] | (unit test) | −9.8819741533958208×10⁻⁴ | \(1\times 10^{-8}\) Ha | Context / unit test |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (CP2K DFTD4 N/A on pin host) | — | Not absolute-parity gated |

**Tolerance rationale.** Matching geometry, PBE-D4 parameters, EEQ2019,
periodic lattice sums, and ATM (`s9=1`, neutral-charge C6 reweight, disp3 =
40 Bohr) are expected to track dftd4 to nano-Hartree on this cell. Unit tests
also pin isolated ATM and free-function periodic ATM on the same nuclei.

Offline re-pin (human only; never ctest):

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
# + FlexiBLAS / GCCcore from buildenv-gcc module
cd tools/cp2k_oracle/inputs/gpw-d4-atm
dftd4 -f pbe --json pin_periodic_atm.json POSCAR
# Dispersion energy: -9.9554475834240E-04 Eh
# two-body context:
dftd4 -f pbe --mbdscale 0.0 --json pin_periodic_s9_0.json POSCAR
```

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

ATM evaluation is \(O(N^3 \times N_{\mathrm{images}}^2)\) on a 6-atom cell
(still milliseconds). Full GPW SCF wall is dominated by the electronic
structure, not D4/ATM. A fair **dispersion-only** microbenchmark vs dftd4 CLI
is not the product gate; a full twin SCF ratio vs CP2K is unavailable on the
pin host (no DFTD4-linked CP2K) and would be unfair for a geometry-only hard
gate.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (CP2K lacks DFTD4 on Arrhenius) | — (not fair-twinned) | **perf_exempt** | — | Twin input family retained for DFTD4-linked hosts |
| E_disp product gate | n/a | short SCF + D4+ATM (reg) | **perf_exempt** | — | Geometry-only deliverable |

**perf_exempt = true.** Reason (from REFERENCE / queue handoff): primary
deliverable is E_disp pin parity; CP2K QS wall is unavailable (no DFTD4 build)
and a full twin SCF ratio is unfair for a geometry-only hard gate (same policy
as `gpw-d4-water` / `gpw-pbe-d3bj-ice`). Re-measure a strict full-SCF twin later
if product policy requires \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on a
host with libdftd4-linked CP2K.

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic
  D4+ATM lattice-sum / EEQ parity; not a structural ice benchmark.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS is
  not asserted (and CP2K DFTD4 is missing on the pin host).
- **ATM on D4 only for this slug** — D3 C9 / ATM is `gpw-d3-atm` / `d3_atm`.
- **Energy product gate** — this slug does not e2e-gate D4 ATM forces or
  stress (related productization: `gpw-d4-forces` / `gpw-d4-stress` family).
- **PBE parameters only** on this pin; other reference functionals share tables
  but are not e2e-gated here.
- **`d3`, `d3bj`, and `d4` mutually exclusive** — any two true throws at run;
  `d4_atm` is ignored unless `d4` is set.
- **Oracle split:** GPW pin is dftd4 3.7.0; molecular `dft-d4-h2o-pbe` is
  PySCF + pyscf-dispersion — do not mix geometries / BC when comparing.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-d4-atm/REFERENCE.md`
- Oracle inputs: `tools/cp2k_oracle/inputs/gpw-d4-atm/`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related two-body GPW D4: [gpw-d4-water](gpw-d4-water.md)
- Related molecular D4: [dft-d4-h2o-pbe](dft-d4-h2o-pbe.md)
- Related D3(BJ) GPW pin: [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- E. Caldeweyher, C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122
  (2019) (DFT-D4)
- E. Caldeweyher et al., *J. Chem. Phys.* **150**, 154122 (2019) and follow-ons
  (periodic D4 / EEQ2019; ATM via \(s_9\))
- B. M. Axilrod, E. Teller, *J. Chem. Phys.* **11**, 299 (1943); Y. Muto,
  *Proc. Phys. Math. Soc. Jpn.* **17**, 629 (1943) (three-body dispersion)
- dftd4 3.7.0 (CLI pin backend; same library family CP2K links as DFTD4)
