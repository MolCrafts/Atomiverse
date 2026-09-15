# CASSCF — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `casscf-h2-sto3g` |
| Regression | `regressions/scf/casscf-h2-sto3g/` |
| Public API | `cpu::scf::RHF::casscf(Frame&, ncas, nelecas)` → `CASSCFResult` |
| Roadmap | [Post-HF §4.1 Active-space methods](../../roadmap/post-hf.md) |

## Summary

Closed-shell **CASSCF** (complete active-space SCF) on H₂ / STO-3G after a
converged RHF reference: freeze inactive core, full CI in a chosen active
space of spatial MOs, then optimize orbitals until the orbital gradient
vanishes. Product surface is
`RHF::casscf(frame, /*ncas=*/2, /*nelecas=*/2)`: the call drives RHF, freezes
no core on this system (`ncore=0`), and runs dense FCI in the 2-orbital /
2-electron valence space with a two-step orbital Newton macro-loop. Returns
`CASSCFResult{e_casscf, e_corr, ncas, nelecas, ncore, iterations, converged}`
in Hartree. On this full-valence CAS(2,2) foothold there are no non-redundant
orbital rotations (`ncas = n_mo`), so CASSCF coincides with CASCI / FCI /
CCSD in the basis. Offline oracle pin is PySCF 2.14.0 `mcscf.CASSCF` (never
live in ctest). End-to-end gate: `regressions/scf/casscf-h2-sto3g` → binary
`reg_casscf-h2-sto3g`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.0435\) (within 10×;
startup-dominated).

## Method

**Ansatz.** Partition spatial MOs into frozen core, active, and virtual
external. Core is doubly occupied and inactive; the CI expands all
determinants with a fixed number of active electrons in the active
orbitals; orbitals are then optimized (not frozen at RHF as in CASCI):

\[
\lvert\Psi_{\mathrm{CAS}}\rangle
  = \sum_{I \in \mathrm{CAS}} c_I \lvert\Phi_I(\mathbf{C})\rangle,\qquad
E_{\mathrm{CASSCF}}
  = \min_{\mathbf{c},\mathbf{C}}
  \frac{\mathbf{c}^\dagger H(\mathbf{C})\,\mathbf{c}}
       {\mathbf{c}^\dagger\mathbf{c}},
\quad
E_{\mathrm{corr}} = E_{\mathrm{CASSCF}} - E_{\mathrm{RHF}}.
\]

CI coefficients \(\mathbf{c}\) come from the dense active-space FCI kernel
(same Slater–Condon / FCI path as `RHF::casci` / `RHF::fci`, with a
frozen-core energy shift). Orbital coefficients \(\mathbf{C}\) are updated
by a Newton step on the orbital gradient built from spin-summed 1- and
2-RDMs until the generalized Fock is symmetric to tolerance (Roos 1987).
Matches PySCF `mcscf.CASSCF(mf, ncas, nelecas)` on the same closed-shell
foothold.

**Spin treatment.** Closed-shell RHF reference only; active electron count
`nelecas` must be even (equal α/β in the active space). No UHF/ROHF/GHF
CASSCF path.

**Key approximations / scope of this foothold:**

- **State-specific CASSCF** — single root; no state averaging (SA-CASSCF).
- Active-space FCI is **dense** (full determinant basis); no RAS/GAS/DMRG
  selection.
- Energy only — no CAS 1-RDM/2-RDM export on the public surface, no analytic
  gradient/Hessian.
- Canonical MO integrals; no DF/CD/local CAS.
- H₂/STO-3G: 2 AO; full valence CAS(2e,2o); `ncore=0`; zero non-redundant
  rotations → CASSCF = CASCI = FCI to \(\sim 10^{-14}\) Ha in the basis
  (one CI-only macro pass still reports `converged`).

## Implementation

Public headers only for consumers (`Frame` + concrete `RHF`; no `*Config`):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

cpu::Frame frame(2);
// H(0,0,0), H(0,0,0.74) Å; Z=1; Box::cubic(20)
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::CASSCFResult cas =
    rhf.casscf(frame, /*ncas=*/2, /*nelecas=*/2);
// cas.e_casscf, cas.e_corr [Ha]; cas.ncas, cas.nelecas, cas.ncore
// cas.iterations, cas.converged; rhf.energy() = E_RHF [Ha]
// On this system: cas.e_casscf == rhf.casci(frame, 2, 2).e_casci
```

`CASSCFResult` fields (`include/atomiverse/cpu/scf/casscf.h`): `e_casscf`,
`e_corr` [Hartree], `ncas`, `nelecas`, `ncore`, `iterations`, `converged`.
`casscf_energy` is private engine code.
(`internal/casscf_pack.h`).

**Build / run regression** (CPU login; modules as in AGENTS.md):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_casscf-h2-sto3g -j$(nproc)
  ./build/regressions/reg_casscf-h2-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/casscf.h` | Public `CASSCFResult` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::casscf(Frame&, ncas, nelecas)` |
| `include/atomiverse/cpu/scf/internal/casscf_pack.h` | Private pack entry |
| `src/cpu/scf/casscf.cpp` | Dense active-space CI + orbital Newton |
| `src/cpu/scf/mean_field_method.cpp` | `MeanFieldMethod::casscf` dispatch |
| `regressions/scf/casscf-h2-sto3g/reg_casscf-h2-sto3g.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/casscf-h2-sto3g/REFERENCE.md`](../../../regressions/scf/casscf-h2-sto3g/REFERENCE.md)
(offline pin; PySCF never in ctest). Protocol:
`docs/tutorial/appendix-pyscf-oracle.md`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CASSCF | `pyscf.mcscf.CASSCF(mf, ncas=2, nelecas=2)`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occ, 1 virt spatial; `ncore=0` |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CASSCF) | **−0.020524527092077 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{tot}}\) (CASSCF) | **−1.137283834488502 Ha** | **\(10^{-6}\) Ha** | Primary product gate |
| Bookkeeping | `ncas=2`, `nelecas=2`, `ncore=0`, `converged` | exact | Product gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |
| CASSCF ≡ CASCI | \(\lvert E_{\mathrm{CASSCF}} - E_{\mathrm{CASCI}}\rvert\) | \(10^{-10}\) Ha | Full-valence identity |

CASSCF(2,2) coincides with CASCI / FCI / CCSD to \(\sim 10^{-14}\) Ha on this
system (full valence active space; zero non-redundant orbital rotations).

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py casscf \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/casscf-h2-sto3g/REFERENCE.md`](../../../regressions/scf/casscf-h2-sto3g/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥5). Campaign
handoff ratio **0.0435** (the case `REFERENCE.md`). No
standing comparison table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `casscf` (H₂/STO-3G) | 0.2112 | 0.0092 | **0.0435** | ✓ | Startup-dominated foothold; ATV faster than PySCF (\(r \ll 1\)); not a production-scale active-space stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method casscf --geom h2 --basis sto-3g --repeat 5 -- \
    ./build/regressions/reg_casscf-h2-sto3g
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization
pass required on this case.

## Limitations

- **Energy only** on closed-shell RHF; no UHF/ROHF/GHF CASSCF, no state
  averaging.
- **No RAS/GAS**, no DMRG/selected-CI solvers for large active spaces.
- **No MRPT** (CASPT2 / NEVPT2) on the product path yet (NEVPT2 is the next
  queue candidate unblocked by this foothold).
- **No CASSCF analytic gradient/Hessian**, no public RDM export.
- **Dense FCI** active-space kernel only — exponential in active size; not
  production for large CAS.
- **CUDA / Kokkos:** ○ (CPU only).
- **Roadmap next axes** (post-hf §4): SA-CASSCF, CASPT2 / NEVPT2, CASSCF
  analytic gradient — see `docs/roadmap/post-hf.md`.
- **Perf twin** is the same method family on a minimal system; wall times
  are dominated by process startup and a 2-orbital active space with no
  non-redundant rotations, so \(r\) is not a reliable large-CAS predictor.

## References

- Offline pin: `regressions/scf/casscf-h2-sto3g/REFERENCE.md`
- Regression consumer: `regressions/scf/casscf-h2-sto3g/reg_casscf-h2-sto3g.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/post-hf.md` (§4.1 Active-space methods)
- Public API: `include/atomiverse/cpu/scf/casscf.h`
- Sibling (CASCI foothold): `docs/tutorial/methods/casci-h2-sto3g.md`
- B. O. Roos, *The Complete Active Space Self-Consistent Field Method and
  its Applications in Electronic Structure Calculations*, Adv. Chem. Phys.
  **69**, 399 (1987). DOI
  [10.1002/9780470142943.ch7](https://doi.org/10.1002/9780470142943.ch7)
