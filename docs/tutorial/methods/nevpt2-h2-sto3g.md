# SC-NEVPT2 — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `nevpt2-h2-sto3g` |
| Regression | `regressions/scf/nevpt2-h2-sto3g/` |
| Public API | `cpu::scf::RHF::nevpt2(Frame&, ncas, nelecas)` → `NEVPT2Result` |
| Roadmap | [Post-HF §4.2 Multireference correlation](../../roadmap/post-hf.md) |

## Summary

Closed-shell **strongly contracted NEVPT2** (SC-NEVPT2) on a full-valence
CASSCF(2,2) reference for H₂ / STO-3G. Product surface is
`RHF::nevpt2(frame, /*ncas=*/2, /*nelecas=*/2)`: the call drives RHF →
CASSCF(2e,2o) → the eight SC-NEVPT2 perturber classes and returns
`NEVPT2Result{e_nevpt2, e_corr, e_casscf, ncas, nelecas, ncore, nvirt,
converged}` in Hartree. On this system the active space spans **all** MOs
(`ncore = 0`, `nvirt = 0`), so every SC perturber class requires at least
one inactive or external index and the PT2 correction is **identically
zero**: \(E_{\mathrm{NEVPT2}} = E_{\mathrm{CASSCF}} = E_{\mathrm{FCI}}\).
Offline oracle pin is PySCF 2.14.0 `mcscf.CASSCF` + `mrpt.NEVPT` (never live
in ctest). End-to-end gate: `regressions/scf/nevpt2-h2-sto3g` → binary
`reg_nevpt2-h2-sto3g`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.0723\) (within 10×;
startup-dominated).

## Method

**Ansatz.** Start from a state-specific CASSCF reference
\(\lvert\Psi^{(0)}\rangle = \lvert\Psi_{\mathrm{CAS}}\rangle\) (same
closed-shell path as `RHF::casscf`). Partition spatial MOs into frozen
core \(i,j\), active \(t,u\), and external virtual \(r,s\). SC-NEVPT2
applies second-order *n*-electron valence perturbation theory with a
Dyall zeroth-order Hamiltonian and **strongly contracted** perturbers
(Angeli et al. 2001, 2002). Each of the eight excitation classes
(Sr, Si, Sij, Srs, Sir, Sijr, Srsi, Sijrs) contributes one contracted
function per class label; the PT2 energy is the sum of class
contributions:

\[
E_{\mathrm{corr}}^{\mathrm{SC\text{-}NEVPT2}}
  = \sum_{\mu}
  \frac{
    \lvert\langle\Psi^{(0)}\lvert H \rvert \Psi_{\mu}^{(1)}\rangle\rvert^{2}
  }{
    E^{(0)} - E_{\mu}^{(0)}
  },
\qquad
E_{\mathrm{NEVPT2}}
  = E_{\mathrm{CASSCF}} + E_{\mathrm{corr}}^{\mathrm{SC\text{-}NEVPT2}}.
\]

Here \(E_{\mathrm{corr}}\) is the PT2 correction **vs CASSCF** (PySCF
`mrpt.NEVPT.e_corr`), not vs HF.

**Full-valence identity (this foothold).** With CAS(2e,2o) on H₂/STO-3G
there are 2 AO / 2 MO and no core or virtual orbitals outside the CAS:

\[
n_{\mathrm{core}} = 0,\quad n_{\mathrm{cas}} = 2,\quad n_{\mathrm{virt}} = 0.
\]

Every SC class needs at least one core or external index, so each class
contribution is zero by construction and
\(E_{\mathrm{corr}} = 0\), \(E_{\mathrm{NEVPT2}} = E_{\mathrm{CASSCF}}\).
Matches PySCF `mrpt.NEVPT(mc).kernel()` (strongly contracted default) on
the same CASSCF reference.

**Spin treatment.** Closed-shell RHF reference only; active electron count
`nelecas` must be even. No UHF/ROHF/GHF NEVPT2 path.

**Key approximations / scope of this foothold:**

- **Strongly contracted** only — no partially contracted (PC-NEVPT2) or
  quasidegenerate (QD-NEVPT2) variants.
- **State-specific** CASSCF reference; no state averaging / multi-state
  NEVPT2.
- Energy only — no NEVPT2 analytic gradient/Hessian, no public RDM export
  of the PT2 correction.
- Canonical MO integrals; no DF/CD/local MRPT.
- H₂/STO-3G: trivial external space → PT2 ≡ 0; gate validates the product
  pipeline and bookkeeping, not dynamic-correlation recovery on a large
  virtual space.

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

const cpu::scf::NEVPT2Result nev =
    rhf.nevpt2(frame, /*ncas=*/2, /*nelecas=*/2);
// nev.e_nevpt2, nev.e_corr, nev.e_casscf [Ha]
// nev.ncas, nev.nelecas, nev.ncore, nev.nvirt, nev.converged
// rhf.energy() = E_RHF [Ha]
// On this system: nev.e_corr == 0; nev.e_nevpt2 == nev.e_casscf
```

`NEVPT2Result` fields (`include/atomiverse/cpu/scf/nevpt2.h`): `e_nevpt2`,
`e_corr`, `e_casscf` [Hartree], `ncas`, `nelecas`, `ncore`, `nvirt`,
`converged`. `nevpt2_energy` is private engine code.
(`internal/nevpt2_pack.h`).

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
  cmake --build build --target reg_nevpt2-h2-sto3g -j$(nproc)
  ./build/regressions/reg_nevpt2-h2-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/nevpt2.h` | Public `NEVPT2Result` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::nevpt2(Frame&, ncas, nelecas)` |
| `include/atomiverse/cpu/scf/internal/nevpt2_pack.h` | Private pack entry |
| `src/cpu/scf/nevpt2.cpp` | SC-NEVPT2 class contributions |
| `src/cpu/scf/mean_field_method.cpp` | `MeanFieldMethod::nevpt2` dispatch |
| `regressions/scf/nevpt2-h2-sto3g/reg_nevpt2-h2-sto3g.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/nevpt2-h2-sto3g/REFERENCE.md`](../../../regressions/scf/nevpt2-h2-sto3g/REFERENCE.md)
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
| NEVPT2 | `pyscf.mrpt.NEVPT(mc).kernel()` (SC default) |
| AO / MO | 2 AO; `ncore=0`, `ncas=2`, `nvirt=0` |
| PySCF | 2.14.0 |
| Pinned | 2026-07-24, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{CASSCF}}\) | −1.137283834488502 Ha | \(10^{-6}\) Ha | Zeroth-order reference |
| \(E_{\mathrm{corr}}\) (SC-NEVPT2 PT2) | **0.0** Ha | **\(10^{-6}\) Ha** | Primary product gate |
| \(E_{\mathrm{tot}}\) (NEVPT2) | **−1.137283834488502 Ha** | **\(10^{-6}\) Ha** | Product gate |
| Bookkeeping | `ncas=2`, `nelecas=2`, `ncore=0`, `nvirt=0`, `converged` | exact | Product gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{CASSCF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |
| NEVPT2 ≡ CASSCF | \(\lvert E_{\mathrm{NEVPT2}} - E_{\mathrm{CASSCF}}\rvert\) | \(10^{-10}\) Ha | Full-valence identity |

`e_corr` is the PT2 correction vs CASSCF (not vs HF). On this system it is
exactly zero by construction of the active space.

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py nevpt2 \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/nevpt2-h2-sto3g/REFERENCE.md`](../../../regressions/scf/nevpt2-h2-sto3g/REFERENCE.md)
(2026-07-24; login node; `OMP_NUM_THREADS=1`; median of ≥5). Campaign
handoff ratio **0.072** (the case `REFERENCE.md`). No
standing comparison table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `nevpt2` (H₂/STO-3G) | 0.1725 | 0.0125 | **0.0723** | ✓ | Startup-dominated foothold; ATV faster than PySCF (\(r \ll 1\)); PT2 is identically zero — not a production-scale MRPT stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method nevpt2 --geom h2 --basis sto-3g --repeat 5 -- \
    ./build/regressions/reg_nevpt2-h2-sto3g
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization
pass required on this case.

## Limitations

- **SC-NEVPT2 only** on closed-shell RHF; no PC-NEVPT2, QD-NEVPT2, or
  multi-state / SA-CASSCF NEVPT2.
- **No CASPT2** product path (no stock PySCF oracle; OpenMolcas / external).
- **Energy only** — no NEVPT2 analytic gradient/Hessian, no public PT2 RDM
  export.
- **Full-valence H₂/STO-3G foothold** has zero external space: \(E_{\mathrm{corr}} = 0\)
  by construction. Does not exercise non-trivial Sr/Si/… class kernels on a
  large virtual space.
- **Dense CASSCF** reference only — no RAS/GAS/DMRG active spaces.
- **CUDA / Kokkos:** ○ (CPU only).
- **Roadmap next axes** (post-hf §4): PC/QD-NEVPT2, CASPT2, SA-CASSCF,
  CASSCF/NEVPT2 analytic gradient — see `docs/roadmap/post-hf.md`.
- **Perf twin** is the same method family on a minimal system; wall times
  are dominated by process startup and a zero-PT2 full-valence case, so
  \(r\) is not a reliable large-CAS / large-virtual-space predictor.

## References

- Offline pin: `regressions/scf/nevpt2-h2-sto3g/REFERENCE.md`
- Regression consumer: `regressions/scf/nevpt2-h2-sto3g/reg_nevpt2-h2-sto3g.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/post-hf.md` (§4.2 Multireference correlation)
- Public API: `include/atomiverse/cpu/scf/nevpt2.h`
- Sibling (CASSCF foothold): `docs/tutorial/methods/casscf-h2-sto3g.md`
- C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu,
  *Introduction of n-electron valence states for multireference perturbation
  theory*, J. Chem. Phys. **114**, 10252 (2001). DOI
  [10.1063/1.1361246](https://doi.org/10.1063/1.1361246)
- C. Angeli, R. Cimiraglia, J.-P. Malrieu, *n-electron valence state
  perturbation theory: A spinless formulation and an efficient implementation
  of the strongly contracted and of the partially contracted variants*,
  J. Chem. Phys. **117**, 9138 (2002). DOI
  [10.1063/1.1515317](https://doi.org/10.1063/1.1515317)
