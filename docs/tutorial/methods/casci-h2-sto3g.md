# CASCI — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `casci-h2-sto3g` |
| Regression | `regressions/scf/casci-h2-sto3g/` |
| Public API | `cpu::scf::RHF::casci(Frame&, ncas, nelecas)` → `CASCIResult` |
| Roadmap | [Post-HF §4.1 Active-space methods](../../roadmap/post-hf.md) |

## Summary

Closed-shell **CASCI** (complete active-space CI) on H₂ / STO-3G after a
converged RHF reference: freeze inactive core, full CI in a chosen active
space of spatial MOs. Product surface is
`RHF::casci(frame, /*ncas=*/2, /*nelecas=*/2)`: the call drives RHF, freezes
no core on this system (`ncore=0`), and runs the dense FCI kernel in the
2-orbital / 2-electron valence space. Returns
`CASCIResult{e_casci, e_corr, ncas, nelecas, ncore}` in Hartree. On this
two-electron foothold CASCI(2,2) coincides with FCI / CCSD / CISD in the
basis. Offline oracle pin is PySCF 2.14.0 `mcscf.CASCI` (never live in
ctest). End-to-end gate: `regressions/scf/casci-h2-sto3g` → binary
`reg_casci-h2-sto3g`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.076\) (within 10×;
startup-dominated).

## Method

**Ansatz.** Partition spatial MOs into frozen core, active, and virtual
external. Core is doubly occupied and inactive; the CI expands all
determinants with a fixed number of active electrons in the active
orbitals (no orbital optimization):

\[
\lvert\Psi_{\mathrm{CASCI}}\rangle
  = \sum_{I \in \mathrm{CAS}} c_I \lvert\Phi_I\rangle,\qquad
E_{\mathrm{CASCI}} = \min_{\mathbf{c}}
  \frac{\mathbf{c}^\dagger H \mathbf{c}}{\mathbf{c}^\dagger\mathbf{c}},
\quad
E_{\mathrm{corr}} = E_{\mathrm{CASCI}} - E_{\mathrm{RHF}}.
\]

Hamiltonian matrix elements use the same Slater–Condon / dense FCI kernel
as `RHF::fci`, restricted to the active space with a frozen-core energy
shift. Matches PySCF `mcscf.CASCI(mf, ncas, nelecas)` on the same RHF
orbitals (Roos 1987).

**Spin treatment.** Closed-shell RHF reference only; active electron count
`nelecas` must be even (equal α/β in the active space). No UHF/ROHF/GHF
CASCI path.

**Key approximations / scope of this foothold:**

- **CASCI, not CASSCF** — orbitals frozen at RHF; no orbital optimization.
- Active-space FCI is **dense** (full determinant basis); no RAS/GAS/DMRG
  selection.
- Energy only — no CAS 1-RDM/2-RDM export, no analytic gradient/Hessian.
- Canonical MO integrals; no DF/CD/local CAS.
- H₂/STO-3G: 2 AO; full valence CAS(2e,2o); `ncore=0`; CASCI = FCI/CCSD
  to \(\sim 10^{-14}\) Ha in the basis.

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

const cpu::scf::CASCIResult cas = rhf.casci(frame, /*ncas=*/2, /*nelecas=*/2);
// cas.e_casci, cas.e_corr [Ha]; cas.ncas, cas.nelecas, cas.ncore
// rhf.energy() = E_RHF [Ha] after the internal RHF
```

`CASCIResult` fields (`include/atomiverse/cpu/scf/casci.h`): `e_casci`,
`e_corr` [Hartree], `ncas`, `nelecas`, `ncore`. Pack-level
`casci_energy` is private engine code.

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
  cmake --build build --target reg_casci-h2-sto3g -j$(nproc)
  ./build/regressions/reg_casci-h2-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/casci.h` | Public `CASCIResult` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::casci(Frame&, ncas, nelecas)` |
| `include/atomiverse/cpu/scf/internal/casci_pack.h` | Private pack entry |
| `src/cpu/scf/fci.cpp` | Dense FCI / CASCI energy kernel |
| `src/cpu/scf/mean_field_method.cpp` | `MeanFieldMethod::casci` dispatch |
| `regressions/scf/casci-h2-sto3g/reg_casci-h2-sto3g.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/casci-h2-sto3g/REFERENCE.md`](../../../regressions/scf/casci-h2-sto3g/REFERENCE.md)
(offline pin; PySCF never in ctest). Protocol:
`docs/tutorial/appendix-pyscf-oracle.md`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CASCI | `pyscf.mcscf.CASCI(mf, ncas=2, nelecas=2)` |
| AO / MO | 2 AO; 1 occ, 1 virt spatial; `ncore=0` |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CASCI) | **−0.020524527092077 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{tot}}\) (CASCI) | **−1.137283834488502 Ha** | **\(10^{-6}\) Ha** | Primary product gate |
| Bookkeeping | `ncas=2`, `nelecas=2`, `ncore=0` | exact | Product gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |

CASCI(2,2) coincides with FCI / CCSD to \(\sim 10^{-14}\) Ha on this system.

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py casci \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/casci-h2-sto3g/REFERENCE.md`](../../../regressions/scf/casci-h2-sto3g/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥5). Campaign
handoff ratio **0.076** (the case `REFERENCE.md`). No
standing comparison table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `casci` (H₂/STO-3G) | 0.1427 | 0.0109 | **0.076** | ✓ | Startup-dominated foothold; ATV faster than PySCF (\(r \ll 1\)); not a production-scale active-space stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method casci --geom h2 --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_casci-h2-sto3g
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization
pass required on this case.

## Limitations

- **Energy only** on closed-shell RHF; no UHF/ROHF CASCI, no state averaging.
- **No CASSCF** (orbital optimization), no RAS/GAS, no DMRG/selected-CI
  solvers.
- **No MRPT** (CASPT2 / NEVPT2), no CAS gradients/Hessians, no RDM export
  on the public surface.
- **Dense FCI** active-space kernel only — exponential in active size; not
  production for large CAS.
- **CUDA / Kokkos:** ○ (CPU only).
- **Roadmap next axes** (post-hf §4): CASSCF (state-specific / SA),
  CASPT2 / NEVPT2, CASCI analytic gradient — all ○ — see
  `docs/roadmap/post-hf.md`.
- **Perf twin** is the same method family on a minimal system; wall times
  are dominated by process startup and a 2-orbital active space, so \(r\)
  is not a reliable large-CAS predictor.

## References

- Offline pin: `regressions/scf/casci-h2-sto3g/REFERENCE.md`
- Regression consumer: `regressions/scf/casci-h2-sto3g/reg_casci-h2-sto3g.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/post-hf.md` (§4.1 Active-space methods)
- Public API: `include/atomiverse/cpu/scf/casci.h`
- B. O. Roos, *The Complete Active Space Self-Consistent Field Method and
  its Applications in Electronic Structure Calculations*, Adv. Chem. Phys.
  **69**, 399 (1987). DOI
  [10.1002/9780470142943.ch7](https://doi.org/10.1002/9780470142943.ch7)
