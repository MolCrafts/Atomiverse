# RKS nuclear Hessian — H₂O / LSDA / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `rks-hessian-h2o-sto3g` |
| Regression | `regressions/scf/rks-hessian-h2o-sto3g/` |
| Public API | `cpu::scf::RKS{XCType::LSDA}.hessian(frame)` |
| Roadmap | `docs/roadmap/dft.md` (RKS analytic Hessian row — product foothold via FD) |

## Summary

Computes the Cartesian nuclear Hessian \(\partial^2 E_{\mathrm{RKS}}/\partial R_i\partial R_j\) for restricted Kohn–Sham DFT by central finite difference of **analytic** RKS nuclear forces (KS gradient with grid-response XC), not a full CPKS closed form. Target system is gas-phase water, LSDA (Slater–VWN5) / STO-3G. Public surface: construct `cpu::scf::RKS` with the XC functional, set members (`basis`, `tol`, `grid_level`, …), call `hessian(frame)` → dense \(3N\times 3N\) matrix in eV/Å² (`cpu::vib::HessianResult`). Offline accuracy pin is PySCF 2.14.0 FD of analytic RKS gradients (same algorithm class as Atomiverse). End-to-end gate: `regressions/scf/rks-hessian-h2o-sto3g` (`reg_rks-hessian-h2o-sto3g`). Fair wall vs the FD twin: \(r = 3.08\) (≤10×).

## Method

**Energy.** Closed-shell RKS with a local density functional (LSDA = Slater exchange + VWN5 correlation). One set of spatial MOs; \(n_\alpha = n_\beta\). XC is integrated on a molecular quadrature grid (`GridLevel::MatchPySCF` for PySCF-parity product gates).

**Forces.** Analytic nuclear gradient from the KS Lagrangian / energy-weighted density path, including semi-local XC and Becke grid-response terms (`RKS::forces`). Public units: eV/Å.

**Hessian (shipped).** Central FD of analytic forces, step \(h = 10^{-4}\) Bohr (displaced in Å via CODATA conversion), then symmetrized:
\[
g = -F,\qquad
H_{ij} \approx \frac{g_i(+h\,e_j) - g_i(-h\,e_j)}{2h},
\]
with \(H\) reported in eV/Å². Fresh RKS SCF per displacement (no DIIS/MO state bleed). For \(N=3\) atoms this is \(2\times 3N = 18\) force evaluations (plus reference SCF restore). Same algorithm class as the offline oracle and as `ROHF::hessian`.

**Not included.**

- Analytic CPKS / PySCF `hessian.RKS` closed form with nuclear XC second derivatives (follow-up; ALDA/response pieces are exercised by TDDFT).
- Mass-weighting, frequencies, IR intensities (those links live on `vib::Hessian*` / RHF analytic Hessian paths).
- Density-fitting RI Hessian (`ri != RI::None` throws; no silent fallback).
- UKS / hybrid-XC Hessian product paths (UKS Hessian still ○ on the DFT roadmap).
- Live PySCF inside ctest (pin is offline only).

## Implementation

Public C++ (Frame + concrete method; no `*Config`, no factories):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

ATV::cpu::Frame frame(3);
// H₂O: O at (0, 0, 0.1173); H at (0, ±0.7572, −0.4692) Å — pin_scf water
// Z = 8,1,1; 20 Å cubic box (gas phase)

ATV::cpu::scf::RKS rks(ATV::cpu::scf::XCType::LSDA);
rks.basis = ATV::cpu::scf::BasisSet::STO3G;
rks.tol = 1e-12;
rks.max_iter = 200;
rks.grid_level = ATV::cpu::scf::GridLevel::MatchPySCF;

const ATV::cpu::vib::HessianResult res = rks.hessian(frame);
// res.hessian: [9,9] dense, eV/Å²; res.n_atoms == 3
// rks.energy() [Ha] after restore of reference geometry
```

Build and run the regression binary:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_rks-hessian-h2o-sto3g -j$(nproc)
  ./build/regressions/reg_rks-hessian-h2o-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/rks.h` | Public `RKS` / `hessian` |
| `src/cpu/scf/rks.cpp` | FD-of-forces implementation |
| `include/atomiverse/cpu/vibration/hessian_result.h` | `HessianResult` contract |
| `regressions/scf/rks-hessian-h2o-sto3g/reg_rks-hessian-h2o-sto3g.cpp` | Product gate |
| `regressions/scf/rks-hessian-h2o-sto3g/REFERENCE.md` | Offline pins |

## Validation (vs PySCF)

Source: [`regressions/scf/rks-hessian-h2o-sto3g/REFERENCE.md`](../../../regressions/scf/rks-hessian-h2o-sto3g/REFERENCE.md)
Oracle class: offline central FD of PySCF **2.14.0** analytic RKS nuclear gradient (`mf.nuc_grad_method().kernel()`), \(h=10^{-4}\) Bohr, symmetrized; units via Atomiverse CODATA factors (`kHartreeInEv = 27.211386245988`, `kBohrInAngstrom = 0.529177210903`). PySCF analytic `mf.Hessian().kernel()` is a **different** algorithm (closed form) and is recorded only as a cross-check (max element abs vs FD twin \(\lesssim 4.34\cdot 10^{-2}\) eV/Å²). Never live-import PySCF in ctest.

**Setup (from REFERENCE):**

| Item | Value |
| ---- | ----- |
| Molecule | H₂O, charge 0, closed shell |
| Geometry [Å] | O (0, 0, 0.1173); H (0, 0.7572, −0.4692); H (0, −0.7572, −0.4692) |
| Box | 20 Å cubic (does not enter molecular SCF) |
| Basis | STO-3G |
| Angular | PySCF pin: `cart=True` (s/p only; cart ≡ spherical for this basis) |
| SCF | RKS, `xc = "LDA,VWN"`, `conv_tol = 1e-12` |
| Grid | ATV `MatchPySCF` (33632 pts) vs PySCF level=3 (33704 pts) |
| \(E_{\mathrm{RKS}}\) (ref, not product gate) | −74.732038853874 Ha |
| Product tol | **2e-2 eV/Å²** on fourteen gated elements |

**Tolerance rationale.** Same algorithm class (FD of analytic forces), but KS grids are not bit-identical. Grid residual appears as \(\sim 2\cdot 10^{-3}\) eV/Å on forces and propagates through \(h=10^{-4}\) Bohr into \(\sim 10^{-2}\) eV/Å² on Hessian elements. Product gate is \(2\cdot 10^{-2}\) (2× that floor) — still catches sign/factor errors on stiff blocks (\(\sim 10^2\) eV/Å²). Full CPKS twin at 1e-5 eV/Å² is follow-up.

**Pinned Cartesian Hessian elements [eV/Å²]** (dense index \(c = 3\cdot\mathrm{atom}+\mathrm{axis}\)):

| Element (row, col) | atoms/axes | Pin [eV/Å²] | Tol |
| ------------------ | ---------- | ----------: | --- |
| \(H_{xx}(0,0)\) | O-x / O-x | −9.632120712096 | 2e-2 |
| \(H_{yy}(1,1)\) | O-y / O-y | +89.527387528595 | 2e-2 |
| \(H_{zz}(2,2)\) | O-z / O-z | +54.829425918251 | 2e-2 |
| \(H_{xx}(0,3)\) | O-x / H1-x | +4.820104284662 | 2e-2 |
| \(H_{xx}(0,6)\) | O-x / H2-x | +4.820104284678 | 2e-2 |
| \(H_{yy}(1,4)\) | O-y / H1-y | −44.758907247881 | 2e-2 |
| \(H_{yz}(1,5)\) | O-y / H1-z | +38.397479554108 | 2e-2 |
| \(H_{yy}(1,7)\) | O-y / H2-y | −44.758906500277 | 2e-2 |
| \(H_{yz}(1,8)\) | O-y / H2-z | −38.397479301244 | 2e-2 |
| \(H_{zz}(2,5)\) | O-z / H1-z | −27.417403943596 | 2e-2 |
| \(H_{yy}(4,4)\) | H1-y / H1-y | +47.406667576594 | 2e-2 |
| \(H_{zz}(5,5)\) | H1-z / H1-z | +26.895192808433 | 2e-2 |
| \(H_{yy}(7,7)\) | H2-y / H2-y | +47.406666805220 | 2e-2 |
| \(H_{zz}(8,8)\) | H2-z / H2-z | +26.895192389626 | 2e-2 |

Regression also asserts Hessian symmetry (\(\lvert H-H^\top\rvert \le 10^{-8}\)) and RKS energy within 1e-6 Ha of the pin after restore (with `rks.converged()`).

## Performance

Fair twin: Atomiverse `RKS::hessian` (18 force evaluations) vs a PySCF loop of 18 RKS analytic gradients (same FD skeleton, \(h=10^{-4}\) Bohr). Analytic `hessian.RKS` is a different algorithm and is **not** the fair wall twin for this product path.

Source: `regressions/scf/rks-hessian-h2o-sto3g/REFERENCE.md` (recorded 2026-07-23, `OMP_NUM_THREADS=8`, Release). Also rolled into the case `REFERENCE.md` / queue notes as ratio **3.08**.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| FD of analytic RKS gradients | 1.019 | 3.138 | **3.08** | ✓ | same algorithm class |
| analytic `Hessian().kernel()` | 0.094 | 3.138 | 33.4 | n/a | unfair — different algorithm |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) on the **FD twin** (satisfied). Comparing to closed-form CPKS is informative only and does not apply the ≤10× gate.

```bash
# Fair FD twin (same work class as RKS::hessian); not a CPKS analytic twin.
python tools/pyscf_oracle/fair_bench.py --method rks -- \
  ./build/regressions/reg_rks-hessian-h2o-sto3g
```

## Limitations

- **Algorithm:** FD of analytic forces, not analytic CPKS; \(O(N_{\mathrm{atoms}})\) SCFs dominate cost.
- **Grid parity:** ATV `MatchPySCF` vs PySCF level=3 leaves ~2e-3 eV/Å force residual → product tol 2e-2 eV/Å² (not 1e-5).
- **PySCF analytic Hessian** exists (`hessian.RKS`) but is a different algorithm class; fair-bench uses the FD twin only.
- **Scope gaps:** no RI Hessian; no frequencies/IR from this path; UKS Hessian not shipped; hybrids not gated here (LSDA only).
- **Roadmap:** full CPKS closed-form RKS Hessian (nuclear XC second-derivative grid response) is the natural follow-up — see `docs/roadmap/dft.md`.
- **Unit tests:** product accuracy lives only under `regressions/`; do not place e2e Hessian solves in `tests/`.

## References

- Offline pin: `regressions/scf/rks-hessian-h2o-sto3g/REFERENCE.md`
- Roadmap: `docs/roadmap/dft.md`
- Related product path: `docs/tutorial/methods/rohf-hessian-o2.md` (ROHF FD Hessian)
- Oracle tooling: `tools/setup_pyscf_env.sh`, `tools/pyscf_oracle/pin_scf.py`
- B. G. Johnson, J. A. Pople, P. M. W. Gill, M. J. Frisch, *Chem. Phys. Lett.* **220**, 377 (1994) — analytic DFT second derivatives (target closed form; this product ships FD of analytic forces).
- PySCF `hessian.rks` / `grad.rks` — external oracle engines.
