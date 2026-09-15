# UKS nuclear Hessian — O₂ / LSDA / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `uks-hessian-o2-sto3g` |
| Regression | `regressions/scf/uks-hessian-o2-sto3g/` |
| Public API | `cpu::scf::UKS{9,7,XCType::LSDA}.hessian(frame, &guess)` |
| Roadmap | `docs/roadmap/dft.md` (UKS nuclear Hessian — product foothold via FD) |

## Summary

Computes the Cartesian nuclear Hessian \(\partial^2 E_{\mathrm{UKS}}/\partial R_i\partial R_j\) for unrestricted Kohn–Sham DFT by central finite difference of **analytic** UKS nuclear forces (KS gradient with grid-response XC), not a full UCPKS closed form. Target system is gas-phase triplet O₂ (\(2S=2\), \(n_\alpha=9\), \(n_\beta=7\)), LSDA (Slater–VWN5) / STO-3G. Public surface: construct `cpu::scf::UKS` with spin counts and XC, set members (`basis`, `tol`, `grid_level`, …), call `hessian(frame, &guess)` → dense \(3N\times 3N\) matrix in eV/Å² (`cpu::vib::HessianResult`). Multi-minima O₂ needs a ROKS density warm-start (same path as `uks-tda-o2-lsda`). Offline accuracy pin is PySCF 2.14.0 FD of analytic UKS gradients (same algorithm class as Atomiverse). End-to-end gate: `regressions/scf/uks-hessian-o2-sto3g` (`reg_uks-hessian-o2-sto3g`). Fair wall vs the FD twin: \(r = 2.13\) (≤10×).

## Method

**Energy.** Unrestricted Kohn–Sham with a local density functional (LSDA = Slater exchange + VWN5 correlation). Separate \(\alpha\)/\(\beta\) spatial MOs and densities; \(n_\alpha \neq n_\beta\) for open shells. XC is integrated on a molecular quadrature grid (`GridLevel::MatchPySCF` for PySCF-parity product gates).

**Forces.** Analytic nuclear gradient from the unrestricted KS Lagrangian / energy-weighted density path, including semi-local XC and Becke grid-response terms (`UKS::forces`). Public units: eV/Å.

**Hessian (shipped).** Central FD of analytic forces, step \(h = 10^{-4}\) Bohr (displaced in Å via CODATA conversion), then symmetrized:
\[
g = -F,\qquad
H_{ij} \approx \frac{g_i(+h\,e_j) - g_i(-h\,e_j)}{2h},
\]
with \(H\) reported in eV/Å². Fresh UKS SCF per displacement (no DIIS/MO state bleed); displaced SCFs warm-start from the reference spin densities captured after the converged reference SCF. For \(N=2\) atoms this is \(2\times 3N = 12\) force evaluations (plus reference SCF). Same algorithm class as the offline oracle and as `RKS::hessian` / `ROHF::hessian`.

**Multi-minima.** Cold UKS (Hcore) on triplet O₂ often lands on a higher local solution. The product regression warm-starts UKS from ROKS \(\alpha\)/\(\beta\) densities so the reference matches the PySCF UKS minimum used for the pin.

**Not included.**

- Analytic UCPKS / PySCF `hessian.UKS` closed form with nuclear XC second derivatives (follow-up; ALDA/response pieces are exercised by UKS TDA/TDDFT).
- Mass-weighting, frequencies, IR intensities (those links live on `vib::Hessian*` / RHF analytic Hessian paths).
- Density-fitting RI Hessian (`ri != RI::None` throws; no silent fallback).
- Hybrid-XC Hessian product gates (LSDA only in this regression).
- Live PySCF inside ctest (pin is offline only).

## Implementation

Public C++ (Frame + concrete method; no `*Config`, no factories). Minimal sequence matching the regression (ROKS warm-start → UKS Hessian):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

ATV::cpu::Frame frame(2);
// O₂: Z=8, r = 1.2075 Å along z; 20 Å cubic box (gas phase)
// ... fill type / mass / x,y,z / box as in the regression ...

ATV::cpu::scf::ROKS roks(/*n_alpha=*/9, /*n_beta=*/7, ATV::cpu::scf::XCType::LSDA);
roks.basis = ATV::cpu::scf::BasisSet::STO3G;
roks.tol = 1e-12;
roks.max_iter = 200;
roks.grid_level = ATV::cpu::scf::GridLevel::MatchPySCF;
roks.run(frame);

using Spin = ATV::cpu::scf::SlaterDeterminant::Spin;
ATV::cpu::scf::SCFInitialGuess guess;
guess.P_alpha = roks.determinant().density(Spin::alpha);
guess.P_beta = roks.determinant().density(Spin::beta);
guess.has_alpha = true;
guess.has_beta = true;

ATV::cpu::scf::UKS uks(/*n_alpha=*/9, /*n_beta=*/7, ATV::cpu::scf::XCType::LSDA);
uks.basis = ATV::cpu::scf::BasisSet::STO3G;
uks.tol = 1e-12;
uks.max_iter = 200;
uks.grid_level = ATV::cpu::scf::GridLevel::MatchPySCF;

const ATV::cpu::vib::HessianResult res = uks.hessian(frame, &guess);
// res.hessian: [6,6] dense, eV/Å²; res.n_atoms == 2
// uks.energy() [Ha] after restore of reference geometry
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
  cmake --build build --target reg_uks-hessian-o2-sto3g -j$(nproc)
  ./build/regressions/reg_uks-hessian-o2-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/uks.h` | Public `UKS` / `hessian` |
| `src/cpu/scf/uks.cpp` | FD-of-forces implementation |
| `include/atomiverse/cpu/vibration/hessian_result.h` | `HessianResult` contract |
| `regressions/scf/uks-hessian-o2-sto3g/reg_uks-hessian-o2-sto3g.cpp` | Product gate |
| `regressions/scf/uks-hessian-o2-sto3g/REFERENCE.md` | Offline pins |

## Validation (vs PySCF)

Source: [`regressions/scf/uks-hessian-o2-sto3g/REFERENCE.md`](../../../regressions/scf/uks-hessian-o2-sto3g/REFERENCE.md)
Oracle class: offline central FD of PySCF **2.14.0** analytic UKS nuclear gradient (`mf.nuc_grad_method().kernel()`), \(h=10^{-4}\) Bohr, symmetrized; units via Atomiverse CODATA factors (`kHartreeInEv = 27.211386245988`, `kBohrInAngstrom = 0.529177210903`). PySCF analytic `hessian.uks.Hessian(mf).kernel()` is a **different** algorithm (closed form) and is recorded only as a cross-check (max element abs vs FD twin \(\lesssim 3.44\cdot 10^{-2}\) eV/Å²; analytic stretch \(H_{zz}(0,0)\approx 128.490\) vs FD \(128.456\) eV/Å²). Never live-import PySCF in ctest.

**Setup (from REFERENCE):**

| Item | Value |
| ---- | ----- |
| Molecule | O₂, charge 0, spin \(2S=2\) (\(n_\alpha=9\), \(n_\beta=7\)) |
| Geometry [Å] | O (0, 0, 0); O (0, 0, 1.2075) |
| Box | 20 Å cubic (does not enter molecular SCF) |
| Basis | STO-3G |
| Angular | PySCF pin: `cart=True` (s/p only; cart ≡ spherical for this basis) |
| SCF | UKS, `xc = "LDA,VWN"`, `conv_tol = 1e-12` |
| Grid | ATV `MatchPySCF` vs PySCF level=3 (not bit-identical) |
| Warm-start | ROKS densities → `SCFInitialGuess` for UKS |
| \(E_{\mathrm{UKS}}\) (ref, not product gate) | −147.193805922283 Ha |
| Product tol | **2e-2 eV/Å²** on nine gated elements; off-axis \(\lvert H_{ij}\rvert < 10^{-2}\) |

**Tolerance rationale.** Same algorithm class (FD of analytic forces), but KS grids are not bit-identical. Grid residual appears as \(\sim 10^{-3}\) eV/Å on forces and propagates through \(h=10^{-4}\) Bohr into \(\sim 10^{-2}\) eV/Å² on Hessian elements — same floor as `rks-hessian-h2o-sto3g`. Product gate is \(2\cdot 10^{-2}\) (2× that floor) — still catches sign/factor errors on stiff stretch blocks (\(\sim 10^2\) eV/Å²). Full UCPKS twin at 1e-5 eV/Å² is follow-up.

**Pinned Cartesian Hessian elements [eV/Å²]** (dense index \(c = 3\cdot\mathrm{atom}+\mathrm{axis}\)):

| Element (row, col) | atoms/axes | Pin [eV/Å²] | Tol |
| ------------------ | ---------- | ----------: | --- |
| \(H_{xx}(0,0)\) | O1-x / O1-x | −6.951874310841 | 2e-2 |
| \(H_{yy}(1,1)\) | O1-y / O1-y | −6.951874310747 | 2e-2 |
| \(H_{zz}(2,2)\) | O1-z / O1-z | +128.455930123774 | 2e-2 |
| \(H_{xx}(0,3)\) | O1-x / O2-x | +6.951874311534 | 2e-2 |
| \(H_{yy}(1,4)\) | O1-y / O2-y | +6.951874311029 | 2e-2 |
| \(H_{zz}(2,5)\) | O1-z / O2-z | −128.455930112985 | 2e-2 |
| \(H_{xx}(3,3)\) | O2-x / O2-x | −6.951874311765 | 2e-2 |
| \(H_{yy}(4,4)\) | O2-y / O2-y | −6.951874311352 | 2e-2 |
| \(H_{zz}(5,5)\) | O2-z / O2-z | +128.455930103923 | 2e-2 |

Off-axis elements are numerically zero on this collinear geometry (\(\lesssim 10^{-8}\) eV/Å² in the PySCF FD twin); the regression asserts \(\lvert H_{ij}\rvert \le 10^{-2}\) eV/Å². Regression also asserts Hessian symmetry (\(\lvert H-H^\top\rvert \le 10^{-8}\)) and UKS energy within 1e-6 Ha of the pin after restore (with `uks.converged()`).

## Performance

Fair twin: Atomiverse `UKS::hessian` (12 force evaluations after ROKS warm-start + reference) vs a PySCF loop of 12 UKS analytic gradients (same FD skeleton, \(h=10^{-4}\) Bohr). Analytic `hessian.UKS` is a different algorithm and is **not** the fair wall twin for this product path.

Source: `regressions/scf/uks-hessian-o2-sto3g/REFERENCE.md` (recorded 2026-07-24, `OMP_NUM_THREADS=8`, `OMP_PROC_BIND=close`, Release; median of 3 runs). Also rolled into the case `REFERENCE.md` / queue notes as ratio **2.13**.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| FD of analytic UKS gradients | 0.894 | 1.904 | **2.13** | ✓ | same algorithm class |
| analytic `hessian.UKS` | 0.177 | 1.904 | 10.8 | n/a | unfair — different algorithm |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) on the **FD twin** (satisfied). Comparing to closed-form UCPKS is informative only and does not apply the ≤10× gate.

```bash
# Fair FD twin (same work class as UKS::hessian); not a UCPKS analytic twin.
python tools/pyscf_oracle/fair_bench.py --method uks -- \
  ./build/regressions/reg_uks-hessian-o2-sto3g
```

## Limitations

- **Algorithm:** FD of analytic forces, not analytic UCPKS; \(O(N_{\mathrm{atoms}})\) SCFs dominate cost.
- **Grid parity:** ATV `MatchPySCF` vs PySCF level=3 leaves ~1e-3 eV/Å force residual → product tol 2e-2 eV/Å² (not 1e-5).
- **PySCF analytic Hessian** exists (`hessian.UKS`) but is a different algorithm class; fair-bench uses the FD twin only.
- **Multi-minima:** cold UKS on O₂ needs ROKS (or equivalent) density warm-start to hit the pin minimum.
- **Scope gaps:** no RI Hessian; no frequencies/IR from this path; hybrids not gated here (LSDA only).
- **Roadmap:** full UCPKS closed-form UKS Hessian (nuclear XC second-derivative grid response) is the natural follow-up — see `docs/roadmap/dft.md`. Related product paths: `rks-hessian-h2o-sto3g`, `rohf-hessian-o2`, `uhf-hessian-o2`.
- **Unit tests:** product accuracy lives only under `regressions/`; do not place e2e Hessian solves in `tests/`.

## References

- Offline pin: `regressions/scf/uks-hessian-o2-sto3g/REFERENCE.md`
- Roadmap: `docs/roadmap/dft.md`
- Related product paths: `docs/tutorial/methods/rks-hessian-h2o-sto3g.md` (RKS FD Hessian), `docs/tutorial/methods/rohf-hessian-o2.md` (ROHF FD Hessian)
- Oracle tooling: `tools/setup_pyscf_env.sh`, `tools/pyscf_oracle/pin_scf.py`
- B. G. Johnson, J. A. Pople, P. M. W. Gill, M. J. Frisch, *Chem. Phys. Lett.* **220**, 377 (1994) — analytic DFT second derivatives (target closed form; this product ships FD of analytic forces).
- Y. Yamaguchi et al., *A New Dimension to Quantum Chemistry* (Oxford, 1994) — unrestricted Hessian framework.
- PySCF `hessian.uks` / `grad.uks` — external oracle engines.
