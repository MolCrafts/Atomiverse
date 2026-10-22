# Unrestricted Kohn–Sham DFT (UKS)

Unrestricted Kohn–Sham DFT uses separate \(\alpha\)/\(\beta\) spatial MOs and
densities (\(n_\alpha \neq n_\beta\) for open shells) with an
exchange–correlation functional integrated on a molecular quadrature grid.
The public method is `cpu::scf::UKS{basis, n_alpha, n_beta, xc}` (members
`tol`, `max_iter`, `grid_level`, …) with `run(frame)` and analytic
`forces(frame)`; the nuclear Hessian is the companion
`cpu::scf::Hessian<cpu::scf::UKS>{uks}.run(frame, &guess)`
(`cpu/scf/response.h`).

The case has one section, the CPU nuclear Hessian of triplet O₂ below.
UKS excited states live in [scf-tddft.md](scf-tddft.md). The case needs
`-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU build environment
`buildenv-gcc/2026.03-mpich`, see the repository build notes).

Case: `ctest -R '^scf/uks$'` — sources and oracle record in [`regressions/scf/uks/`](../../../regressions/scf/uks/). One section of the case per `##` heading below: `atv_regressions --case scf/uks --section <name>`.

## uks-hessian-o2-sto3g — UKS nuclear Hessian, O₂ / LSDA / STO-3G {#uks-hessian-o2-sto3g}

### Summary

Computes the Cartesian nuclear Hessian
\(\partial^2 E_{\mathrm{UKS}}/\partial R_i\partial R_j\) by central finite
difference of **analytic** UKS nuclear forces (KS gradient with grid-response
XC), not a full UCPKS closed form. The system is gas-phase triplet O₂
(\(2S=2\), \(n_\alpha=9\), \(n_\beta=7\)), LSDA (Slater–VWN5) / STO-3G; the
result is a dense \(3N\times 3N\) matrix in eV/Å² (`cpu::vib::HessianResult`).
Multi-minima O₂ needs a ROKS density warm-start (the same path as the
`uks-tda-o2-lsda` section of `scf/tddft`). The accuracy pin is a PySCF 2.14.0
FD of analytic UKS gradients (same algorithm class). Fair wall vs the FD
twin: \(r = 2.13\) (≤10×). Roadmap: `docs/roadmap/dft.md` (UKS nuclear
Hessian via FD).

### Method

**Energy.** LSDA = Slater exchange + VWN5 correlation. XC is integrated on a
molecular grid (`GridLevel::MatchPySCF` for PySCF-parity gates).

**Forces.** Analytic nuclear gradient from the unrestricted KS Lagrangian /
energy-weighted density path, including semi-local XC and Becke grid-response
terms (`UKS::forces`). Public units: eV/Å.

**Hessian.** Central FD of analytic forces, step \(h = 10^{-4}\) Bohr
(displaced in Å via CODATA conversion), then symmetrized:
\[
g = -F,\qquad
H_{ij} \approx \frac{g_i(+h\,e_j) - g_i(-h\,e_j)}{2h},
\]
with \(H\) reported in eV/Å². A fresh UKS SCF runs per displacement (no
DIIS/MO state bleed); displaced SCFs warm-start from the reference spin
densities captured after the converged reference SCF. For \(N=2\) atoms this
is \(2\times 3N = 12\) force evaluations (plus the reference SCF) — the same
algorithm class as the offline oracle and as `Hessian<RKS>` / `Hessian<ROHF>`.

**Multi-minima.** Cold UKS (Hcore) on triplet O₂ often lands on a higher
local solution. The section warm-starts UKS from ROKS \(\alpha\)/\(\beta\)
densities so the reference matches the PySCF UKS minimum used for the pin.

**Not included.**

- Analytic UCPKS / PySCF `hessian.UKS` closed form with nuclear XC second
  derivatives (ALDA/response pieces are exercised by UKS TDA/TDDFT).
- Mass-weighting, frequencies, IR intensities (those live on `vib::Hessian*` /
  RHF analytic Hessian paths).
- Density-fitting RI Hessian (`ri != RI::None` throws; no silent fallback).
- Hybrid-XC Hessian gates (LSDA only).

### Implementation

ROKS warm-start → UKS Hessian:

```cpp
// O₂: Z=8, r = 1.2075 Å along z; 20 Å cubic box (gas phase)
ATV::cpu::scf::ROKS roks{ATV::cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7, ATV::cpu::scf::XCType::LSDA};
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

ATV::cpu::scf::UKS uks{ATV::cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7, ATV::cpu::scf::XCType::LSDA};
uks.tol = 1e-12;
uks.max_iter = 200;
uks.grid_level = ATV::cpu::scf::GridLevel::MatchPySCF;

const ATV::cpu::vib::HessianResult res =
    ATV::cpu::scf::Hessian<ATV::cpu::scf::UKS>{uks}.run(frame, &guess);
// res.hessian: [6,6] dense, eV/Å²; res.n_atoms == 2
// uks.energy() [Ha] at the restored reference geometry
```

| Path | Role |
| ---- | ---- |
| `uks.h`, `response.h` | Public `UKS`, `Hessian<Method>` |
| `src/cpu/scf/uks.cpp` | FD-of-forces implementation |
| `hessian_result.h` | `HessianResult` contract |

### Validation (vs PySCF)

Oracle class: offline central FD of the PySCF **2.14.0** analytic UKS nuclear
gradient (`mf.nuc_grad_method().kernel()`), \(h=10^{-4}\) Bohr, symmetrized;
units via Atomiverse CODATA factors (`kHartreeInEv = 27.211386245988`,
`kBohrInAngstrom = 0.529177210903`). PySCF analytic
`hessian.uks.Hessian(mf).kernel()` is a **different** algorithm (closed form)
and is recorded only as a cross-check (max element abs vs the FD twin
\(\lesssim 3.44\cdot 10^{-2}\) eV/Å²; analytic stretch
\(H_{zz}(0,0)\approx 128.490\) vs FD \(128.456\) eV/Å²). Nothing in ctest
imports PySCF.

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
| \(E_{\mathrm{UKS}}\) | −147.193805922283 Ha (checked to 1e-6 Ha) |
| Tol | **2e-2 eV/Å²** on nine gated elements; off-axis \(\lvert H_{ij}\rvert < 10^{-2}\) |

**Tolerance rationale.** Same algorithm class (FD of analytic forces), but KS
grids are not bit-identical. The grid residual appears as \(\sim 10^{-3}\)
eV/Å on forces and propagates through \(h=10^{-4}\) Bohr into
\(\sim 10^{-2}\) eV/Å² on Hessian elements — the same floor as
[`rks-hessian-h2o-sto3g`](scf-rks.md#rks-hessian-h2o-sto3g). The gate is
\(2\cdot 10^{-2}\) (2× that floor) and still catches sign/factor errors on
stiff stretch blocks (\(\sim 10^2\) eV/Å²). A full UCPKS twin at 1e-5 eV/Å²
is open.

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

Off-axis elements are numerically zero on this collinear geometry
(\(\lesssim 10^{-8}\) eV/Å² in the PySCF FD twin); the section asserts
\(\lvert H_{ij}\rvert \le 10^{-2}\) eV/Å², Hessian symmetry
(\(\lvert H-H^\top\rvert \le 10^{-8}\)) and the UKS energy within 1e-6 Ha of
the pin after restore (with `uks.converged()`).

### Performance

Fair twin: Atomiverse `Hessian<UKS>` (12 force evaluations after ROKS
warm-start + reference) vs a PySCF loop of 12 UKS analytic gradients (same FD
skeleton, \(h=10^{-4}\) Bohr). Analytic `hessian.UKS` is a different
algorithm and is **not** the fair wall twin. Measured with
`OMP_NUM_THREADS=8`, `OMP_PROC_BIND=close`, Release; median of 3 runs.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| FD of analytic UKS gradients | 0.894 | 1.904 | **2.13** | ✓ | same algorithm class |
| analytic `hessian.UKS` | 0.177 | 1.904 | 10.8 | n/a | unfair — different algorithm |

Gate: \(r \le 10\) on the **FD twin** (satisfied); the closed-form UCPKS
comparison is informative only. Twin command:
`python <pyscf fair-bench> --method uks -- <section binary>`.

### Limitations

- **Algorithm:** FD of analytic forces, not analytic UCPKS;
  \(O(N_{\mathrm{atoms}})\) SCFs dominate cost.
- **Grid parity:** ATV `MatchPySCF` vs PySCF level=3 leaves ~1e-3 eV/Å force
  residual → tol 2e-2 eV/Å² (not 1e-5).
- **Multi-minima:** cold UKS on O₂ needs a ROKS (or equivalent) density
  warm-start to hit the pin minimum.
- **Scope gaps:** no RI Hessian; no frequencies/IR from this path; hybrids not
  gated (LSDA only).
- **Open:** a full UCPKS closed-form UKS Hessian (nuclear XC second-derivative
  grid response) — see `docs/roadmap/dft.md`. Related FD/analytic Hessians:
  [`rks-hessian-h2o-sto3g`](scf-rks.md#rks-hessian-h2o-sto3g),
  [`rohf-hessian-o2`](scf-rohf.md#rohf-hessian-o2), and the
  `uhf-hessian-o2` section of [scf/uhf](scf-uhf.md).

### References

- B. G. Johnson, J. A. Pople, P. M. W. Gill, M. J. Frisch, *Chem. Phys. Lett.*
  **220**, 377 (1994) — analytic DFT second derivatives (target closed form;
  this path ships FD of analytic forces).
- Y. Yamaguchi et al., *A New Dimension to Quantum Chemistry* (Oxford, 1994) —
  unrestricted Hessian framework.
- PySCF `hessian.uks` / `grad.uks` — external oracle engines.
