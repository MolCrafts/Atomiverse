# Restricted Kohn–Sham DFT (RKS)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Restricted Kohn–Sham DFT describes a closed-shell system (\(n_\alpha =
n_\beta\), one set of spatial MOs) with an exchange–correlation functional
integrated on a molecular quadrature grid; hybrids add a fraction of exact
exchange in the Fock build and double hybrids add scaled MP2 on the KS
orbitals. The public method is `cpu::scf::RKS{basis, xc}` (members `tol`,
`max_iter`, `grid_level`, …) with `run(frame)` and analytic `forces(frame)`.
Companion types bind it for the rest: `cpu::scf::MP2<cpu::scf::RKS>{rks}`
(members `os_scale`, `ss_scale`; `run(frame)`) completes a double hybrid,
and `cpu::scf::Hessian<cpu::scf::RKS>{rks}.run(frame)` gives the nuclear
Hessian (`cpu/scf/post_hf.h`, `cpu/scf/response.h`). Functional metadata are
free lookups: `hybrid_exchange_fraction`, `hybrid_pt2_fraction`, `is_gga`,
`rsh_exchange_params`.

All sections run on CPU. Besides the three documented below, the case carries
`b3lyp-fused-jk-h2o` (B3LYP / def2-SVP H₂O through the restricted-hybrid
Fock build) and `cam-b3lyp-h2-sto3g` (H₂ CAM-B3LYP range-separated hybrid),
whose oracle records live in the case `REFERENCE.md`. Background:
[Ch. 5](../ch05-kohn-sham-dft.md), [Ch. 6](../ch06-hybrid-dft.md).

Case: `ctest -R '^scf/rks$'` — sources and oracle record in [`regressions/scf/rks/`](../../../regressions/scf/rks/). One section of the case per `##` heading below: `atv_regressions --case scf/rks --section <name>`.

## Common setup and oracle

- System: gas-phase H₂O, charge 0, closed shell; O (0, 0, 0.1173),
  H (0, 0.7572, −0.4692), H (0, −0.7572, −0.4692) Å (PySCF `pin_scf.py water`);
  20 Å cubic box (does not enter the molecular SCF). Basis STO-3G, Cartesian
  in the pins (`cart=True`; s/p only, cart ≡ spherical for this basis).
- Oracle: offline PySCF **2.14.0** (`docs/tutorial/appendix-pyscf-oracle.md`),
  an external-engine pin; nothing in ctest imports PySCF.
- Build: `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU build environment
  `buildenv-gcc/2026.03-mpich`, see the repository build notes).
- Roadmap: `docs/roadmap/dft.md`.

## blyp-water-sto3g — BLYP GGA, H₂O / STO-3G {#blyp-water-sto3g}

Pure semi-local GGA: B88 exchange + LYP correlation, **no** HF exchange
(\(\alpha_{\mathrm{hf}}=0\)), exposed as `XCType::BLYP`.

### Theory

BLYP is the pure-GGA counterpart of B3LYP: the same semi-local pieces, no
hybrid mixing. It compares hybrid vs pure GGA without changing the
correlation form.

### Practice

```cpp
cpu::scf::RKS rks{cpu::scf::BasisSet::STO3G, cpu::scf::XCType::BLYP};
rks.tol = 1e-10;
rks.max_iter = 100;
rks.run(frame);   // rks.energy() [Ha]; kEnergy [eV] on the Frame
```

The section also checks `hybrid_exchange_fraction(BLYP) == 0` and
`is_gga(BLYP)`.

### Results

#### Accuracy vs PySCF 2.14.0 (`xc='blyp'`)

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{BLYP}}\) | **−75.277447482665** | **1e-6** |

#### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `rks` BLYP | 6.57 s | 14.87 s | **2.26** | ✓ |

## b2plyp-h2o — B2PLYP double hybrid, H₂O / STO-3G {#b2plyp-h2o}

### Summary

Restricted B2PLYP (Grimme 2006) total energy on water / STO-3G: hybrid KS
(\(a_x=0.53\) HF + semi-local B88/LYP) plus scaled canonical RMP2 on the KS
orbitals (\(a_c=0.27\)). The path is `RKS{basis, XCType::B2PLYP}` bound by
`MP2<RKS>` with `os_scale = ss_scale = hybrid_pt2_fraction(B2PLYP)`; KS SCF
alone is not the double-hybrid total. The PySCF pin is composed as
`0.53*HF + 0.47*B88 + 0.73*LYP` + \(a_c\cdot E_{\mathrm{MP2}}\) (hosted libxc
often lacks a named B2PLYP composite). Roadmap: `docs/roadmap/dft.md` §7
Double-Hybrid Functionals. Fair wall ratio
\(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\approx 1.75\) (≤10×).

### Method

**Double-hybrid XC (Grimme B2PLYP).**

\[
E_{\mathrm{xc}}
  = (1-a_x)\,E_x^{\mathrm{B88}}
  + a_x\,E_x^{\mathrm{HF}}
  + (1-a_c)\,E_c^{\mathrm{LYP}}
  + a_c\,E_c^{\mathrm{MP2}},
\qquad a_x = 0.53,\quad a_c = 0.27.
\]

Semi-local absolute scales in the grid evaluator: \(0.47\) B88 + \(0.73\)
LYP. Exact exchange \(a_x\) is injected by the Fock builder
(`hybrid_exchange_fraction`); PT2 is **not** inside `evaluate_xc`.

**Spin / reference.** Correlation is closed-shell canonical RMP2 on the
converged KS orbitals and orbital energies (the composition class of PySCF
`mp.MP2(mf)`). Spin-component scales are set equal,
\(c_{\mathrm{OS}}=c_{\mathrm{SS}}=a_c\), so

\[
E_{\mathrm{corr}} = a_c\,(E_{\mathrm{OS}}+E_{\mathrm{SS}}),\qquad
E_{\mathrm{tot}} = E_{\mathrm{KS}} + E_{\mathrm{corr}}.
\]

**Key approximations / exclusions.**

- Canonical \(O(N^5)\) RMP2 (no RI, no frozen core: `ri=false`, `frozen=0`).
- MatchPySCF quadrature for the hybrid-KS semi-local part.
- No B2PLYP analytic forces, Hessian, TDDFT, or open-shell (UKS/ROKS) double
  hybrid.
- No other double hybrids (B2GP-PLYP, DSD-*, ωB97X-2, SCS-DH) — roadmap rows ○.

### Implementation

```cpp
using ATV::cpu::scf::XCType;
ATV::cpu::scf::RKS rks{ATV::cpu::scf::BasisSet::STO3G, XCType::B2PLYP};
rks.tol = 1e-10;
rks.max_iter = 100;

const double a_c = ATV::cpu::scf::hybrid_pt2_fraction(XCType::B2PLYP); // 0.27
ATV::cpu::scf::MP2<ATV::cpu::scf::RKS> mp_corr{rks};
mp_corr.os_scale = a_c;
mp_corr.ss_scale = a_c;

const ATV::cpu::scf::MP2Result dh = mp_corr.run(frame);
// rks.energy()      [Ha] hybrid KS only
// dh.e_corr_os/ss   [Ha] unscaled channels
// dh.e_corr         [Ha] a_c * (OS+SS)
// dh.e_tot          [Ha] E_KS + a_c * E_MP2
```

Functional metadata (no bag type):

| Helper | B2PLYP value |
| ------ | -----------: |
| `hybrid_exchange_fraction` | 0.53 |
| `hybrid_pt2_fraction` | 0.27 |
| `is_gga` | `true` |

| Path | Role |
| ---- | ---- |
| `rks.h` | `RKS` ctor / `XCType` latch |
| `xc_functional.h` | `XCType::B2PLYP`, hybrid/PT2 fractions, `evaluate_xc` |
| `post_hf.h` | `MP2<Method>`, `os_scale` / `ss_scale` |
| `src/cpu/scf/xc_functional.cpp` | B2PLYP semi-local grid + fraction tables |

### Validation (vs PySCF)

| Item | Value |
| ---- | ----- |
| Oracle SCF | `conv_tol = 1e-12` |
| Atomiverse SCF | `tol = 1e-10`, MatchPySCF grid |
| PySCF XC string | `0.53*HF + 0.47*GGA_X_B88 + 0.73*GGA_C_LYP` |
| Tol | **1×10⁻⁶ Ha** on e_ks, unscaled MP2, PT2, e_tot |

**Pinned energies [Hartree]:**

| Quantity | Pin (PySCF 2.14.0) | Tol | Notes |
| -------- | -----------------: | --- | ----- |
| \(E_{\mathrm{KS}}\) (hybrid only) | −75.198832360592 | 1e-6 Ha | `rks.energy()` after the MP2 run |
| \(E_{\mathrm{MP2}}\) (unscaled) | −0.044726379786 | 1e-6 Ha | `e_corr_os + e_corr_ss` |
| \(a_c\cdot E_{\mathrm{MP2}}\) | −0.012076122542 | 1e-6 Ha | `dh.e_corr` |
| **\(E_{\mathrm{B2PLYP}}\) total** | **−75.210908483134** | **1e-6 Ha** | `dh.e_tot` |

The section also asserts \(E_{\mathrm{tot}}=E_{\mathrm{KS}}+E_{\mathrm{corr}}\),
\(E_{\mathrm{corr}}=a_c\cdot(E_{\mathrm{OS}}+E_{\mathrm{SS}})\), `!dh.ri`,
`dh.frozen==0`, and SCF convergence.

Pin command (offline only):

```bash
python <pyscf pin> b2plyp \
  --geom water --basis sto-3g --conv-tol 1e-12
```

### Performance

Protocol: PySCF twin, median of ≥3 runs, `OMP_NUM_THREADS=1`,
`OMP_PROC_BIND=close`.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | --------------------------------------: | ---- | ------ |
| `b2plyp` H₂O/STO-3G | 0.291 | 0.509 | **1.75** | ✓ | composed hybrid KS + scaled MP2 both sides |

```bash
python <pyscf fair-bench> \
  --method b2plyp --geom water --basis sto-3g --repeat 3 --max-ratio 10 \
  -- <section binary>
```

**Caveats.** (1) The twin is the full double-hybrid class (hybrid SCF + MP2),
not KS-only. (2) On this tiny STO-3G system the wall ratio does not forecast
large-basis \(O(N^5)\) MP2 scaling. (3) Hosted libxc may lack a named
B2PLYP; fair-bench and pin compose the same hybrid string as Atomiverse.

### Limitations

- Energy only: no B2PLYP nuclear forces, Hessian, or response (TDA/TDDFT).
- Restricted closed-shell only; no open-shell double hybrids.
- No RI-B2PLYP, frozen-core B2PLYP, or spin-component-scaled double hybrids
  (SCS-DH).
- Sibling functionals (B2GP-PLYP, DSD-*, PBE0-DH, ωB97X-2) remain roadmap ○
  (`docs/roadmap/dft.md` §7).
- CUDA / Kokkos B2PLYP ○.
- Speed: ~1.75× PySCF here; larger systems are dominated by the RMP2 step
  without RI/CD.
- The oracle depends on the composed XC when libxc omits `B2PLYP` — record
  this when re-pinning.

### References

- Grimme, S. *J. Chem. Phys.* **124**, 034108 (2006), doi:10.1063/1.2148954 (B2PLYP parameters \(a_x=0.53\), \(a_c=0.27\)).
- Becke, A. D. *Phys. Rev. A* **38**, 3098 (1988) (B88); Lee–Yang–Parr *Phys. Rev. B* **37**, 785 (1988) (LYP).

## rks-hessian-h2o-sto3g — RKS nuclear Hessian, H₂O / LSDA / STO-3G {#rks-hessian-h2o-sto3g}

### Summary

Computes the Cartesian nuclear Hessian
\(\partial^2 E_{\mathrm{RKS}}/\partial R_i\partial R_j\) by central finite
difference of **analytic** RKS nuclear forces (KS gradient with grid-response
XC), not a full CPKS closed form, for water with LSDA (Slater–VWN5) / STO-3G.
`Hessian<RKS>::run` returns a dense \(3N\times 3N\) matrix in eV/Å²
(`cpu::vib::HessianResult`). The pin is a PySCF 2.14.0 FD of analytic RKS
gradients (same algorithm class). Fair wall vs the FD twin: \(r = 3.08\)
(≤10×).

### Method

**Energy.** LSDA = Slater exchange + VWN5 correlation, integrated on a
molecular grid (`GridLevel::MatchPySCF` for PySCF-parity gates).

**Forces.** Analytic nuclear gradient from the KS Lagrangian /
energy-weighted density path, including semi-local XC and Becke grid-response
terms (`RKS::forces`). Public units: eV/Å.

**Hessian.** Central FD of analytic forces, step \(h = 10^{-4}\) Bohr
(displaced in Å via CODATA conversion), then symmetrized:
\[
g = -F,\qquad
H_{ij} \approx \frac{g_i(+h\,e_j) - g_i(-h\,e_j)}{2h},
\]
with \(H\) reported in eV/Å². A fresh RKS SCF runs per displacement (no
DIIS/MO state bleed). For \(N=3\) atoms this is \(2\times 3N = 18\) force
evaluations (plus the reference SCF restore) — the same algorithm class as the
offline oracle and as `Hessian<ROHF>`.

**Not included.**

- Analytic CPKS / PySCF `hessian.RKS` closed form with nuclear XC second
  derivatives (ALDA/response pieces are exercised by TDDFT).
- Mass-weighting, frequencies, IR intensities (those live on `vib::Hessian*` /
  RHF analytic Hessian paths).
- Density-fitting RI Hessian (`ri != RI::None` throws; no silent fallback).
- Hybrid-XC Hessian gates (LSDA only). The UKS Hessian is
  [`uks-hessian-o2-sto3g`](scf-uks.md#uks-hessian-o2-sto3g).

### Implementation

```cpp
// H₂O: Z = 8,1,1; 20 Å cubic box (gas phase)
ATV::cpu::scf::RKS rks{ATV::cpu::scf::BasisSet::STO3G, ATV::cpu::scf::XCType::LSDA};
rks.tol = 1e-12;
rks.max_iter = 200;
rks.grid_level = ATV::cpu::scf::GridLevel::MatchPySCF;

const ATV::cpu::vib::HessianResult res =
    ATV::cpu::scf::Hessian<ATV::cpu::scf::RKS>{rks}.run(frame);
// res.hessian: [9,9] dense, eV/Å²; res.n_atoms == 3
// rks.energy() [Ha] at the restored reference geometry
```

| Path | Role |
| ---- | ---- |
| `rks.h`, `response.h` | Public `RKS`, `Hessian<Method>` |
| `src/cpu/scf/rks.cpp` | FD-of-forces implementation |
| `hessian_result.h` | `HessianResult` contract |

### Validation (vs PySCF)

Oracle class: offline central FD of the PySCF **2.14.0** analytic RKS nuclear
gradient (`mf.nuc_grad_method().kernel()`), \(h=10^{-4}\) Bohr, symmetrized;
units via Atomiverse CODATA factors (`kHartreeInEv = 27.211386245988`,
`kBohrInAngstrom = 0.529177210903`). PySCF analytic `mf.Hessian().kernel()`
is a **different** algorithm (closed form) and is recorded only as a
cross-check (max element abs vs the FD twin \(\lesssim 4.34\cdot 10^{-2}\)
eV/Å²).

| Item | Value |
| ---- | ----- |
| SCF | RKS, `xc = "LDA,VWN"`, `conv_tol = 1e-12` |
| Grid | ATV `MatchPySCF` (33632 pts) vs PySCF level=3 (33704 pts) |
| \(E_{\mathrm{RKS}}\) | −74.732038853874 Ha (checked to 1e-6 Ha) |
| Tol | **2e-2 eV/Å²** on fourteen gated elements |

**Tolerance rationale.** Same algorithm class (FD of analytic forces), but KS
grids are not bit-identical. The grid residual appears as
\(\sim 2\cdot 10^{-3}\) eV/Å on forces and propagates through \(h=10^{-4}\)
Bohr into \(\sim 10^{-2}\) eV/Å² on Hessian elements. The gate is
\(2\cdot 10^{-2}\) (2× that floor) and still catches sign/factor errors on
stiff blocks (\(\sim 10^2\) eV/Å²). A full CPKS twin at 1e-5 eV/Å² is open.

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

The section also asserts Hessian symmetry (\(\lvert H-H^\top\rvert \le 10^{-8}\))
and the RKS energy within 1e-6 Ha of the pin after restore (with
`rks.converged()`).

### Performance

Fair twin: Atomiverse `Hessian<RKS>` (18 force evaluations) vs a PySCF loop of
18 RKS analytic gradients (same FD skeleton, \(h=10^{-4}\) Bohr). Analytic
`hessian.RKS` is a different algorithm and is **not** the fair wall twin.
Measured with `OMP_NUM_THREADS=8`, Release.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| FD of analytic RKS gradients | 1.019 | 3.138 | **3.08** | ✓ | same algorithm class |
| analytic `Hessian().kernel()` | 0.094 | 3.138 | 33.4 | n/a | unfair — different algorithm |

Gate: \(r \le 10\) on the **FD twin** (satisfied); the closed-form CPKS
comparison is informative only. Twin command:
`python <pyscf fair-bench> --method rks -- <section binary>`.

### Limitations

- **Algorithm:** FD of analytic forces, not analytic CPKS;
  \(O(N_{\mathrm{atoms}})\) SCFs dominate cost.
- **Grid parity:** ATV `MatchPySCF` vs PySCF level=3 leaves ~2e-3 eV/Å force
  residual → tol 2e-2 eV/Å² (not 1e-5).
- **Scope gaps:** no RI Hessian; no frequencies/IR from this path; hybrids not
  gated (LSDA only).
- **Open:** a full CPKS closed-form RKS Hessian (nuclear XC second-derivative
  grid response) — see `docs/roadmap/dft.md`. Related:
  [`rohf-hessian-o2`](scf-rohf.md#rohf-hessian-o2) (ROHF FD Hessian).

### References

- B. G. Johnson, J. A. Pople, P. M. W. Gill, M. J. Frisch, *Chem. Phys. Lett.*
  **220**, 377 (1994) — analytic DFT second derivatives (target closed form;
  this path ships FD of analytic forces).
- PySCF `hessian.rks` / `grad.rks` — external oracle engines.
