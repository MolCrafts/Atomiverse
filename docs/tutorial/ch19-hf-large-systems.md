# Efficient Electrostatics

Status tables: [Hartree–Fock roadmap](../roadmap/hartree-fock.md) · [Force Fields](../roadmap/force-fields.md).

Large calculations become useful when their approximations remain controlled.
Integral screening, density fitting, and long-range algorithms reduce work
in different ways. Their accuracy must be measured against a defined
reference rather than inferred from an advertised scaling exponent.

**Learning goals.** Apply an integral bound, distinguish fitting from a
boundary-condition change, and build an explicit numerical error budget.

**Before you begin.** HF integrals and basis overlap from
[Hartree–Fock Theory](ch02-rhf.md) and [Basis Sets](ch04-basis-sets.md).

## Bounds and auxiliary representations

With $M$ orbital functions there are formally $O(M^4)$ four-index repulsion
integrals. Direct SCF trades stored integrals for recomputation. The Coulomb
inner product and Cauchy–Schwarz inequality give

$$|(\mu\nu|\lambda\sigma)|\le
\sqrt{(\mu\nu|\mu\nu)}\sqrt{(\lambda\sigma|\lambda\sigma)}.$$

A small bound allows screening, but the accumulated energy error also
depends on density weights and how many terms were omitted. Locality,
diffuse functions, geometry, and thresholds determine practical savings.

Density fitting expands orbital products in auxiliary functions and fits
them in a chosen metric. Fitting Coulomb only and fitting both Coulomb and
exchange are different approximations; the auxiliary basis must match the
purpose. Conditioning of the auxiliary metric requires its own numerical checks.

## Long-range boundary problems

Ewald methods split periodic Coulomb interactions into short- and long-range
parts; PME treats the reciprocal contribution on a mesh. Real-space cutoff,
mesh resolution, and interpolation order share an accuracy budget. An
open-boundary fast multipole method instead groups distant sources and
approximates their fields. These are different boundary-value problems.

Continuous FMM for HF acts on basis-product charge distributions, not merely
on point nuclei. Accelerating the Coulomb matrix does not eliminate exchange,
diagonalization, or communication costs. QM/MM belongs to
[Molecules and Environment](molecular-environment.md), because partitioning
changes the physical model rather than just accelerating an unchanged one.

## Direct SCF and Schwarz screening

### Theory

Direct SCF recomputes integrals each iteration. Cauchy–Schwarz screening

\[
|(\mu\nu|\lambda\sigma)|
  \le \sqrt{(\mu\nu|\mu\nu)}\,\sqrt{(\lambda\sigma|\lambda\sigma)}
\]

screens terms whose bounds fall below a selected threshold. Practical
scaling depends on locality, basis diffuseness, geometry, and accuracy;
a quadratic cost is not a general guarantee. Re-evaluation still has a cost.

### Practice

No extra API: the default HF path uses shell-block Schwarz + Rys quadrature
(`ShellBlockPairList`). Keep `ri = RI::None` (default).

### Results

This is the baseline for the density-fitting and continuous-FMM energy
comparisons in this chapter.

---

## Practice plan

Use small water clusters for which direct HF is affordable. Hold geometry
and orbital basis fixed while testing screening and density fitting; compare
energy and supported derivatives at tighter reference settings. Verify force
availability separately from energy availability.

For periodic electrostatics use a neutral ionic cell with matched conventions;
for open FMM use an isolated charge distribution with a direct pair reference.
Increase size at comparable density and record memory and timings over a
stated range. Two small timing points do not establish asymptotic scaling.

## Setup fragments

These original fragments and build notes are retained for reference.
They may omit initialization; use the complete model example when running code.

> **Build flags.** The SCF sections need `ATV_BUILD_HF=ON` and the
> PME and FMM examples need `ATV_BUILD_KSPACE=ON`.
> **Both are OFF by default** — with a flag off, that domain's sources,
> headers, tests and regressions are excluded from the build entirely.
>
> ``bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release \
>       -DATV_BUILD_HF=ON -DATV_BUILD_KSPACE=ON -DBLA_VENDOR=OpenBLAS
> cmake --build build -j$(nproc)
> ``

## Density Fitting

### Theory

Expand product densities in an auxiliary basis of size \(O(N)\) and fit in the
Coulomb metric (Vahtras–Almlöf–Feyereisen):

\[
(\mu\nu|\lambda\sigma)
  \approx \sum_{PQ}
  (\mu\nu|P)\,[V^{-1}]_{PQ}\,(Q|\lambda\sigma).
\]

Energy error is second order in the residual. This engine provides **def2
universal JKFIT** for both `RI::J` and `RI::JK` (exchange needs JKFIT quality).

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Def2TZVP};
rhf.ri = cpu::scf::RI::JK; // None | J | JK
rhf.aux_basis = cpu::scf::AuxBasis::Def2JKFIT;
rhf.tol = 1e-12;
rhf.run(frame);
```

Unsupported combinations (e.g. analytic forces with RI) throw.

### Results

| System | Quantity | Engine | PySCF / exact |
|---|---|---:|---:|
| H₂O def2-TZVP | \(E_{\mathrm{RI\text{-}JK}}\) / Ha | −76.058999347289 | −76.058999347289 (PySCF DF) |
| H₂O def2-TZVP | \(E_{\mathrm{exact}}\) / Ha | −76.059004760577 | −76.059004760577 |
| H₂O def2-TZVP | \(E_{\mathrm{RI\text{-}JK}}-E_{\mathrm{exact}}\) | \(+5.4\,\mu\mathrm{Ha}\) | — |

Regression: `h2o_ri_jk`.

---

## Periodic Coulomb: Ewald and PME

### Theory

Periodic \(\sum q_i q_j/r_{ij}\) is conditionally convergent. Ewald splits
\(1/r\) into short-range \(\mathrm{erfc}(\alpha r)/r\) and long-range
reciprocal space. Real and reciprocal halves must share \(\alpha\).

Direct Ewald is \(\sim O(N^2)\) at fixed precision. SPME (Essmann et al., 1995)
uses B-spline charge assignment + FFT → \(O(N\log N)\).

### Practice

```cpp
const double alpha = 0.3;  // 1/Å — same on both halves
const double cutoff = 8.0; // Å
sim.set_evaluator(cpu::CoulLong{alpha, cutoff});
sim.set_evaluator(cpu::KSpacePME{alpha, /*grid_spacing=*/1.0, /*order=*/6});
```

`set_evaluator` is plural — the real-space and reciprocal halves are two energy
sources, and both go through the same categorical setter. They form a
`LongRangeParticipant` pair, so exactly one matched real/reciprocal pair must be
registered and `alpha` must agree on both sides.

Reciprocal-space solvers and their math live in
`longrange`; the Component wrappers are
`KSpaceEwald` / `KSpacePME`.

### Results

| Check | Value |
|---|---|
| Rock-salt Madelung constant (per ion pair) | \(\approx 1.7476\) (literature) |
| Unit / regression | `tests/cpu/longrange/test_*_coul`, `npt_pme_nacl` |

---

## Open-Boundary FMM

### Theory

Far-field multipole expansion of a charge cluster (Greengard–Rokhlin); error
falls as \((1/c)^p\) with separation ratio \(c\). Adaptive octree: P2M → M2M →
M2L → L2L → L2P + near-field direct sums. Complements PME (periodic vs open
boundary).

### Practice

```cpp
#include "atomiverse/cpu/longrange/fmm_tree.h"

ATV::longrange::FmmTree tree(/*order=*/8, /*max_per_leaf=*/8);
double e = tree.energy(positions, charges); // dimensionless Coulomb sum
```

### Results

| Check | Observation |
|---|---|
| Relative energy error vs \(O(N^2)\) direct | \(\lesssim 10^{-10}\) at \(p=16\) |
| Scaling smoke \(N\to 4N\) | wall-time ratio \(\ll 16\) |

Regression: `fmm_point_charges`.

---

## CFMM for the HF \(J\) Matrix

### Theory

Gaussian product distributions have extent
\(r_{\mathrm{ext}}=\sqrt{2/\zeta}\,\mathrm{erfc}^{-1}(\varepsilon_{\mathrm{thr}})\)
(White–Johnson–Gill–Head-Gordon). Multipoles only if centres are well
separated; otherwise analytic ERIs. CFMM is a third **\(J\)** route (exact /
RI / CFMM). **\(K\) never uses multipoles.**

PySCF has no molecular FMM; CFMM is validated against this engine’s Direct
\(J\).

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.coulomb = cpu::scf::Coulomb::CFMM;
rhf.cfmm_order = 10;
rhf.cfmm_threshold = 1e-9;
rhf.run(frame);
// CFMM + ri = RI::J/JK → throws
```

Default `Coulomb::Direct` is bit-identical to the historical path.

### Results

| System | Quantity | Tolerance | Status |
|---|---|---|---|
| H₂O STO-3G | \(\max\|J_{\mathrm{CFMM}}-J_{\mathrm{Direct}}\|\) | \(\le 10^{-8}\) Ha | unit `test_cfmm_j` |
| C₁₀H₂₂ STO-3G | \(\|E_{\mathrm{CFMM}}-E_{\mathrm{Direct}}\|\) | \(\le 10^{-7}\) Ha | `reg_c10h22_cfmm_j` |

Small molecules at default thresholds may be all near-field (CFMM ≡ Direct
numerically); timing gains appear only for larger systems
(`benchmarks/scf/validation/cfmm_scaling/`).

---

## Further reading

- Jensen, *Introduction to Computational Chemistry* (SCF bottleneck / scaling).
- Weigend, *Phys. Chem. Chem. Phys.* **8**, 1057 (2006); Essmann et al., *J. Chem. Phys.* **103**, 8577 (1995).
- Greengard & Rokhlin, *J. Comput. Phys.* **73**, 325 (1987); White et al., *Chem. Phys. Lett.* **230**, 8 (1994).

## Questions for reflection

Why does bounding each omitted integral not bound the total energy by the same number? What must agree before two electrostatic solvers can be compared fairly?
