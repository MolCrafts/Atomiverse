# Chapter 19. Large Molecules

The Fock matrix has \(O(N^2)\) elements; four-centre integrals scale as
\(O(N^4)\). Each section below is one remedy, in the fixed order
**Theory → Practice → Results**.

See also [Ch. 2](ch02-rhf.md) and the [Appendix](appendix-pyscf-oracle.md).

> **Build flags.** The SCF sections need `ATV_BUILD_HF=ON` and the
> electrostatics sections (19.3 PME, 19.4 FMM) need `ATV_BUILD_KSPACE=ON`.
> **Both are OFF by default** — with a flag off, that domain's sources,
> headers, tests and regressions are excluded from the build entirely.
>
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release \
>       -DATV_BUILD_HF=ON -DATV_BUILD_KSPACE=ON -DBLA_VENDOR=OpenBLAS
> cmake --build build -j$(nproc)
> ```

---

## 19.1 Direct SCF and Schwarz screening

### Theory

Direct SCF recomputes integrals each iteration. Cauchy–Schwarz screening

\[
|(\mu\nu|\lambda\sigma)|
  \le \sqrt{(\mu\nu|\mu\nu)}\,\sqrt{(\lambda\sigma|\lambda\sigma)}
\]

drops exponentially small shell pairs. Practical cost approaches \(O(N^2)\) at
fixed density; the prefactor of re-evaluation remains.

### Practice

No extra API: the default HF path uses shell-block Schwarz + Rys quadrature
(`ShellBlockPairList`). Keep `ri = RI::None` (default).

### Results

This is the baseline against which RI and CFMM are judged (energy tables in
§19.2 and §19.5).

---

## 19.2 Density Fitting

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
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Def2TZVP;
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

Regression: `regressions/scf/h2o_ri_jk`.

---

## 19.3 Periodic Coulomb: Ewald and PME

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
`include/atomiverse/cpu/longrange/`; the Component wrappers are
`KSpaceEwald` / `KSpacePME`.

### Results

| Check | Value |
|---|---|
| Rock-salt Madelung constant (per ion pair) | \(\approx 1.7476\) (literature) |
| Unit / regression | `tests/cpu/longrange/test_*_coul`, `regressions/md/npt_pme_nacl` |

---

## 19.4 Open-Boundary FMM

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

Regression: `regressions/longrange/fmm_point_charges`.

---

## 19.5 CFMM for the HF \(J\) Matrix

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
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
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

## QM/MM electrostatic coupling (esmm Wave 2)

Same shape as AIMD: one outer `MD` (or `Evaluate` / `Minimize`) and plural
`set_evaluator`. Declare the QM region with **`kAtomQmZ`** (0 = MM, >0 = QM
with atomic number Z). Hand any embedding-capable engine to
`QmmmElectrostatic` via **`set_engine`**. Covalent **link cuts** come from
mask × Frame bond topology (no `set_links`; no bonds ⇒ zero links). MM–MM
uses ordinary pair styles; the coupling owns QM–MM cross terms and third-law
site forces.

```cpp
frame.blocks.at("atoms").insert("qm_z",
    cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone, kDynamicFieldId});
auto qm = frame.span(cpu::kAtomQmZ);
qm[0] = 8; qm[1] = 1; qm[2] = 1; // water QM
// optional: frame.add_bond(...) across QM–MM for link H

cpu::QmmmElectrostatic qmmm{{{2, 2, 0.0065, 3.15}}, 10.0, 12.0};
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.predictor(3);
qmmm.set_engine(std::move(rhf));
// Optional MC posture: qmmm.reset_history_each_launch() = true;

sim.set_evaluator(cpu::PairLJCut{...}); // MM–MM
sim.set_evaluator(std::move(qmmm));     // QM + cross (+ auto links)
```

See `reg_qmmm_es_h2o_pyscf`, `reg_qmmm_link_ch3oh`, `reg_qmmm_md_conservation`.
