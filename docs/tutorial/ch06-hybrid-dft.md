# Hybrid Functionals

Status tables: [DFT roadmap](../roadmap/dft.md).

Semilocal DFT and HF describe exchange differently. A hybrid combines
orbital-dependent exchange with density-functional contributions, changing
both the energy approximation and the self-consistent electronic state.

**Learning goals.** Interpret exchange mixing, recognize property-dependent
accuracy, and validate gradients before using a hybrid force provider.

**Before you begin.** Exchange in [Hartree–Fock Theory](ch02-rhf.md) and
energy functionals in [Density Functional Theory](ch05-kohn-sham-dft.md).

## Mixing exchange descriptions

A simple global hybrid has the form

$$E_{\mathrm{xc}}^{\mathrm{hyb}}
=aE_x^{\mathrm{HF}}+(1-a)E_x^{\mathrm{DFA}}+E_c^{\mathrm{DFA}}.$$

This describes a family, not every hybrid. PBE0 uses one-quarter HF exchange
with PBE components; B3LYP has a different multi-term parameterization.
The orbital-dependent term yields a nonlocal operator, naturally leading
to generalized Kohn–Sham equations. It is not an energy correction applied
after an unchanged semilocal density has been found.

More exact exchange is not uniformly more accurate. Charge localization,
reaction energies, strongly correlated bonds, and optical excitations can
respond differently. A changed orbital gap is not automatically a corrected
optical gap. Hold basis, grid, geometry, and numerical accuracy fixed when
trying to identify the effect of the functional.

## Differentiating the full model

With atom-centered functions the basis moves with the nuclei, giving Pulay
terms in derivatives. Stationarity is important, but does not authorize
dropping all basis and grid contributions. For a coordinate $R_{A\alpha}$,
compare the force with a central difference of freshly converged energies:

$$F_{A\alpha}^{\mathrm{FD}}=-\frac{E(\mathbf R+h\mathbf e_{A\alpha})
-E(\mathbf R-h\mathbf e_{A\alpha})}{2h}.$$

Small $h$ reduces truncation error but amplifies electronic and round-off
noise. A stable interval of displacements is more convincing than a match
at one arbitrarily chosen step.

## Exchange mixing

Pure semi-local functionals (LDA, GGA) describe exchange approximately, from
the local density and its gradient. That approximation carries a
**self-interaction error**: an electron feels a spurious repulsion from its
own charge density, which pure functionals do not fully cancel. Hartree-Fock
exchange, being computed from the orbitals themselves, cancels
self-interaction *exactly* — but HF has its own failing (no correlation, and
it overshoots exchange for delocalized systems).

A **global hybrid** takes the pragmatic middle: replace a fixed fraction
$\alpha_{\text{HF}}$ of the functional's semi-local exchange with exact HF
exchange, and keep the rest semi-local. The KS exchange-correlation energy
becomes

$$
E_{\text{xc}}^{\text{hybrid}}
  = \alpha_{\text{HF}}\, E_{\text{x}}^{\text{HF}}
  + (1 - \alpha_{\text{HF}})\, E_{\text{x}}^{\text{semi-local}}
  + E_{\text{c}}^{\text{semi-local}}.
$$

The two functionals in this chapter fix $\alpha_{\text{HF}}$ at values
established by their authors:

- **B3LYP** — $\alpha_{\text{HF}} = 0.20$, over the mix
  `0.08 Slater-X + 0.72 B88-X + 0.19 VWN_RPA-C + 0.81 LYP-C`
  (libxc `XC_HYB_GGA_XC_B3LYP`).
- **PBE0** — $\alpha_{\text{HF}} = 0.25$, over `0.75 PBE-X + 1.00 PBE-C`
  (libxc `XC_HYB_GGA_XC_PBEH`).

The word "global" means the fraction is a single constant everywhere in
space (as opposed to range-separated hybrids, where it varies with
interelectronic distance — out of scope here).

## Practice plan

Use a slightly distorted water molecule, where nonzero forces reveal more
than symmetry-enforced zeros. Compare two hybrids at the same geometry and
numerical settings. Then check oxygen and hydrogen force components with
finite differences and test the net force for an isolated field-free model.

The code's placeholder frame must be replaced by an explicit initialized
geometry. Record settings instead of treating historical comments about
defaults as guarantees. Attach the evaluator to optimization only after its
energy and force conventions have been checked.

## Matrix construction

Atomiverse selects a functional with the `XCType` enum
([`xc_functional.h`cpp
enum class XCType { LSDA, PBE, B3LYP, PBE0 };
```

`LSDA` and `PBE` are pure; `B3LYP` and `PBE0` are hybrids. The split between
semi-local and exact exchange is handled cleanly: the grid engine's
`evaluate_xc` returns only the **semi-local** part, and the Fock builder adds
the HF-exchange admixture separately, weighted by the functional's fraction:

```cpp
double a = cpu::scf::hybrid_exchange_fraction(cpu::scf::XCType::B3LYP); // 0.20
bool   g = cpu::scf::is_gga(cpu::scf::XCType::B3LYP);                   // true
```

`hybrid_exchange_fraction` returns `0.0` for pure functionals — the signal
"no exchange-matrix build needed" — and the nonzero fraction for hybrids.
`is_gga` is `true` for PBE, B3LYP, and PBE0 (all carry a density-gradient
term), which tells the grid engine it must build basis-function gradients.
The practical consequence is cost: a hybrid pays for *both* a KS grid *and* a
Hartree-Fock exchange (K) matrix build every SCF iteration, so it is
meaningfully more expensive than a pure GGA. That expense is the price of the
self-interaction correction.

## Hybrid energies and forces

Constructing `RKS` with a hybrid `XCType` presets a production-grade
closed-shell configuration — restricted KS, the PySCF-matching
`GridLevel::MatchPySCF` integration grid, MINAO initial guess, `tol = 1e-9`,
DIIS on — the two hybrids differ only in the functional:

```cpp
using namespace ATV;

cpu::Frame frame = /* H2O, positions in Å, type = Z */;

cpu::scf::RKS rks{cpu::scf::BasisSet::Pople631Gs, cpu::scf::XCType::B3LYP};  // or XCType::PBE0
rks.max_iter = 300;

rks.run(frame);                  // eV energy on frame.meta[kEnergy]
double e_ha = rks.energy();      // total energy [Hartree]
bool   ok   = rks.converged();   // loop state lives on the method's driver
```

Either constructor argument sets only the functional, so everything else —
grid, guess, convergence — is the same locked production default. The
hybrids use a GGA grid, so a polarized basis such as
6-31G* (`Pople631Gs`) or def2-SVP is the natural choice; see
[Basis Sets](ch04-basis-sets.md).

Forces come from `forces(frame)` on the same method object, which runs a
fresh SCF and then the analytic gradient. The return type is
[`gradient_result.h`cpp
cpu::scf::GradientResult grad = rks.forces(frame);

// grad.forces : xtensor [n_atoms, 3] in eV/Å   (FORCE = -dE/dR)
// grad.energy : total energy in eV
```

`forces()` is the *unified* analytic gradient for every mean-field
method — underneath, one routine computes the HF Coulomb+exchange
two-electron terms for HF methods and Coulomb + XC-grid contributions for KS,
with the hybrid case adding the exchange gradient weighted by
$\alpha_{\text{HF}}$. The same entry point serves RHF, UHF, RKS, UKS, and both
hybrids; you never pick a method-specific gradient routine. Note the units
and sign: `forces()` returns **forces** (the negative energy gradient) in
eV/Å, consistent with what the frame's `fx/fy/fz` arrays expect, whereas
PySCF's `grad` returns $\partial E/\partial R$ in Hartree/Bohr — convert with
`core/units.h` before comparing.

Putting the pieces together, a complete single-point-energy-and-forces
routine for a hybrid is:

```cpp
auto run_hybrid = [](cpu::Frame& frame, cpu::scf::XCType xc) {
    cpu::scf::RKS rks{cpu::scf::BasisSet::Pople631Gs, xc};
    rks.max_iter = 300;
    return rks.forces(frame);   // runs the SCF, then the analytic gradient
};
```

## Using the force provider

Everything in this chapter produced a `GradientResult` by hand, which is the
right way to *validate* a gradient. In a real workflow you rarely call
`forces()` directly — you bind the **method itself** at the `force` phase.
Nested there, `RKS` runs exactly this "rebuild integrals → run SCF → compute
analytic gradient → write forces into the frame" sequence on every outer
step, and it is functional-agnostic: construct it with `XCType::B3LYP` and it
provides B3LYP forces to whichever driver it is plugged into.

```cpp
// B3LYP geometry optimization: hybrid forces into the Minimize driver
cpu::minimizer::Minimize opt;
opt.set_optimizer(cpu::minimizer::LBFGS{/* memory, alpha0, max_ls */});
opt.add_criterion(cpu::ForceConverged{/* the five tolerances */});
cpu::scf::RKS rks{cpu::scf::BasisSet::Pople631Gs, cpu::scf::XCType::B3LYP};
opt.on(minimizer::Phase::force).add(std::move(rks));
opt.build();
opt.run(frame, /*max_iter=*/100);
```

The same `RKS{basis, XCType::B3LYP}` drops into `cpu::md::MD` for
hybrid-functional AIMD ([Comparing Models](ch15-cross-method.md)) and into `cpu::vib::Hessian`
for hybrid vibrational frequencies ([Molecular Vibrations](ch14-vibrations.md)). One
gradient routine, one method object, three drivers — the "swap the
method, keep the driver" architecture that
[Comparing Models](ch15-cross-method.md) makes explicit. The functional you chose
in this chapter travels with the method wherever forces are needed.

The following numerical material is retained from the source snapshot.
It was not rerun for this editorial revision; match its input and provenance
before using it as a validation target.

## Energy and force reference data

For a closed-shell molecule at a sensible geometry, a hybrid SCF should
converge in a modest number of DIIS iterations (the MINAO guess and
`GridLevel::MatchPySCF` grid are chosen for reliable convergence), and the
analytic forces should reproduce PySCF's `grad.RKS` component by component.
As a concrete anchor, B3LYP/6-31G* on water yields forces on the order of a
few times $10^{-3}$ Ha/Bohr with the correct symmetry (the two O–H forces
mirror across the C₂ axis), and the sum of forces over all atoms vanishes to
$\sim 10^{-8}$ Ha/Bohr — translational invariance, a free structural check
that no net force can push the isolated molecule.

The optimized geometry follows the forces: minimizing the B3LYP/6-31G*
energy of water lands at an O–H bond of ≈ 0.969 Å and an H–O–H angle of
≈ 103.7°, close to experiment and matching a PySCF-optimized reference to
sub-milliångström RMS. This is the pay-off of the self-interaction
correction — pure LDA over-binds and gives too-short bonds; the hybrid gets
the structure right.

## Independent gradient checks

Hybrid gradients are validated two independent ways, and the redundancy is
deliberate:

1. **Against PySCF (`grad.RKS`).** The Atomiverse analytic force is compared,
   component by component, against a genuine PySCF 2.x `nuc_grad_method`
   reference at the same functional / basis / grid, to **≤ 1e-5 Ha/Bohr** on
   H₂O, CH₄, and NH₃. These retained cases provide evidence of agreement
   with an independent engine at the stated settings, not universal certification.
2. **Against a moving-grid finite difference.** Because a shared bug could in
   principle affect *both* Atomiverse and PySCF's analytic derivations, the
   test *also* compares the analytic force against central differences of the
   hybrid energy, with the XC grid regenerated at each displaced geometry —
   again to **≤ 1e-5 Ha/Bohr**. This oracle trusts nothing but the energy
   itself (the pattern from [Systems and Units](ch01-foundations.md)), and you can
   write it from the public surface alone — each displaced energy is just a
   fresh `run(frame)`:

```cpp
const double h_ang = 1e-3 / ATV::units::kAngstromInBohr;   // 1e-3 Bohr in Å
auto e_at = [&](int A, int d, double h) {         // E at a displaced geometry
    cpu::Frame moved = frame;
    /* shift atom A's coordinate d by h [Å] */
    cpu::scf::RKS rks{cpu::scf::BasisSet::Pople631Gs, cpu::scf::XCType::B3LYP};
    rks.run(moved);
    return rks.energy();                          // [Hartree]
};
const double f_fd =                               // force, central difference
    -(e_at(A, d, +h_ang) - e_at(A, d, -h_ang)) / (2 * h_ang);
// compare (after unit conversion) against rks.forces(frame).forces(A, d)
```

The moving-grid FD check matters for hybrids specifically: the XC grid moves
with the nuclei, so the gradient has a **grid-response** term (the
weight-derivative and grid-coordinate contributions) that a naïve fixed-grid
implementation would miss. Agreement between the analytic gradient and a
genuinely moving-grid finite difference is what proves that term is present
and correct.

Underneath, the semi-local part of each hybrid is itself pinned: the native
XC functionals match libxc to **1e-12**, so the only remaining freedom is the
exact-exchange admixture and its gradient, which the two checks above nail
down. Each result carries its class: the analytic-vs-FD agreement is a
**numerical-tolerance** claim (same quantity, two computations); the
optimized-geometry-vs-experiment gap is a **physical-approximation** claim
(the functional, not the code).

## Reference provenance

Hybrid single-point energies are pinned against PySCF in
`b3lyp-fused-jk-h2o`
and, for the range-separated case,
`cam-b3lyp-h2-sto3g`;
the double hybrid is
`b2plyp-h2o`.
The semi-local XC parts are checked against libxc to 1e-12 in
[`tests/cpu/scf/test_xc_vs_libxc.cpp`](https://github.com/MolCrafts/Atomiverse/blob/master/tests/cpu/scf/test_xc_vs_libxc.cpp)
(built only with `ATV_BUILD_XC_ORACLE=ON`; libxc is never a runtime
dependency).

## Questions for reflection

Why is a distorted geometry useful for gradient validation? Why might improving a hybrid total energy leave an excitation energy inaccurate?
