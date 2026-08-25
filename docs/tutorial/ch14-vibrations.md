# Chapter 14. Vibrations

Preset `cpu::vib::Hessian` builds a finite-difference Hessian and harmonic
thermochemistry. Bind any force component you trust for Minimize.

```cpp
cpu::vib::Hessian vib;
vib.on(ATV::vib::Phase::force).add(/* Potential or RHF/UHF/RKS/UKS or AM1 */);
vib.build();
cpu::vib::HessianResult H = vib.run(frame);
```

> **Build flag.** The ab initio examples below need `ATV_BUILD_HF=ON`, which is
> **OFF by default** — with it off the `cpu::scf::…` headers are not installed.
> The `Hessian` preset itself is `ATV_BUILD_VIB`, ON by default.
>
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release \
>       -DATV_BUILD_HF=ON -DBLA_VENDOR=OpenBLAS
> cmake --build build -j$(nproc)
> ```

Frequencies are only as good as the underlying forces: validate analytic vs FD
forces ([Chapter 1](ch01-foundations.md)) before trusting thermochemistry. The
in-tree acceptance for this driver is
`regressions/scf/h2o_rhf_freq`, which pins the three H₂O HF/6-31G* harmonic
wavenumbers against PySCF 2.14.0 (see "The Validation" below).

A geometry optimization ([Chapter 12](ch12-geometry-optimization.md)) finds a
stationary point on the potential-energy surface, but it cannot tell you
*what kind* of stationary point it is, nor what the molecule's thermodynamic
properties are at finite temperature. Both answers come from the same object:
the **Hessian**, the matrix of second derivatives of the energy with respect
to the nuclear coordinates. Diagonalizing the mass-weighted Hessian gives the
**vibrational normal modes and frequencies**; feeding those frequencies into
harmonic-oscillator/rigid-rotor statistical mechanics gives the
**thermochemistry** — zero-point energy, enthalpy, entropy, and Gibbs free
energy. This chapter shows how Atomiverse computes both through one driver.

The six questions:

1. **The concept.** Why the Hessian's eigenvalues are frequencies, and what
   they tell you.
2. **The Atomiverse representation.** The `cpu::vib::Hessian` driver and its
   finite-difference machinery.
3. **The C++ API.** `cpu::vib::Hessian`, `HessianResult`, and
   `Thermochem` public members / constructor args.
4. **The expected result.** `3N−6` real frequencies at a minimum, and a
   Gibbs free-energy correction.
5. **The validation.** Frequencies against PySCF — and what is *not* pinned.
6. **The recurring pattern.** Another loop that reuses any force evaluator,
   exactly like MD and minimization.

This chapter assumes the `SCF`-at-the-force-phase pattern from
Tutorials 6 and 8.

## The Concept: Curvature Is Frequency

Near a minimum $\mathbf{R}_0$ of the potential-energy surface, expand the
energy to second order:

$$
E(\mathbf{R}) \approx E(\mathbf{R}_0)
  + \tfrac{1}{2}\, \Delta\mathbf{R}^{\mathsf T}\, \mathbf{H}\, \Delta\mathbf{R},
\qquad
H_{i\alpha,\,j\beta} = \frac{\partial^2 E}{\partial R_{i\alpha}\,\partial R_{j\beta}}.
$$

The linear term vanishes because the gradient is zero at a stationary point,
so the local shape is governed entirely by the Hessian $\mathbf{H}$. Dividing
out the masses (each atom's coordinate scaled by $1/\sqrt{m_i}$) turns the
harmonic problem into an ordinary eigenvalue problem, and its eigenvalues
$\lambda_k$ are squared angular frequencies. Positive $\lambda_k$ gives a real
vibrational frequency $\nu_k \propto \sqrt{\lambda_k}$; a **negative**
eigenvalue gives an *imaginary* frequency, which is the mathematical
signature of a transition state (a first-order saddle, one imaginary mode)
rather than a minimum (zero imaginary modes).

Two facts fall out immediately:

- **Frequencies classify a stationary point.** Count the imaginary
  frequencies: zero ⇒ a genuine minimum; one ⇒ a transition state. This is
  the check every optimized geometry should pass before you trust it.
- **Frequencies unlock thermodynamics.** A molecule's finite-temperature
  free energy is not just its electronic energy — it includes zero-point
  vibration and the temperature-dependent translational, rotational, and
  vibrational contributions to enthalpy and entropy. Harmonic frequencies are
  the input to the standard statistical-mechanics formulas for all of them.

Of the $3N$ Cartesian modes, six (five for a linear molecule) are not
vibrations at all — they are the rigid translations and rotations, which have
zero restoring force. A correct analysis projects those out, leaving $3N-6$
(or $3N-5$) genuine vibrational modes.

## The Representation: A Fixed-Budget Loop Over Finite Differences

Computing an *analytic* Hessian is possible but involves second-derivative
integrals; Atomiverse instead takes the numerically robust route of
**finite-differencing the analytic gradient**. Each nuclear coordinate is
displaced by $\pm h$, the forces are evaluated at both displaced geometries,
and the central difference of the force gives one row of the Hessian. For $N$
atoms that is $6N$ force evaluations plus one reference — a fixed budget with
no convergence loop.

That fixed-budget, no-early-stop structure is what
[`cpu::vib::Hessian`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/vibration/hessian.h)
is: a preset over the CPU loop node (`cpu::Driver<vib::Phase>`), like
`MD` and `Minimize`, but with its own phase enum and **no termination
criterion at all** — the displacement schedule *is* the budget, and there is
nothing to converge. Its phase order —

```text
setup → displace → force → accumulate → diagonalize → thermochem → end_of_step
```

— places the *same* user-registered force evaluator at the `force` phase that
MD and minimization use, and auto-wires the surrounding machinery
(`ReferenceGeometry` caches the equilibrium geometry, `DisplaceStep` applies
the ±h shifts, `HessianAccumulator` builds the rows, `Symmetrize`,
`MassWeighting`, `HessianDiagonalizer`, and the thermochemistry chain). The evaluator
writes the `kAtom_fx/fy/fz` columns; everything else the driver supplies. On return
the frame carries the equilibrium geometry again — the displacements are
transparent to the caller.

## The C++ API: One Driver, One Force Evaluator

The recipe mirrors AIMD and geometry optimization: construct the driver,
register a force-writing evaluator on the `force` phase, `build()`, and
`run()`. The result is a plain `HessianResult` returned by value (Frame purity
means the Hessian is *not* stored on the frame):

```cpp
using namespace ATV;

cpu::Frame frame = /* an OPTIMIZED geometry, positions in Å, type = Z */;

// Tight SCF so each displaced force is gradient-stationary (see below).
// Integrals stay private inside the method.
cpu::scf::RHF rhf{cpu::scf::BasisSet::Pople631G, /*charge=*/0, /*tol=*/1e-12};
rhf.conv_tol_grad = 1e-9;

cpu::vib::Hessian driver;
driver.on(ATV::vib::Phase::force).add(std::move(rhf));
driver.build();

cpu::vib::HessianResult H = driver.run(frame);   // 6N + 1 force evaluations
```

`HessianResult`
([`hessian_result.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/vibration/hessian_result.h))
carries the full analysis:

```cpp
H.hessian;        // [3N, 3N] second derivatives, eV/Å²
H.frequencies;    // wavenumbers [cm⁻¹], length 3N−6 (or 3N−5), ascending;
                  //   imaginary modes reported as NEGATIVE values
H.normal_modes;   // [3N, n_modes] eigenvectors in the Cartesian basis
H.n_imaginary;    // 0 at a minimum, 1 at a transition state
H.is_linear;      // true ⇒ 3N−5 modes
```

Two API details that decide whether your result is meaningful:

- **Start from an optimized geometry.** The harmonic expansion is only valid
  at a stationary point. Run [Chapter 12](ch12-geometry-optimization.md)'s
  minimizer first, at the *same* level of theory you will use for the Hessian.
  A Hessian at a non-stationary geometry has spurious low frequencies from the
  residual gradient.
- **Converge each displaced SCF tightly.** Finite differences amplify noise:
  a loosely converged SCF injects convergence error into every $\pm h$ force,
  and the second difference magnifies it. An `RHF` with `tol = 1e-12` with
  `conv_tol_grad = 1e-9` above is exactly what the parity tests use so each
  displaced force is gradient-stationary. This is the
  [Chapter 1](ch01-foundations.md) finite-difference discipline applied one
  derivative order higher.

### Thermochemistry

Thermochemistry runs automatically as one of the driver's own phases
(`thermochem`); you supply the conditions as plain constructor arguments on
`Hessian` — there is no config bag to fill in:

```cpp
cpu::vib::Hessian driver{
    /*temperature=*/     298.15,    // K
    /*pressure=*/        101325.0,  // Pa
    /*multiplicity=*/    1,         // 2S+1; 1 for a closed-shell singlet
    /*symmetry_number=*/ 2};        // rotational σ, per molecule (H2O: 2)

driver.on(ATV::vib::Phase::force)
      .add(cpu::scf::RHF{cpu::scf::BasisSet::Pople631G});
driver.build();
cpu::vib::HessianResult H = driver.run(frame);

double zpe       = H.zpe;        // harmonic zero-point energy (Hartree)
double g_thermal = H.g_thermal;  // thermal Gibbs correction G(T) − E0 (Hartree)
double s_tot     = H.s_tot;      // total entropy [Hartree/K]
```

The thermochemistry is a faithful port of PySCF's `thermo()`, reporting the
translational / rotational / vibrational contributions to internal energy,
enthalpy, entropy, and Gibbs free energy — plus the electronic contribution
to the *entropy*, $S_\mathrm{elec} = R \ln M$ for spin multiplicity $M$ —
each in Hartree (or Hartree/K), consistent with PySCF. The default
constructor uses standard conditions (298.15 K, 1 atm, closed-shell singlet,
σ = 1); the four-argument overload above lets you set temperature, pressure,
spin multiplicity, and the rotational symmetry number. Note that σ is **supplied**, not auto-detected —
you provide the molecule's point-group rotational symmetry number (2 for
water, 3 for ammonia, 12 for methane).

**The electronic energy $E_0$ is carried as zero** inside the port
([`thermochem.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/src/cpu/vibration/thermochem.cpp)):
`u_tot`, `h_tot` and `g_tot` are the *thermal* parts of $U$, $H$ and $G$, and
`g_tot == g_thermal` as a result. Add your own SCF total energy to obtain an
absolute quantity.

The single most-compared quantity is `g_thermal` — the thermal correction to
the Gibbs free energy, $G(T) - E_0$ — because it is independent of the
electronic energy $E_0$ and therefore isolates the harmonic/statistical part
that this module owns.

### File output: on by default in C++, off in Python

The six-argument constructor
`Hessian(T, P, mult, sigma, frequency_tsv, mode_xyz_prefix)` routes a
tab-separated frequency report and one `<prefix>_mode<k>.xyz` animation per
*real* mode to disk. The other C++ constructors do not leave this off — they
default to `"frequencies.tsv"` and prefix `"mode"`, so a plain
`cpu::vib::Hessian driver;` writes those files into the process's working
directory. An **empty** path (or prefix) is the explicit "write nothing"
spelling that both Recorders honour, which is what the Python binding
(`atomiverse.vibration.Hessian`) passes by default:

```cpp
cpu::vib::Hessian quiet{298.15, 101325.0, 1, 2, /*frequency_tsv=*/"",
                        /*mode_xyz_prefix=*/""};   // no files at all
```

## The Expected Result: Real Frequencies and a Free-Energy Correction

For a well-optimized closed-shell molecule you should see:

- **`n_imaginary == 0`** and exactly `3N−6` real frequencies (`3N−5` if
  `is_linear`). Any imaginary frequency at what you believed was a minimum
  means the geometry was not fully optimized (or you found a saddle).
- **Frequencies in the right physical ranges.** O–H and C–H stretches near
  3000–3800 cm⁻¹, bends near 1000–1700 cm⁻¹, and soft modes below that. For
  water at RHF/6-31G the three modes are the symmetric stretch, the
  antisymmetric stretch, and the bend — the classic three-line spectrum.
  (Harmonic RHF frequencies are systematically ~10% too high versus
  experiment; that overshoot is the *method*, not a bug — a physical
  approximation, not a numerical error.)
- **A sensible thermochemistry.** A positive zero-point energy and a Gibbs
  thermal correction that grows with temperature.

The classes are, as always, distinct: Atomiverse frequencies vs *PySCF at the
same geometry and basis* is a **numerical-tolerance** claim (finite-difference
truncation only); Atomiverse frequencies vs *experiment* is a
**physical-approximation** claim (the harmonic and method errors).

## The Validation: Frequencies vs PySCF

Vibrational analysis is an end-to-end calculation, so its acceptance lives in
`regressions/`, not in `tests/` (the unit suite holds only single-component
checks such as the force-accumulator zeroing). One regression owns the
frequency claim:

- **Frequencies —
  [`regressions/scf/h2o_rhf_freq`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/scf/h2o_rhf_freq/reg_h2o_rhf_freq.cpp).**
  H₂O at HF/6-31G\*, two gates at **1 cm⁻¹**: the full finite-difference
  pipeline (`Hessian::run` over `cpu::scf::RHF`) against the analytic CPHF
  Hessian routed through `Hessian::run_from_cartesian`, and that analytic
  result against pinned PySCF 2.14.0 `hessian.RHF` wavenumbers
  (1843.7084 / 4036.1737 / 4146.3782 cm⁻¹; provenance in its `REFERENCE.md`).
  The logic is pleasingly circular in the useful direction: the numerical
  Hessian is a finite-difference check on the analytic *gradient*, and the
  analytic Hessian is a check on the numerical one.
- **Structure —
  [`regressions/semiempirical/se-analytic-hessian`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/semiempirical/se-analytic-hessian/reg_se-analytic-hessian.cpp).**
  The same driver over DFTB2 forces on H₂: the Hessian is symmetric to
  1e-8 eV/Å², the projected mode count is **exactly** 1 (3N−5 for a linear
  diatomic — a bit-precise property no tolerance applies to), and the stretch
  is real. The absolute wavenumber is not pinned there.

**Thermochemistry has no in-tree oracle pin.** The RRHO port is a faithful
transcription of PySCF `thermo()`, but nothing in the repository currently
compares `g_thermal` or `s_tot` against a PySCF run; the closest live check is
the Python suite's closed-form zero-point energy on an LJ dimer
([`python/tests/test_vib.py`](https://github.com/MolCrafts/Atomiverse/blob/dev/python/tests/test_vib.py),
`0.5 h c ν̃` to 1e-9 Hartree), which validates the constant set rather than the
statistical mechanics. Treat the thermochemistry as unpinned until that
regression exists.

## The Recurring Pattern: Another Loop, Same Evaluator

Vibrational analysis is the clearest proof that Atomiverse's "swap the
evaluator, keep the driver" architecture is not confined to MD. `cpu::vib::
Hessian` is its own loop — its own phase enum, its own fixed
$6N$-displacement schedule, no termination criterion — yet it consumes the
*identical* concrete method (`cpu::scf::RHF{basis}`) that:

- MD ([Chapter 15](ch15-cross-method.md)) plugs into `cpu::md::MD` for AIMD,
- the minimizer ([Chapter 12](ch12-geometry-optimization.md)) plugs into
  `cpu::minimizer::Minimize` for geometry optimization,
- and the hybrid-DFT chapter ([Chapter 6](ch06-hybrid-dft.md)) swaps in
  `cpu::scf::RKS{XCType::B3LYP}` for hybrid forces.

So the natural workflow — optimize, then characterize — is *one method, two
drivers*:

```cpp
// 1. Optimize
cpu::minimizer::Minimize opt;
opt.set_optimizer(cpu::minimizer::LBFGS{/* ... */});
opt.on(minimizer::Phase::force).add(cpu::scf::RHF{basis});
opt.build();  opt.run(frame, 100);

// 2. Characterize at the optimized geometry — same method, same basis
cpu::vib::Hessian vib{298.15, 101325.0, /*multiplicity=*/1, /*sigma=*/2};
vib.on(ATV::vib::Phase::force).add(cpu::scf::RHF{basis});
vib.build();  auto H = vib.run(frame);
```

Because anything that writes the force columns works at the `force` phase, the
very same driver produces *classical* vibrational frequencies if you register a
force-field evaluator (`PairLJCut` + bonded terms) instead. The Hessian
machinery never learns whether its forces are
empirical or quantum — which is exactly the invariant
[Chapter 15](ch15-cross-method.md) elevates to a design principle.

## Validation Reference

| What | Where | Gate |
| --- | --- | --- |
| H₂O HF/6-31G\* wavenumbers, FD pipeline vs analytic CPHF vs PySCF 2.14.0 | [`regressions/scf/h2o_rhf_freq`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/scf/h2o_rhf_freq/reg_h2o_rhf_freq.cpp) | 1 cm⁻¹ on both gates |
| H₂ over DFTB2 forces: symmetry, exact 3N−5 mode count, real stretch | [`regressions/semiempirical/se-analytic-hessian`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/semiempirical/se-analytic-hessian/reg_se-analytic-hessian.cpp) | 1e-8 eV/Å² asymmetry; ω > 100 cm⁻¹ |
| H₂O GFN1/GFN2 FD Hessian of analytic \(G\) vs xtb 6.7.1 `--hess` | [`regressions/semiempirical/gfn-xtb-analytic-hessian`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/semiempirical/gfn-xtb-analytic-hessian/reg_gfn-xtb-analytic-hessian.cpp) | ω pin; see REFERENCE |
| Ar₂ stretch vs the closed-form LJ well curvature, and the ZPE closed form | [`python/tests/test_vib.py`](https://github.com/MolCrafts/Atomiverse/blob/dev/python/tests/test_vib.py) | 1e-3 relative; 1e-9 Hartree |
| Force-accumulator zeroing across the 6N displacement passes (unit) | [`tests/cpu/vibration/test_hessian_zeroing.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/cpu/vibration/test_hessian_zeroing.cpp) | exact |

Pin provenance for the PySCF wavenumbers is recorded in the regression's
`REFERENCE.md`. The RRHO thermochemistry is not pinned against an external
code anywhere in the tree — see the note at the end of "The Validation".
