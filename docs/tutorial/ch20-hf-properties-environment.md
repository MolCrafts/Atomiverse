# Charges and Multipoles

Status tables: [Multiscale roadmap](../roadmap/multiscale.md).

A converged electronic state contains more than its total energy. Molecular
multipoles characterize its charge distribution, while atomic population
schemes divide that distribution according to a convention. The distinction
matters whenever numbers are compared across programs or interpreted chemically.

**Learning goals.** Evaluate a one-electron observable, test origin and
rotation conventions, and distinguish molecular moments from atomic partitions.

**Before you begin.** Density matrices in [Hartree–Fock Theory](ch02-rhf.md).

## Expectation values and origins

For a one-electron operator with matrix $O$, a consistent spin-summed
density gives electronic expectation $\operatorname{Tr}(PO)$. Nuclear
contributions must be added when the observable includes them. A
two-electron observable generally requires more than this one-particle density.

In atomic units the molecular dipole is

$$\boldsymbol\mu=\sum_A Z_A\mathbf R_A-
\operatorname{Tr}(P\mathbf r).$$

The minus sign accounts for electron charge. If the new origin is shifted
by $\mathbf d$, then $\boldsymbol\mu'=\boldsymbol\mu-Q\mathbf d$.
A neutral molecule has an origin-independent dipole; a charged one does not.
Higher multipoles can depend on lower moments under translation, so a
neutral molecule's quadrupole can still depend on origin. State the tensor
definition, axes, origin, and units before comparing components.

## Population is a choice of partition

Mulliken populations use overlap-weighted AO density; Löwdin populations
use an orthogonalized representation. Real-space partitions and potential
fits answer other questions. Individual atomic charges are not unique
quantum observables. A charge sum can be conserved even when two legitimate
schemes assign different values to an oxygen atom.

Basis and fitting choices can strongly affect these partitions. Numerical
precision in a printed charge is not evidence that an atom-by-atom electron
count has been uniquely measured.

## Practice plan

For neutral water, converge the density and evaluate the actual observer;
constructing an observer alone does not perform a measurement. Translate
the molecule and check dipole invariance, then rotate it and check that its
components rotate while its magnitude is unchanged.

Use a small molecular ion to test the predicted origin shift, and compare
supported population schemes at a common geometry and basis. Keep the
charge sum, molecular dipole, and individual populations distinct in the report.

## Setup fragments

These original fragments and build notes are retained for reference.
They may omit initialization; use the complete model example when running code.

> **Build flag.** Everything in this chapter lives behind `ATV_BUILD_HF`, which
> is **OFF by default** — with it off the `cpu::scf::…` headers, sources, tests
> and regressions are excluded from the build entirely.
>
> ``bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release \
>       -DATV_BUILD_HF=ON -DBLA_VENDOR=OpenBLAS
> cmake --build build -j$(nproc)
> ``

## Multipole moments

### Theory

For a single determinant, \(\langle\hat O\rangle=\operatorname{Tr}(P O)\). Dipole
and traceless quadrupole (Buckingham):

\[
\boldsymbol\mu = \sum_A Z_A \mathbf R_A - \operatorname{Tr}(P\,\mathbf r),\qquad
\Theta_{ab}=\tfrac12\sum q\,(3r_ar_b-r^2\delta_{ab}).
\]

Origin shift \(\mathbf d\): \(\boldsymbol\mu'=\boldsymbol\mu-q_{\mathrm{tot}}\mathbf d\).
Neutral systems are origin-independent; charged systems must report the origin.
This engine uses the **centre of nuclear charge** (`kMultipoleOrigin`).

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Pople631Gs};
rhf.run(frame);

cpu::MultipoleObserver obs{rhf.result_handle()};
// meta: kDipole, kQuadrupole, kMultipoleOrigin
```

### Results

| System | Quantity | Engine | PySCF 2.14.0 | Tol. |
|---|---|---:|---:|---|
| H₂O RHF/6-31G* | \(\mu_z\) / a.u. | −0.87578587 | −0.87578587 | \(10^{-6}\) |

HF overestimates the water dipole vs experiment (~2.23 D vs ~1.85 D)—method
error, not an implementation bug. Regression: `h2o_rhf_dipole`.

---

## Atomic charges

### Theory

Atomic charges are conventions, not observables.

| Method | Idea |
|---|---|
| Mulliken | Split overlap populations |
| Löwdin | Symmetrically orthogonalise, then partition |
| MK / CHELPG | Fit exterior electrostatic potential (ESP) |

Löwdin uses unit-normalised AOs here so Cartesian d/f norms cancel. ESP
fitting is ill-conditioned for buried atoms; the engine falls back to an SVD
minimum-norm solution when needed.

### Practice

```cpp
auto q = cpu::lowdin_charge(rhf.result_handle(), frame);

cpu::MKCharges mk;
auto q_mk = mk.fit(rhf.result_handle(), frame);

cpu::ChelpgCharges chelpg;
auto q_chelpg = chelpg.fit(rhf.result_handle(), frame);
```

### Results

H₂O RHF/6-31G* (oxygen charge / \(e\)):

| Method | Engine | PySCF | Tol. |
|---|---:|---:|---|
| Löwdin | −0.7284 | −0.7284 | \(10^{-6}\) |
| MK | −0.8071 | −0.8071 | \(10^{-3}\) |
| CHELPG | −0.8050 | −0.8050 | \(10^{-3}\) |

Regression: `h2o_charges_pyscf`.

---

## Further reading

- Jensen, *Introduction to Computational Chemistry* (properties, solvation, relativity).
- Szabo & Ostlund, *Modern Quantum Chemistry*, §§3, 7.
- Excited-state references accompany [Electronic Excitations](electronic-excitations.md).
- Solvent references accompany [Molecules and Environment](molecular-environment.md).
- Scalar-relativistic references accompany [Basis Sets](ch04-basis-sets.md).

External pinning rules: [Appendix](appendix-pyscf-oracle.md).

## Questions for reflection

Why can a neutral molecule have an origin-dependent quadrupole? Can two different population schemes both conserve charge while disagreeing on each atom?
