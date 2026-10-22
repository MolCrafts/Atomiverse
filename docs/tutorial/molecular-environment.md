# Molecules and Environment

Status tables: [Multiscale roadmap](../roadmap/multiscale.md).

A molecule near an ion or in a solvent is not the same system as the
isolated molecule. Fields, point charges, QM/MM regions, and dielectric
continua describe different aspects of those surroundings. More detail is
useful only when its parameters and sampling are controlled.

**Learning goals.** Distinguish environmental models, include all relevant
coupling energies, and interpret a continuum electrostatic contribution.

**Before you begin.** [Charges and Multipoles](ch20-hf-properties-environment.md)
and the electronic Hamiltonian in [Hartree–Fock Theory](ch02-rhf.md).

## Fields and explicit charges

For a weak uniform field $\boldsymbol{\mathcal E}$,
$E\simeq E_0-\boldsymbol\mu\cdot\boldsymbol{\mathcal E}
-\tfrac12\boldsymbol{\mathcal E}^T\boldsymbol\alpha\boldsymbol{\mathcal E}$.
Dipole and polarizability are response coefficients. Field differences
need consistent units, origins, and state tracking; periodic fields require
additional boundary treatment.

A fixed external charge contributes to both electronic and nuclear
interactions. Keeping only its electronic attraction gives an incomplete
total energy and force. Nearby unscreened charges can overpolarize a
quantum density, exposing a model limitation rather than an SCF defect.

QM/MM divides the system into quantum and molecular-mechanics regions.
Electrostatic embedding lets the MM field polarize the quantum subsystem;
mutually polarizable embedding adds another response problem. Region
boundaries, link atoms, exclusions, and double counting must be defined.
Partitioning changes the model, which is why this material belongs here
rather than under numerical acceleration.

## A dielectric reaction field

Continuum solvent replaces surrounding molecules by a dielectric outside
a cavity. In linear response, induced charges respond to the solute
potential; gradual charging gives an electrostatic energy with the
characteristic one-half factor. The solute density and reaction field
must be mutually consistent, and nuclear derivatives include cavity and
response changes.

An electrostatic solvent contribution is not automatically a full hydration
free energy. Cavitation, dispersion, thermal sampling, and standard-state
conventions may be absent. A specified dielectric constant alone does not
define a solvent model at every temperature.

## Practice plan

First place a water molecule near a fixed ion, then move the ion away and
test the isolated limit. Check forces on every mobile degree of freedom.
For continuum water, compare gas and solvent models at a common geometry
before comparing relaxed structures. Record radii, cavity discretization,
dielectric, convergence controls, and the precise reported energy.

The original field, solvent, and QM/MM examples are retained below in their
new subject location. Their numerical values remain historical references,
not newly executed benchmarks for this revision.

## External field and point-charge embedding

### Theory

These prescribed electrostatic fields add terms to the one-electron
Hamiltonian, together with the appropriate classical interactions. This
statement concerns the models below, not every possible environmental coupling:

- Point charges: \(h\leftarrow h-\sum_C q_C\langle 1/|\mathbf r-\mathbf R_C|\rangle\),
  plus classical nucleus–charge and charge–charge terms.
- Uniform field: \(h\leftarrow h+\mathbf F\cdot\langle\mathbf r-\mathbf R_0\rangle\),
  \(\mathbf R_0\) = nuclear-charge centre (same as multipoles).

For a fixed charge distribution, the field coupling is
\(-\boldsymbol\mu\cdot\mathbf F\). A self-consistently relaxed density also
polarizes, so its total energy generally contains quadratic and higher
field terms; the linear expression is not an exact finite-field identity.
Test a range of small fields and electronic tolerances. Strong fields and
diffuse bases can destabilize the intended state. The companion snapshot
documents analytic forces for fixed
point charges (`embed_h2o_grad`), uniform field (`field_h2o_grad`), and
C-PCM free energy (`pcm-grad-h2o`).

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.point_charges = {{0.0, 0.0, 3.0, 0.5}, {0.0, 0.0, -3.0, -0.5}};
rhf.external_field = {0.0, 0.0, 1e-4};
rhf.run(frame);
```

### Results

| Check | Observation |
|---|---|
| H₂O STO-3G finite-field \(\mu_z\) vs analytic | agree to \(10^{-5}\) a.u. |
| H₂O STO-3G + ±0.5 \(e\) charges vs PySCF `qmmm.mm_charge` | \(10^{-8}\) Ha after MM–MM bookkeeping |

Regression: `h2o_embedding_field`. PySCF omits MM–MM self-energy by default;
this engine includes it in nuclear repulsion.

---

## Continuum Solvation

### Theory

Dielectric continuum \(\varepsilon\); cavity of scaled Bondi spheres; apparent
surface charges from \(S\mathbf q=-f(\varepsilon)\mathbf V\) with
\(f=(\varepsilon-1)/(\varepsilon+x)\). \(x=0\): C-PCM; \(x=1/2\): COSMO.
IEF-PCM (`SolventModel::IEFPCM` as the `WithSolvent` constructor argument) uses
\(f=(\varepsilon-1)/(\varepsilon+1)\) and the \(K q = R V\) system with the
double-layer operator \(D\).
Only electrostatic \(\Delta G_{\mathrm{el}}=\frac12\mathbf q\cdot\mathbf V\).
No cavitation/dispersion; cavity uses hard bury-delete (not SWIG-smoothed).
Free-energy forces: vacuum HF gradient on the solvent density plus
\(\partial\Delta G_{\mathrm{el}}/\partial R|_P\) (fixed-density rebuild of
the cavity).

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Pople631Gs};
cpu::scf::WithSolvent<cpu::scf::RHF> sol{std::move(rhf), cpu::scf::SolventModel::CPCM,
                                          /*eps=*/78.4};
sol.radii_scale = 1.2;  // Bondi-radius multiplier
sol.x = 0.0;            // 0 = C-PCM, 0.5 = COSMO
sol.run(frame);
double dg_kcal = sol.e_solvent() * 627.509474;
// forces of G = E_HF + ΔG_el:
auto g = sol.forces(frame); // eV/Å on frame fx/fy/fz
```

Gas phase is the unwrapped method. `eps = 0` is vacuum.

### Results

| System | Quantity | Engine | PySCF | Tol. |
|---|---|---:|---:|---|
| H₂O RHF/6-31G* C-PCM \(\varepsilon=78.4\) | \(\Delta G_{\mathrm{el}}\) / kcal mol⁻¹ | −8.02 | −7.96 | 0.1 |
| H₂O RHF/6-31G* IEF-PCM \(\varepsilon=78.4\) | \(\Delta G_{\mathrm{el}}\) / Ha | −0.012691 | −0.012658 | \(10^{-4}\) Ha |
| H₂O RHF/6-31G* C-PCM | forces vs FD of \(G\) | — | — | \(10^{-5}\) Ha/Bohr |

Regressions: `h2o_rhf_631gs_cpcm`, `ief-pcm-h2o`, `pcm-grad-h2o`.

---

## QM/MM electrostatic embedding

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
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.predictor(3);
qmmm.set_engine(std::move(rhf));
// Optional MC posture: qmmm.reset_history_each_launch() = true;

sim.set_evaluator(cpu::PairLJCut{...}); // MM–MM
sim.set_evaluator(std::move(qmmm));     // QM + cross (+ auto links)
```

See sections `qmmm_es_h2o_pyscf`, `qmmm_link_ch3oh` and `qmmm_md_conservation`
of case `multiscale/qmmm`.

## Further reading

- Barone & Cossi, *J. Phys. Chem. A* **102**, 1995 (1998).
- Jensen, *Introduction to Computational Chemistry*, environmental models.
- [Reference provenance](appendix-pyscf-oracle.md) and the companion field,
  solvent, and embedding regression records should accompany comparisons.

## Questions for reflection

Why does the induced reaction field carry a one-half charging factor? Which additional terms are needed before an electrostatic solvent energy can be compared with an experimental solvation free energy?
