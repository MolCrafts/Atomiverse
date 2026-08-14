# Reference — AM1 water force

## Method

AM1 (Austin Model 1), Dewar et al., *J. Am. Chem. Soc.* **107**, 3902 (1985).

## What is checked

1. **Public API path:** `cpu::Evaluate` + `cpu::semiempirical::AM1Evaluator` with
   `with_gradient(true)` writes a negative bound total energy [eV] and non-zero
   nuclear forces [eV/Å] on a slightly distorted water geometry; net force ≈ 0.
   The net-force gate is a plumbing check, not a physics one: the solver fills
   the last atom's force as minus the sum of the others (translational
   invariance), so a non-zero net would mean the write-back path lost a
   component.
2. **Heat of formation:** MOPAC-style
   `ΔHf = (E_tot − Σ eisol)·23.061 + Σ E_atom` [kcal/mol] near the AM1 water
   regime (~−59 kcal/mol; expt. −57.8). See openmopac calc_of_hof.
3. **SCF gradient identity:** the published forces — a central finite
   difference of the energy at *fixed* converged density, step `1×10⁻⁴` Å —
   match a *re-converged* central finite difference of the total energy
   (step `5×10⁻⁴` Å) to `5×10⁻³` eV/Å. That agreement is the point: at SCF
   convergence the density-response term of `dE/dR` vanishes, and NDDO's
   orthonormal (zero-differential-overlap) basis contributes no
   basis-derivative "Pulay" term, so the fixed-density derivative is the whole
   gradient. Re-solving the SCF at each displaced geometry is the independent
   check of that claim.

This is a force/self-consistency gate, not a thermochemistry pin of ΔHf.

## Geometry

O at origin; H at (1.05, 0, 0) and (−0.26, 1.0, 0) Å (intentionally off
equilibrium so residual forces are measurable).
