# Reference — AM1 water analytic force

## Method

AM1 (Austin Model 1), Dewar et al., *J. Am. Chem. Soc.* **107**, 3902 (1985).

## What is checked

1. **Public API path:** `cpu::Evaluate` + `cpu::semiempirical::AM1Evaluator` with
   `with_gradient(true)` writes a negative bound total energy [eV] and non-zero
   nuclear forces [eV/Å] on a slightly distorted water geometry; net force ≈ 0
   (translational invariance).
2. **Heat of formation:** MOPAC-style
   `ΔHf = (E_tot − Σ eisol)·23.061 + Σ E_atom` [kcal/mol] near the AM1 water
   regime (~−59 kcal/mol; expt. −57.8). See openmopac calc_of_hof.
3. **SCF gradient identity:** NDDO analytic forces (Hellmann–Feynman contraction
   of the fixed SCF density with integral derivatives) match re-SCF central
   finite-difference of the total energy to `5×10⁻³` eV/Å.

This is a force/self-consistency gate, not a thermochemistry pin of ΔHf.

## Geometry

O at origin; H at (1.05, 0, 0) and (−0.26, 1.0, 0) Å (intentionally off
equilibrium so residual forces are measurable).
