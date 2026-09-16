# Reference — dimer inverted-harmonic saddle

## Potential

\[
V=\tfrac12 x^2-\tfrac12 y^2
\]

First-order saddle at the origin. Analytic forces \(F=(-x,+y)\).

## Method

Henkelman–Jónsson dimer (J. Chem. Phys. **111**, 7010 (1999),
DOI 10.1063/1.480097): finite-difference rotation of mode \(N\), climb with
\(F^\dagger=F-2(F\cdot N)N\).

## Pass

- `converged() == true`
- distance of single atom to origin \(r < 0.1\) Å
