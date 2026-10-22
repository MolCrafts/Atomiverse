# path/saddle-search

One case; every section below is run by `atv_regressions --case path/saddle-search` and keeps its own oracle record.


## Reference — dimer inverted-harmonic saddle

### Potential

\[
V=\tfrac12 x^2-\tfrac12 y^2
\]

First-order saddle at the origin. Analytic forces \(F=(-x,+y)\).

### Method

Henkelman–Jónsson dimer (J. Chem. Phys. **111**, 7010 (1999),
DOI 10.1063/1.480097): finite-difference rotation of mode \(N\), climb with
\(F^\dagger=F-2(F\cdot N)N\).

### Pass

- `converged() == true`
- distance of single atom to origin \(r < 0.1\) Å


## Reference — IRC on Müller–Brown

- Fukui, *Acc. Chem. Res.* **14**, 363 (1981)
- Müller–Brown saddle (−0.822, 0.624), E ≈ −40.665
- Minima energies ≈ −146.7 and −108.0

Gate: both branches end with E < −50.
