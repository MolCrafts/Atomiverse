# DFTB+U design

**Scope:** Dudarev energy plus re-SCC finite-difference forces on `DFTB2`
(`u_eff[Z]`, `last_e_u()`), MioSketch Γ only. Stress and multi-k throw when any
`u_eff` is nonzero.

## Physics

Dudarev mean-field form with equal orbital filling on each atom (Hourahine et
al., DFTB+; Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998)):

\[
E_U = \sum_A \frac{U_A}{2}\sum_{i\in A} n_i(1-n_i)
\approx \sum_A \frac{U_A}{2}\, n_A\Bigl(1-\frac{n_A}{N_A}\Bigr)
\]

with Mulliken population \(n_A = n_{\mathrm{val},A}-\Delta q_A\) and
\(N_A = 2\,n_{\mathrm{AO},A}\). \(U_A\) is a per-element parameter [eV], not
the SCC Hubbard \(U\) from SK tables.

```cpp
cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch};
dftb.u_eff[8] = 4.0;       // O effective U [eV]
```

## Forces

With `with_forces` and any nonzero `u_eff`, forces are central finite
differences of \(E_{\mathrm{SCC}}+E_U\) with a full re-SCC per displacement
(the plain SCC analytic gradient is not \(-\nabla(E_{\mathrm{SCC}}+E_U)\)).
Under Evaluate/MD the FD path parks the per-launch Frame resolution window
around the probes and resolves columns once at the outer launch — never
re-`span` inside the displacement loop.

## Non-goals

Shell-resolved \(U_\ell\); analytic \(\partial n/\partial R\) (CPHF) forces;
self-consistent \(U\) in the Fock (post-SCC add-on only); cell stress;
OpenMOPAC pins (wrong family).

## Oracle

Formula identity, and \(U=0\) recovers plain SCC-DFTB2; absolute DFTB+ pins
when a binary is available offline.
