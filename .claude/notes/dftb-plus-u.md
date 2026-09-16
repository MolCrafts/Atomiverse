# DFTB+U design (Atomiverse, 2026-07-29)

**Status:** energy + **re-SCC FD forces** on `DFTB2` (`u_eff[Z]`, `last_e_u()`).
Stress / multi-k / ThreeOb / shell-resolved \(U_\ell\) remain out of scope.

## Physics

Dudarev mean-field form on an atomic shell with equal orbital filling
(Hourahine et al. DFTB+; Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998)):

\[
E_U = \sum_A \frac{U_A}{2}\sum_{i\in A} n_i(1-n_i)
\approx \sum_A \frac{U_A}{2}\, n_A\Bigl(1-\frac{n_A}{N_A}\Bigr)
\]

with Mulliken population \(n_A = n_{\mathrm{val},A}-\Delta q_A\) and
\(N_A = 2\,n_{\mathrm{AO},A}\) spin-orbitals on atom \(A\). \(U_A\) is a
per-element parameter [eV] (not the SCC Hubbard \(U\) from SK tables).

## Public API

```cpp
cpu::semiempirical::DFTB2 dftb;
dftb.u_eff[8] = 4.0;       // O effective U [eV]
dftb.with_forces = false;  // required for v1
```

## Forces

When `with_forces` and any `u_eff` is nonzero, forces are **central finite
differences of the total energy** \(E_{\mathrm{SCC}}+E_U\) (full re-SCC at each
displaced geometry). The plain SCC analytic gradient alone is not \(-\nabla
(E_{\mathrm{SCC}}+E_U)\).

Under the Driver/Runtime path (`Evaluate` / `MD`), each component launch has a
Frame resolution budget. The +U FD path parks that window around the multi-SCC
probes (same pattern as `PWEvaluator` force/stress and MeanFieldMethod
accumulator guards) and resolves Frame columns once at the outer launch
boundary — never re-`span` inside the displacement loop.

## Non-goals (v1)

- Shell-resolved \(U_{\ell}\) (s/p/d separate)
- Analytic \(\partial n/\partial R\) (CPHF) forces of \(E_U\)
- Self-consistent \(U\) in the Fock (only post-SCC energy add-on)
- Cell stress with \(U\)
- OpenMOPAC pins (wrong family)

## Oracle

Formula identity + \(U=0\) recovers plain SCC-DFTB2. Absolute DFTB+ pins when
a binary is available offline.

## Unblock

This note removes the “no design note” gate on `dftb-plus-u`. Remaining
product depth (forces, shell-resolved) is ordinary pending work, not blocked.
