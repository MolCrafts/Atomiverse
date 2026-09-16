# thermo-integration-capacitance

oracle: literature
oracle_version: Reed/Scalfi Gaussian electrode linear-response capacitance;
Beutler et al. Chem. Phys. Lett. 222, 529 (1994) soft-core Kirkwood TI
oracle_kind: closed-form linear capacitor + soft-core TI identity

## Physics (map E10)

Soft-core / λ **thermodynamic integration workflow** for electrode capacitance:

1. **Differential C** of an empty const-pot parallel plate: \(C = Q/\Delta\Psi\)
   independent of \(\Psi\) for linear Gaussian BO.
2. **Kirkwood identity** on `PairCoulSoft`: \(\int_0^1 \partial U/\partial\lambda\,d\lambda = U(1)-U(0)\).
3. **Charging free energy** \(A = \tfrac12 C\Psi^2 = \tfrac12 Q\Psi\), consistent
   with `ChargingWork` \(W = -\sum_i \Psi_i Q_i\).

## Assertions

| gate | check |
|------|-------|
| A | C(0.5 V) ≈ C(1 V) within 1%; Q ∝ Ψ |
| B | soft-Coulomb TI identity ≤ 1e-5 eV |
| C | W = −QΨ; A = ½ QΨ |

## Perf

`perf_exempt` — algebraic + fixed-geometry TI, not a wall-clock twin.
