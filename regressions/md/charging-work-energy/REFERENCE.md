# Reference — charging work \(-\sum_i \Psi_i Q_i\) (parallel plate)

oracle: literature
oracle_version: MetalWalls JOSS 5(53) 2373 (2020) eq.6; Scalfi et al. PCCP 22 10480 (2020) / arXiv:1911.09351 eq.12

## Physical statement

Under constant applied electrode potentials, the classical potential energy
includes the **charging work** exchanged with the external circuit. MetalWalls
(Marin-Laflèche et al., *JOSS* **5**, 2373 (2020), DOI 10.21105/joss.02373)
writes

\[
V(Q) = V_c(Q) - \sum_{i=1}^{N} \Psi_i Q_i
\tag{JOSS eq. 6}
\]

where \(V_c\) is the Coulomb energy of the charge distribution and the second
term is the work to charge the electrodes. Scalfi et al. (*PCCP* **22**,
10480 (2020)) identify \(q^{\mathrm{T}}\Psi = Q_L\Delta\psi\) as that work and
use \(-\,q^{\mathrm{T}}\Psi\) in the constant-potential Boltzmann weight.

This regression pins **only** the second term (the bookkeeping Component
`cpu::electrodes::ChargingWork`). Coulomb \(V_c\) and the BO charge solve are
other E1 specs.

## Geometry (offline pin)

Symmetric parallel-plate toy with **fixed** charges (proxy for post-const-pot
\(Q\)):

| Region | type | \(N\) | \(Q_i\) [e] | \(\Psi\) [V] |
| ------ | ---: | ----: | ----------: | -----------: |
| Left wall | 1 | 4 | \(+0.10\) | \(+0.50\) |
| Right wall | 2 | 4 | \(-0.10\) | \(-0.50\) |

Positions: left \(z = 0\), right \(z = 10\) Å; cubic box \(L = 20\) Å (unused
by the bookkeeping term).

## Closed-form pin

\[
E_{\mathrm{charge}}
  = -\bigl(4\cdot 0.50\cdot 0.10 + 4\cdot(-0.50)\cdot(-0.10)\bigr)
  = -0.4~\mathrm{eV}.
\]

Scalfi check: \(Q_L = 0.4\,\mathrm{e}\), \(\Delta\psi = 1.0\,\mathrm{V}\),
\(q^{\mathrm{T}}\Psi = Q_L\Delta\psi = 0.4\), \(E = -q^{\mathrm{T}}\Psi = -0.4\) eV.

**Units:** ATV metal — \(Q\) [e], \(\Psi\) [V], energy [eV]
(\(1\,e\cdot 1\,\mathrm{V} = 1\,\mathrm{eV}\)).

## Tolerance

Absolute \(1\times 10^{-12}\) eV (exact rational arithmetic).

## What is asserted

```text
| frame.meta.get(kEnergy) − (−0.4) | ≤ 1e-12 eV
| frame.meta.get(kEnergyChargingWork) − (−0.4) | ≤ 1e-12 eV
```

via public `cpu::Evaluate` + `cpu::electrodes::ChargingWork` only.

## Perf

perf_exempt: true

reason: No MetalWalls/LAMMPS twin binary for this named energy term in the
agent environment; the Component is pure \(O(N_{\mathrm{electrode}})\) arithmetic
with no force evaluation. Fair wall vs an oracle binary is not applicable
until a MetalWalls install is wired under `tools/metalwalls_oracle/`.

ratio_atv_over_oracle: n/a
within_10x: exempt
