# Reference — charged NaCl NPT with PME

## System

Rock-salt (NaCl) simple-cubic lattice of even side length `n` (here `n = 4`,
`N = 64` ions). Site charges alternate as `(-1)^{i+j+k}`; the lattice is
charge-neutral. Nearest-neighbour spacing is `d = 2.82 Å` (the experimental
room-temperature NaCl value). The cubic box edge is `L = n·d`.

## Madelung constant (static energy oracle)

The Coulomb energy of an infinite rock-salt lattice of unit charges is

```text
E = −½ N M k_e / d
```

with the Madelung constant for the NaCl structure

```text
M = 1.747564594633182
```

(Ziman, *Principles of the Theory of Solids*, 2nd ed., Cambridge 1972, §2.3;
also tabulated in many solid-state texts). Here `k_e = units::coulomb` is
Atomiverse’s Coulomb prefactor in engineering units (eV·Å/e²).

The factor `½` converts the per-ion Madelung sum to a total energy over `N`
ions. The numerical value is literature, not a golden dump of Atomiverse’s own
output.

## Uniform scaling law

Pure Coulomb energy is homogeneous of degree `−1` in all lengths. Scaling
every coordinate and the cell edge by `s` multiplies the total electrostatic
energy by `1/s`:

```text
E(s) = E(1) / s
```

This is the cache-invalidation oracle: a KSpace component that reused build-cell
influence data at a new volume would fail the `1/s` check.

## Ewald / PME split (Essmann et al.)

Real-space `CoulLong` and reciprocal `KSpacePME` share one splitting parameter
`α` [1/Å]. The smooth PME reciprocal energy is

```text
E_recip = k_e/(2π V) · Σ_{m≠0} exp(−π²|m|²/α²)/|m|² · B(m) · |Q(m)|²
```

(Essmann et al., *J. Chem. Phys.* **103**, 8577 (1995)). The six-component
reciprocal virial follows the same influence function (see unit gates
`PmeStress.*`). Self and net-charge terms match the Ewald convention used by
`KSpaceEwald`.

## Soft-core repulsion (dynamics only)

Bare point-charge Coulomb has no short-range repulsion, so isotropic NPT at
`P_target = 0` collapses the crystal (Madelung cohesion wins). The dynamics
path therefore adds a soft Lennard-Jones core (`ε = 0.05 eV`, `σ = 1.5 Å`) as
a stand-in for Born–Mayer / Pauli repulsion. At the experimental lattice
spacing `d = 2.82 Å`, `|U_LJ| ≈ 0.004 eV` per neighbour — two orders of
magnitude below the per-ion Madelung energy — so the static Madelung and
`1/s` checks (pure electrostatic Evaluate, no LJ) remain the literature
oracles. The core only becomes important under unphysical close approach.

## NPT dynamics (MTK)

`NoseHooverNPT` implements the Martyna–Tobias–Klein isotropic barostat with
Nosé–Hoover chains (Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177
(1994); Shinoda et al., *Phys. Rev. B* **69**, 134103 (2004)). The conserved
extended Hamiltonian is written to `frame.meta.get(kConserved)` each step. A
short deterministic run must keep the relative drift of that quantity inside
an `O(dt²)` envelope; a secular ramp indicates a force/stress or cell-refresh
bug.

Pressure uses the LAMMPS-standard sign: `P = tr(σ)/(3V)` with positive trace
compressive. The regression sets `P_target = 0` so the barostat balances the
instantaneous (electrostatic + soft-core) virial against the ideal-gas kinetic
term.

## Tolerances (why these numbers)

| Check | Tolerance | Justification |
|---|---|---|
| Madelung `E/E_ref` | `5×10⁻⁴` relative | PME grid spacing `0.5 Å`, order-6 B-splines on a `~11 Å` box; matches the unit-test Madelung gate |
| Scaling `E(s)·s / E(1)` | `5×10⁻⁴` relative | Same discretization budget at both volumes |
| NPT conserved `max|ΔH|/max(1,|H₀|)` | `5×10⁻³` over 400 steps of `dt = 0.5 fs` | MTK is symplectic; Coulomb forces set the step, soft core prevents collapse; fails on a secular ramp or K→0 freeze |
| Cell motion | expanded start → `V₁ < V₀` and `V₁ > 0.2 V₀` | Barostat mutates the box; soft core prevents free collapse |

## Units

Distance Å, energy eV, force eV/Å, mass amu, time fs, temperature K, pressure
eV/Å³ (`1 bar = 6.2415×10⁻⁷ eV/Å³`).
