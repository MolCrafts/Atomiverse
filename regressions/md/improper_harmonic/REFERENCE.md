# Reference — harmonic improper energy

## Form

\[
V(\chi) = K\,(\chi - \chi_0)^2
\]

with \(K\) [eV/rad²] and equilibrium improper angle \(\chi_0\) [rad]. This is
LAMMPS `improper_style harmonic` (see LAMMPS documentation; functional form also
in CHARMM out-of-plane restraints, MacKerell et al., *J. Phys. Chem. B* **102**,
3586 (1998)).

\(\chi\) is the torsion angle between the two planes of the ordered 4-tuple
(same plane-normal geometry as a proper dihedral).

## Geometry

Four atoms with orthogonal bond vectors (owner = atom 2):

| atom | position [Å] |
|------|----------------|
| 1    | (0, 0, 0)      |
| 2    | (1, 0, 0)      |
| 3    | (1, 1, 0)      |
| 4    | (1, 1, 1)      |

⇒ \(\chi = \pi/2\) exactly.

## Pins

| \(K\) [eV/rad²] | \(\chi_0\) | \(V\) [eV] | why |
|-----------------|------------|------------|-----|
| 0.4             | \(\pi/2\)  | 0          | at equilibrium |
| 0.4             | 0          | \(0.4\,(\pi/2)^2\) | analytic |

Tolerance: \(10^{-10}\) eV.
