# Reference — CHARMM periodic dihedral energy

## Form

\[
V(\phi) = K\bigl(1 + \cos(n\phi - \delta)\bigr)
\]

with \(K\) [eV], multiplicity \(n \ge 1\), phase \(\delta\) [rad]. This is the
standard CHARMM / AMBER torsion term and matches LAMMPS `dihedral_style charmm`
**without** the optional 1–4 nonbonded weight (special-bond scaling stays on
pair styles). See MacKerell et al., *J. Phys. Chem. B* **102**, 3586 (1998)
(CHARMM22 torsion form) and LAMMPS `dihedral_style charmm` documentation.

## Geometry

Four atoms with orthogonal bond vectors:

| atom | position [Å] |
|------|----------------|
| 1    | (0, 0, 0)      |
| 2    | (1, 0, 0)      |
| 3    | (1, 1, 0)      |
| 4    | (1, 1, 1)      |

Planes n1 = b1×b2 and n2 = b2×b3 are orthogonal, so \(\phi = \pi/2\) exactly
(\(\cos\phi = 0\), \(\sin\phi = +1\)).

## Pins

| \(K\) [eV] | \(n\) | \(\delta\) | \(V\) [eV] | why |
|------------|-------|------------|------------|-----|
| 0.5        | 1     | 0          | 0.5        | \(1+\cos(\pi/2)=1\) |
| 0.5        | 2     | 0          | 0.0        | \(1+\cos(\pi)=0\) |
| 0.5        | 1     | \(\pi\)    | 0.5        | \(1+\cos(-\pi/2)=1\) |

Tolerance: \(10^{-10}\) eV (analytic identity, double precision).

Forces are checked by unit FD in `tests/cpu/md/test_pe_fd_force.cpp`, not here.
