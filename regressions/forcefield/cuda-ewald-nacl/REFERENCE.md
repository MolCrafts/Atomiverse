# CUDA EWALD rock-salt NaCl

oracle: literature

J. M. Ziman, *Principles of the Theory of Solids*, 2nd ed., Cambridge
University Press (1972), §2.3, gives the NaCl Madelung lattice sum.
The same reference and constant are used by `md/npt_pme_nacl`.

The 64 unit-charge ions alternate sign on a cubic lattice with nearest-neighbor
spacing d = 2.82 Å and box edge 4d. The lattice energy is

    E = −N M k_e / (2d),  M = 1.747564594633182.

The case composes CUDA CoulLong with KSpaceEwald and performs exactly one
VelocityVerlet step with zero initial velocities and forces. Positions do not
move before the force evaluation, so the energy is the input lattice's energy.
The cutoff is 1.9d, strictly below half the periodic box edge.

The original unit-test assertion is preserved: abs(E/E_reference − 1) ≤ 2e-3.
No oracle executable is invoked. Run on Arrhenius; the case is designed to
finish within 10 seconds on a GH200, including CUDA initialization.
