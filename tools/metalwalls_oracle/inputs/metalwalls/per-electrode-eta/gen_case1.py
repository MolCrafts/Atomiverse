#!/usr/bin/env python3
"""Generate the MetalWalls twin of ATV regression md/per-electrode-eta case 1.

Two single-site electrodes, per-electrode Gaussian widths, DeltaPsi = 1 V,
separation 4 A, 2D-periodic square cell of edge L (variable, for the
image-free extrapolation).

ATV public units: eV, A, e, V.  MetalWalls input units: Hartree, Bohr, e,
Hartree/e for potentials.
"""
import os
import sys
from math import sqrt

BOHR = 0.529177210903          # A per bohr  (CODATA-2018, ATV core/units.h)
ANG = 1.0 / BOHR               # bohr per A
HA_EV = 27.211386245988        # eV per Hartree

ETA1_A = 1.979                 # 1/A   bottom electrode (type 1, Psi = 0 V)
ETA2_A = 0.955                 # 1/A   top    electrode (type 2, Psi = 1 V)
D_A = 4.0                      # A     wall separation
DPSI_V = 1.0                   # V     applied potential difference

ETA1 = ETA1_A * BOHR           # 1/bohr
ETA2 = ETA2_A * BOHR
D = D_A * ANG                  # bohr
PSI2 = DPSI_V / HA_EV          # Hartree/e


def write_case(path, L_A, Lz_A=None, rtol=1.0e-15, ktol=1.0e-15):
    """One MW case at square in-plane edge L_A [A]."""
    os.makedirs(path, exist_ok=True)
    L = L_A * ANG
    Lz = (Lz_A if Lz_A is not None else L_A) * ANG
    rcut = 0.499 * min(L, Lz)   # MW requires rcut < min(a,b,c)/2
    with open(os.path.join(path, "runtime.inpt"), "w") as f:
        f.write(f"""\
# MetalWalls twin of Atomiverse regression md/per-electrode-eta, case 1.
# Two single-site electrodes at fixed potentials, per-electrode Gaussian
# width eta_e, vacuum (no electrolyte), 2D PBC (xy), separation 4 A.
#
# ATV units -> MW atomic units
#   eta_bot = {ETA1_A} 1/A   = {ETA1:.15e} 1/bohr
#   eta_top = {ETA2_A} 1/A   = {ETA2:.15e} 1/bohr
#   d       = {D_A} A        = {D:.15e} bohr
#   DeltaPsi= {DPSI_V} V     = {PSI2:.15e} Ha/e
#   L_xy    = {L_A} A        = {L:.15e} bohr

num_steps    0
timestep     41.341
temperature  0
num_pbc      2

species

  # Dummy chargeless, non-interacting spectator: MW computes an instantaneous
  # temperature every printed step and divides by the mobile DOF count, which
  # is NaN for an electrode-only cell. One neutral mobile atom at rest makes
  # T = 0 K well-defined and contributes nothing to the electrostatics.
  species_type
    name  Null
    count 1
    mass  1.0
    mobile True
    charge neutral

  species_type
    name  Ebot
    count 1
    mass  196.966569
    mobile False
    charge gaussian {ETA1:.15e} 0.0

  species_type
    name  Etop
    count 1
    mass  196.966569
    mobile False
    charge gaussian {ETA2:.15e} 0.0

electrodes

  electrode_type
    name      Ebot
    species   Ebot
    potential 0.0

  electrode_type
    name      Etop
    species   Etop
    potential {PSI2:.15e}

  electrode_charges matrix_inversion 1.0e-14
  charge_neutrality true electrodes
  compute_force false

interactions
  coulomb
    coulomb_rtol {rtol:.1e}
    coulomb_rcut {rcut:.15e}
    coulomb_ktol {ktol:.1e}

output
  default 0
  step 1
  charges 1
""")
    with open(os.path.join(path, "data.inpt"), "w") as f:
        f.write("# header\n")
        f.write("step                         0\n")
        f.write("num_atoms                    3\n")
        f.write("num_electrode_atoms          2\n")
        f.write("# box\n")
        f.write(f" {L:.15e} {L:.15e} {Lz:.15e}\n")
        f.write("# coordinates\n")
        f.write(f"Null  {0.25 * L:.15e} {0.25 * L:.15e} 0.0\n")
        f.write(f"Ebot  0.0 0.0 {-0.5 * D:.15e}\n")
        f.write(f"Etop  0.0 0.0 {+0.5 * D:.15e}\n")
        f.write("# velocities\n")
        f.write("Null  0.0 0.0 0.0\n")
        f.write("Ebot  0.0 0.0 0.0\n")
        f.write("Etop  0.0 0.0 0.0\n")


if __name__ == "__main__":
    root = sys.argv[1]
    lz_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    for L in (40.0, 60.0, 80.0, 120.0, 160.0, 240.0, 320.0):
        write_case(os.path.join(root, f"L{int(L)}"), L, Lz_A=lz_factor * L)
    print("written")
