#!/usr/bin/env python3
"""Case-2 MW twin as an n x n supercell.

MW applies the minimum-image convention, so coulomb_rcut <= L/2. In the
primitive 4 A cell that caps rcut at 2 A, while the electrode-electrode short
range term carries erfc(eta_ij r) with eta_ij = 0.86 1/A -- still 1.5e-2 at
r = 2 A. The physical system is unchanged by replicating the cell; the larger
edge lets rcut grow until erfc(eta_ij * rcut) is negligible.
"""
import os
import sys

BOHR = 0.529177210903
ANG = 1.0 / BOHR
HA_EV = 27.211386245988

A_LAT = 2.0
GAP = 5.0
ETA_BOT, ETA_TOP = 1.979, 0.955


def write_case(path, n, lz_A=480.0, rtol=1.0e-15, ktol=1.0e-15):
    os.makedirs(path, exist_ok=True)
    lxy = 4.0 * n
    nsite = 4 * n * n
    L = lxy * ANG
    Lz = lz_A * ANG
    rcut = 0.499 * min(L, Lz)
    e1, e2 = ETA_BOT * BOHR, ETA_TOP * BOHR
    psi2 = 1.0 / HA_EV
    with open(os.path.join(path, "runtime.inpt"), "w") as f:
        f.write(f"""\
# MW twin of ATV regression md/per-electrode-eta case 2, {n}x{n} supercell
# ({lxy} x {lxy} A in plane, {nsite} sites per wall, gap {GAP} A).
# rcut = {rcut * BOHR:.4f} A -> erfc(eta_ij*rcut) < 1e-20.
num_steps    0
timestep     41.341
temperature  0
num_pbc      2

species

  species_type
    name  Null
    count 1
    mass  1.0
    mobile True
    charge neutral

  species_type
    name  Ebot
    count {nsite}
    mass  196.966569
    mobile False
    charge gaussian {e1:.15e} 0.0

  species_type
    name  Etop
    count {nsite}
    mass  196.966569
    mobile False
    charge gaussian {e2:.15e} 0.0

electrodes

  electrode_type
    name      Ebot
    species   Ebot
    potential 0.0

  electrode_type
    name      Etop
    species   Etop
    potential {psi2:.15e}

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
        f.write("# header\nstep                         0\n")
        f.write(f"num_atoms                    {2 * nsite + 1}\n")
        f.write(f"num_electrode_atoms          {2 * nsite}\n")
        f.write("# box\n")
        f.write(f" {L:.15e} {L:.15e} {Lz:.15e}\n")
        f.write("# coordinates\n")
        f.write(f"Null  {1.0 * ANG:.15e} {1.0 * ANG:.15e} {20.0 * ANG:.15e}\n")
        for name, z in (("Ebot", -0.5 * GAP), ("Etop", +0.5 * GAP)):
            for i in range(2 * n):
                for j in range(2 * n):
                    f.write("%s  %.15e %.15e %.15e\n"
                            % (name, A_LAT * i * ANG, A_LAT * j * ANG, z * ANG))
        f.write("# velocities\n")
        f.write("Null  0.0 0.0 0.0\n")
        for name in ("Ebot", "Etop"):
            for _ in range(nsite):
                f.write(f"{name}  0.0 0.0 0.0\n")


if __name__ == "__main__":
    root = sys.argv[1]
    for n in (1, 2, 3, 4, 6):
        write_case(os.path.join(root, f"n{n}"), n)
    print("written")
