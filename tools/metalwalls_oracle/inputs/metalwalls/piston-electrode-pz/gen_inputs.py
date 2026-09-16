#!/usr/bin/env python3
"""Generate the MetalWalls twin of regressions/md/piston-electrode-pz.

The ATV regression builds, in eV / Angstrom / fs:

  box            orthorhombic Lx = Ly = 10 A, Lz = 20 A     (A = Lx*Ly = 100 A^2)
  lower wall     4 atoms, type 2, m = 196.9665 amu, z = 2 A   (FixFreeze)
  upper piston   4 atoms, type 3, m =  12.011  amu, z = 8 A   (PistonElectrode::Upper)
  fluid          16 atoms, type 1, m = 39.948 amu, two layers z = 4.2 / 5.8 A
  pair           LJ eps = 0.10 eV, sigma = 3.0 A, r_c = CUTOFF_A

MetalWalls works in atomic units (Hartree / Bohr / e) and refuses r_c > L/2,
so the twin is emitted in Bohr using *MetalWalls' own* CODATA constants
(src/constants.F90) so that sigma/r ratios and the energy scale round-trip
exactly.

Writes runtime.inpt + data.inpt next to this script and prints the analytic
minimum-image LJ z-force on the upper wall for an independent cross-check.
"""

import math
import pathlib

# ── MetalWalls' own constants (src/constants.F90) ────────────────────────
BOHR2ANG = 0.52917721067  # MW: bohr2angstrom
HARTREE_SI = 4.359744650e-18  # MW: hartree_si  [J]
AVOGADRO = 6.02214076e23  # MW: avogadro_si
KJPERMOL2HARTREE = 1.0e3 / (HARTREE_SI * AVOGADRO)
EV_PER_HARTREE = 27.211386245988  # ATV core/units.h

ANG2BOHR = 1.0 / BOHR2ANG

# ── ATV regression system (see reg_piston-electrode-pz.cpp) ──────────────
LXY_A = 10.0
LZ_A = 20.0
Z_LO_A = 2.0
Z_HI_A = 8.0
EPS_EV = 0.10
SIGMA_A = 3.0
CUTOFF_A = 4.5  # < Lxy/2 = 5 A  (min-image legal; MW-legal)
DT_FS = 0.5
M_FLUID = 39.948
M_LOWER = 196.9665
M_UPPER = 12.011

# Applied cell potential [a.u.].  The ATV regression has no charges at all,
# but MetalWalls' constant-potential CG divides by <d,Ad> and hits 0/0 = NaN
# when the right-hand side is identically zero (src/cg.f90, alpha = dp/dAd).
# A non-zero Delta psi makes the charge solve well posed.  The Lennard-Jones
# force on the electrode is a function of geometry alone, so the
# `f*_lennard_jones` columns of electrode_forces.out are *unchanged* by this
# and are what we compare against ATV.
POT_AU = 0.5

# MW timestep unit: hbar/E_h.  1 fs = 41.341373336493 a.u.
AU_PER_FS = 41.341373336493


def atv_geometry():
    """Exactly mirrors the coordinate loops in reg_piston-electrode-pz.cpp."""
    lower, upper, fluid = [], [], []
    for ix in range(2):
        for iy in range(2):
            lower.append(((ix + 0.5) * (LXY_A / 2.0), (iy + 0.5) * (LXY_A / 2.0), Z_LO_A))
    for ix in range(2):
        for iy in range(2):
            upper.append(((ix + 0.5) * (LXY_A / 2.0), (iy + 0.5) * (LXY_A / 2.0), Z_HI_A))
    for i in range(16):
        ix = i % 4
        iy = (i // 4) % 2
        iz = i // 8
        fluid.append(
            (
                (ix + 0.5) * (LXY_A / 4.0),
                (iy + 0.5) * (LXY_A / 2.0),
                Z_LO_A + 2.2 if iz == 0 else Z_HI_A - 2.2,
            )
        )
    return lower, upper, fluid


def min_image(d, L):
    while d > 0.5 * L:
        d -= L
    while d < -0.5 * L:
        d += L
    return d


def analytic_fz_upper(lower, upper, fluid):
    """Minimum-image LJ z-force on the upper wall [eV/A].

    Includes fluid->upper and lower->upper.  Upper-upper pairs are coplanar
    (dz = 0) so they contribute nothing to F_z; MetalWalls does not compute
    electrode-electrode LJ at all, which is why the comparison is exact.
    """
    fz = 0.0
    rc2 = CUTOFF_A**2
    for ux, uy, uz in upper:
        for others in (fluid, lower):
            for ox, oy, oz in others:
                dx = min_image(ux - ox, LXY_A)
                dy = min_image(uy - oy, LXY_A)
                dz = min_image(uz - oz, LZ_A)
                r2 = dx * dx + dy * dy + dz * dz
                if r2 >= rc2:
                    continue
                inv2 = 1.0 / r2
                inv6 = inv2**3
                s6 = SIGMA_A**6
                # F = 24 eps (2 (s/r)^12 - (s/r)^6) / r^2 * d
                coef = 24.0 * EPS_EV * inv2 * s6 * inv6 * (2.0 * s6 * inv6 - 1.0)
                fz += coef * dz
    return fz


def main():
    here = pathlib.Path(__file__).resolve().parent
    lower, upper, fluid = atv_geometry()

    fz_ev_per_a = analytic_fz_upper(lower, upper, fluid)
    area_a2 = LXY_A * LXY_A
    print(f"[analytic] cutoff            = {CUTOFF_A} A")
    print(f"[analytic] F_z(upper wall)   = {fz_ev_per_a:.12e} eV/A")
    print(f"[analytic] A                 = {area_a2:.6f} A^2")
    print(f"[analytic] P_z = F_z/A       = {fz_ev_per_a / area_a2:.12e} eV/A^3")

    # ── data.inpt (Bohr) ────────────────────────────────────────────────
    lines = []
    lines.append("# header")
    lines.append("step                               0        0")
    lines.append(f"num_atoms                         {len(fluid) + len(lower) + len(upper)}")
    lines.append(f"num_electrode_atoms                {len(lower) + len(upper)}")
    lines.append("# box")
    lines.append(
        " {:.15E} {:.15E} {:.15E}".format(
            LXY_A * ANG2BOHR, LXY_A * ANG2BOHR, LZ_A * ANG2BOHR
        )
    )
    lines.append(f"# coordinates : {len(fluid) + len(lower) + len(upper)} atoms - step 0")
    for x, y, z in fluid:
        lines.append(
            "Ar       {:+.15E} {:+.15E} {:+.15E}".format(
                x * ANG2BOHR, y * ANG2BOHR, z * ANG2BOHR
            )
        )
    for x, y, z in lower:
        lines.append(
            "Wlo      {:+.15E} {:+.15E} {:+.15E}".format(
                x * ANG2BOHR, y * ANG2BOHR, z * ANG2BOHR
            )
        )
    for x, y, z in upper:
        lines.append(
            "Wup      {:+.15E} {:+.15E} {:+.15E}".format(
                x * ANG2BOHR, y * ANG2BOHR, z * ANG2BOHR
            )
        )
    lines.append(f"# velocities : {len(fluid) + len(lower) + len(upper)} atoms - step 0")
    for name, group in (("Ar", fluid), ("Wlo", lower), ("Wup", upper)):
        for _ in group:
            lines.append(
                "{:<8s} {:+.15E} {:+.15E} {:+.15E}".format(name, 0.0, 0.0, 0.0)
            )
    # Non-zero *initial guess* for the electrode charges.  Every ion here is
    # neutral and both electrode potentials are 0, so the exact solution of
    # the constant-potential problem is q = 0 -- but MetalWalls' CG divides
    # by <d,Ad> and would hit 0/0 = NaN if it started from the exact answer
    # (src/cg.f90 alpha = dp/dAd).  Seeding a tiny charge-neutral guess makes
    # the first iterate well defined; CG then converges back to q = 0, so the
    # total electrode force is pure Lennard-Jones.
    lines.append(f"# charges : {len(lower) + len(upper)} atoms - step 0")
    for name, group in (("Wlo", lower), ("Wup", upper)):
        for k in range(len(group)):
            lines.append("{:<8s} {:+.15E}".format(name, 1.0e-3 * (1 if k % 2 == 0 else -1)))
    (here / "data.inpt").write_text("\n".join(lines) + "\n")

    # ── runtime.inpt (atomic units) ─────────────────────────────────────
    eps_kjmol = (EPS_EV / EV_PER_HARTREE) / KJPERMOL2HARTREE
    rcut_bohr = CUTOFF_A * ANG2BOHR
    dt_au = DT_FS * AU_PER_FS
    # P_target is supplied on the command line of the comparison driver; the
    # step-0 force / pressure readout does not depend on it, so pin it at 0
    # for the static twin (the piston then feels only the LJ force).
    rt = f"""# MetalWalls twin of ATV regressions/md/piston-electrode-pz
# ---------------------------------------------------------
# Static (step-0) readout of the LJ z-force and P_z = F_z/A on the
# rigid upper piston electrode.  All charges are zero, so every Coulomb
# term vanishes and the electrode force is pure Lennard-Jones.

num_steps 1
timestep  {dt_au:.15E}
temperature 300.0

num_pbc 2

species

  species_type
    name Ar
    count {len(fluid)}
    charge point 0.0
    mass {M_FLUID}

  species_type
    name Wlo
    count {len(lower)}
    charge gaussian 0.9563 0.0
    mass {M_LOWER}
    mobile False

  species_type
    name Wup
    count {len(upper)}
    charge gaussian 0.9563 0.0
    mass {M_UPPER}
    mobile False

electrodes

  electrode_type
    name Wlo
    species Wlo
    potential {-0.5*POT_AU:+.4f}

  electrode_type
    name Wup
    species Wup
    potential {+0.5*POT_AU:+.4f}
    piston 0.0 -1

  electrode_charges cg 1.e-12 100
  charge_neutrality true

interactions
  coulomb
    coulomb_rtol 1.0e-6
    coulomb_rcut {rcut_bohr:.15E}
    coulomb_ktol 1.0e-7

  lennard-jones
    lj_rcut {rcut_bohr:.15E}
    lj_pair Ar  Ar  {eps_kjmol:.15E} {SIGMA_A:.15E}
    lj_pair Ar  Wlo {eps_kjmol:.15E} {SIGMA_A:.15E}
    lj_pair Ar  Wup {eps_kjmol:.15E} {SIGMA_A:.15E}

output
  step 1
  energies 1 Hamiltonian Kinetic Potential Coulomb van_der_Waals Lennard-Jones Piston_work
  elec_forces 1
"""
    (here / "runtime.inpt").write_text(rt)
    print(f"[gen] lj eps = {eps_kjmol:.15e} kJ/mol  ({EPS_EV} eV)")
    print(f"[gen] lj rcut = {rcut_bohr:.15e} bohr  ({CUTOFF_A} A)")
    print(f"[gen] dt = {dt_au:.15e} a.u.  ({DT_FS} fs)")
    print(f"[gen] wrote {here/'runtime.inpt'} and {here/'data.inpt'}")


if __name__ == "__main__":
    main()
