#!/usr/bin/env python3
"""MetalWalls twins of ``regressions/md/const-pot-cg-large-electrode``.

Offline oracle generator — never run at ctest time.  Emits MetalWalls
``runtime.inpt`` / ``data.inpt`` pairs for the two systems the regression
actually simulates, plus the convergence ladders used to extrapolate them.

Cases
-----
``primary/``   the 4-particle system whose ``Q_top`` the regression pins
               (``madelung_primary()``: Gaussian electrode sites at z = 8 and
               12 A held at 0 and 1 V, point ions +-0.5 e at z = 9 and 11 A,
               all sharing the same x, y).  Atomiverse builds it with
               ``xy_image_range = 0``, i.e. min-image distances and no lattice
               sum; since every separation is along z and short compared with
               L_z = 20 A, that is *exactly* the isolated 4-particle cluster.
               MetalWalls has no non-periodic mode (``num_pbc`` is 2 or 3), so
               the isolated limit is reached by widening BOTH the lateral cell
               L and the z cell L_z and extrapolating.  Both directions matter:
               the lateral image term decays as 1/L^3 and MetalWalls' 2D k_z
               quadrature has spacing 2*pi/L_z.

``plate/``     the 4x4 + 4x4 site plate of the regression's large case at its
               own cell (L_x = L_y = 4 A, L_z = 20 A), solved with the full 2D
               Ewald sum.  This quantifies the error of Atomiverse's
               min-image-truncated elastance on a genuinely periodic wall.

Units.  Atomiverse public units are eV / A / e; MetalWalls is atomic units
(Hartree / Bohr / e).  Conversions used here:

    1 Ha   = 27.211386245988 eV       -> psi [V] / 27.211386245988 = psi [Ha/e]
    1 Bohr = 0.529177210903 A         -> z [A] * 1.889726125458    = z [Bohr]
    eta [1/A] * 0.529177210903        = eta [1/Bohr]

Run:  python3 gen_inputs.py            (writes the case directories in place)
Then: cd <case>; mpirun -n 1 <path>/mw
"""

import math
import os

HA_EV = 27.211386245988      # 1 Hartree [eV]
BOHR_A = 0.529177210903      # 1 Bohr [A]
A_BOHR = 1.0 / BOHR_A        # 1 Angstrom [Bohr] = 1.8897261254578281

# --- Atomiverse regression parameters --------------------------------------
ETA_INV_A = 2.0              # ConstPotCG eta [1/A]
PSI_BOT_V = 0.0              # [V]
PSI_TOP_V = 1.0              # [V]
Z_BOT_A, Z_TOP_A = 8.0, 12.0
Z_ION_P_A, Z_ION_M_A = 9.0, 11.0
Q_ION = 0.5
MASS_ELEC = 196.966553       # amu (Au) - irrelevant at num_steps 0
MASS_ION = 1.0

ETA_INV_BOHR = ETA_INV_A * BOHR_A          # 1.058354421806 /Bohr
PSI_BOT_AU = PSI_BOT_V / HA_EV
PSI_TOP_AU = PSI_TOP_V / HA_EV             # 0.03674932217565499 Ha/e

HERE = os.path.dirname(os.path.abspath(__file__))


def runtime(species, rcut_bohr, rtol=1.0e-12, ktol=1.0e-12,
            cg_tol=1.0e-14, cg_max=2000):
    """runtime.inpt text.  species = [(name, count, mass, kind, value)]."""
    out = ["# MetalWalls twin of Atomiverse",
           "# regressions/md/const-pot-cg-large-electrode",
           "# Single-point constant-potential electrode charge solve.",
           "",
           "num_steps    0",
           "timestep     41.341",
           "temperature  0",
           "num_pbc      2",
           "",
           "species"]
    for name, count, mass, kind, value in species:
        out += ["", "  species_type", f"    name   {name}",
                f"    count  {count}", f"    mass   {mass!r}"]
        if kind == "point":
            out += [f"    charge point {value!r}", "    mobile True"]
        else:
            out += [f"    charge gaussian {value!r} 0.0", "    mobile False"]
    out += ["", "electrodes", "",
            "  electrode_type", "    name       EB", "    species    EB",
            f"    potential  {PSI_BOT_AU!r}", "",
            "  electrode_type", "    name       ET", "    species    ET",
            f"    potential  {PSI_TOP_AU!r}", "",
            f"  electrode_charges cg {cg_tol:.1e} {cg_max}",
            "  charge_neutrality true electrodes", "",
            "interactions", "  coulomb",
            f"    coulomb_rtol {rtol:.3e}",
            f"    coulomb_rcut {rcut_bohr!r}",
            f"    coulomb_ktol {ktol:.3e}", "",
            "output", "  default 0", "  step 1", "  charges 1",
            "  energies 1 Hamiltonian Potential Coulomb Coulomb_lr"
            " Coulomb_keq0 Coulomb_sr Coulomb_self", ""]
    return "\n".join(out)


def data(box_bohr, ions, atoms):
    """data.inpt text.  ions/atoms = [(name, x, y, z)] in Bohr."""
    out = ["# header", "step                            0",
           f"num_atoms                {len(ions) + len(atoms):7d}",
           f"num_electrode_atoms      {len(atoms):7d}",
           "# box",
           " {:.15E} {:.15E} {:.15E}".format(*box_bohr),
           "# coordinates"]
    for name, x, y, z in ions + atoms:
        out.append(f"{name:<4s} {x:.15E} {y:.15E} {z:.15E}")
    # MetalWalls reads ion AND electrode-atom velocities; the latter must be
    # exactly 0.0 (src/system.F90 read_data).
    out.append("# velocities")
    for name, _x, _y, _z in ions + atoms:
        out.append(f"{name:<4s} {0.0:.15E} {0.0:.15E} {0.0:.15E}")
    out.append("")
    return "\n".join(out)


def write_case(relpath, runtime_txt, data_txt):
    path = os.path.join(HERE, relpath)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "runtime.inpt"), "w") as fh:
        fh.write(runtime_txt)
    with open(os.path.join(path, "data.inpt"), "w") as fh:
        fh.write(data_txt)
    print("wrote", relpath)


def primary(L_lat_A, L_z_A):
    """4-particle twin at lateral cell L_lat_A^2 and z cell L_z_A."""
    Lx, Lz = L_lat_A * A_BOHR, L_z_A * A_BOHR
    xc = 0.5 * Lx
    zb = lambda z_A: (z_A - 10.0) * A_BOHR   # noqa: E731 (centre in z)
    ions = [("IP", xc, xc, zb(Z_ION_P_A)), ("IN", xc, xc, zb(Z_ION_M_A))]
    atoms = [("EB", xc, xc, zb(Z_BOT_A)), ("ET", xc, xc, zb(Z_TOP_A))]
    species = [("IP", 1, MASS_ION, "point", Q_ION),
               ("IN", 1, MASS_ION, "point", -Q_ION),
               ("EB", 1, MASS_ELEC, "gaussian", ETA_INV_BOHR),
               ("ET", 1, MASS_ELEC, "gaussian", ETA_INV_BOHR)]
    rcut_A = 0.45 * min(L_lat_A, L_z_A)      # MW requires 2*rcut <= min(a,b,c)
    write_case(f"primary/L{int(L_lat_A)}_Z{int(L_z_A)}",
               runtime(species, rcut_A * A_BOHR),
               data((Lx, Lx, Lz), ions, atoms))


def plate(L_z_A, nx=4, ny=4):
    """nx*ny + nx*ny site plate at the regression's own lateral cell."""
    Lx_A, Ly_A = float(nx), float(ny)
    Lx, Ly, Lz = Lx_A * A_BOHR, Ly_A * A_BOHR, L_z_A * A_BOHR
    zb = lambda z_A: (z_A - 10.0) * A_BOHR   # noqa: E731
    ions = [("IP", 0.5 * Lx, 0.5 * Ly, zb(Z_ION_P_A)),
            ("IN", 0.5 * Lx, 0.5 * Ly, zb(Z_ION_M_A))]
    atoms = []
    for name, z_A in (("EB", Z_BOT_A), ("ET", Z_TOP_A)):
        for iy in range(ny):
            for ix in range(nx):
                atoms.append((name,
                              (ix + 0.5) * (Lx_A / nx) * A_BOHR,
                              (iy + 0.5) * (Ly_A / ny) * A_BOHR,
                              zb(z_A)))
    species = [("IP", 1, MASS_ION, "point", Q_ION),
               ("IN", 1, MASS_ION, "point", -Q_ION),
               ("EB", nx * ny, MASS_ELEC, "gaussian", ETA_INV_BOHR),
               ("ET", nx * ny, MASS_ELEC, "gaussian", ETA_INV_BOHR)]
    rcut_A = 0.45 * min(Lx_A, Ly_A, L_z_A)
    write_case(f"plate/Z{int(L_z_A)}",
               runtime(species, rcut_A * A_BOHR),
               data((Lx, Ly, Lz), ions, atoms))


if __name__ == "__main__":
    print(f"eta {ETA_INV_A} 1/A -> {ETA_INV_BOHR!r} 1/Bohr")
    print(f"psi {PSI_TOP_V} V   -> {PSI_TOP_AU!r} Ha/e")
    # primary: double-limit ladder (lateral 1/L^3, z quadrature 2*pi/L_z)
    for L in (200.0, 400.0, 800.0, 1600.0):
        for Lz in (200.0, 400.0, 800.0, 1600.0):
            primary(L, Lz)
    # plate: at the regression's own cell; Lz ladder shows it is z-insensitive
    for Lz in (20.0, 40.0, 80.0, 160.0, 320.0):
        plate(Lz)
