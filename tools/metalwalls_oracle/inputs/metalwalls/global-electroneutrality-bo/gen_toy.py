import math, os, sys

BOHR = 0.529177210903          # A per Bohr
HA_EV = 27.211386245988        # eV per Ha

A2B = 1.0 / BOHR               # Bohr per A

# ---- ATV regression system (regressions/md/global-electroneutrality-bo) ----
Lx_A, Ly_A, Lz_A = 10.0, 10.0, 20.0
z_e1_A, z_e2_A, z_ion_A = 5.0, 15.0, 10.0
eta_invA = 2.0                 # ConstPotBO(2.0) [1/A]
psi1_V, psi2_V = 0.0, 1.0
q_ion = 1.0

# ---- conversions ----
Lx, Ly, Lz = Lx_A*A2B, Ly_A*A2B, Lz_A*A2B
z_e1, z_e2, z_ion = z_e1_A*A2B, z_e2_A*A2B, z_ion_A*A2B
eta_au = eta_invA * BOHR       # [1/Bohr]
psi1_au, psi2_au = psi1_V/HA_EV, psi2_V/HA_EV

print(f"box  [Bohr]  {Lx!r} {Ly!r} {Lz!r}")
print(f"eta  [1/Bohr] {eta_au!r}")
print(f"psi2 [Ha/e]   {psi2_au!r}")

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

with open(os.path.join(out, "data.inpt"), "w") as f:
    f.write("# header\n")
    f.write("step                               0\n")
    f.write("num_atoms                          3\n")
    f.write("num_electrode_atoms                2\n")
    f.write("# box\n")
    f.write(f" {Lx:.15E} {Ly:.15E} {Lz:.15E}\n")
    f.write("# coordinates : 3 atoms - step 0.0\n")
    f.write(f"Ion  {0.0:.15E}  {0.0:.15E}  {z_ion:.15E}\n")
    f.write(f"E1   {0.0:.15E}  {0.0:.15E}  {z_e1:.15E}\n")
    f.write(f"E2   {0.0:.15E}  {0.0:.15E}  {z_e2:.15E}\n")
    f.write("# velocities : 3 atoms - step 0.0\n")
    for nm in ("Ion", "E1", "E2"):
        f.write(f"{nm}  {0.0:.15E} {0.0:.15E} {0.0:.15E}\n")

rcut = float(sys.argv[2]) if len(sys.argv) > 2 else 9.4
with open(os.path.join(out, "runtime.inpt"), "w") as f:
    f.write(f"""# ATV regression twin: global-electroneutrality-bo
# two single-atom electrodes at Psi = 0 V / 1 V, one +1e point ion between them

num_steps 0
timestep  41.341
temperature 0.0

num_pbc 2

species

  species_type
    name Ion
    count 1
    charge point {q_ion:.15E}
    mass 1.0

  species_type
    name E1
    count 1
    charge gaussian {eta_au:.15E} 0.0
    mass 1.0
    mobile False

  species_type
    name E2
    count 1
    charge gaussian {eta_au:.15E} 0.0
    mass 1.0
    mobile False

electrodes

  electrode_type
    name E1
    species E1
    potential {psi1_au:.15E}

  electrode_type
    name E2
    species E2
    potential {psi2_au:.15E}

  electrode_charges matrix_inversion
  charge_neutrality true global

interactions
  coulomb
    coulomb_rtol 1.0e-12
    coulomb_rcut {rcut:.6f}
    coulomb_ktol 1.0e-14

output
  step 1
  charges 1
  total_charges 1
  energies 1 Hamiltonian Potential Coulomb
""")
print("written to", out)
