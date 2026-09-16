import os, sys, shutil
BOHR=0.529177210903
r_b = 3.0/BOHR
aA = 1.0/BOHR**3
aB = 1.5/BOHR**3
DAMP = """
  damping
     tt_pair   A        B  1.0  1  0.0
     tt_pair   B        A  1.0  1  0.0
"""
def write(dirp, L_ang, pol):
    os.makedirs(dirp, exist_ok=True)
    L = L_ang/BOHR
    rcut = 0.47*L
    open(os.path.join(dirp,"data.inpt"),"w").write(
f"""# header
step                         0
num_atoms                    2
num_electrode_atoms          0
# box
   {L:.15f} {L:.15f} {L:.15f}
# coordinates
A    0.000000000000000        0.000000000000000        0.000000000000000
B    {r_b:.15f}        0.000000000000000        0.000000000000000
# velocities
A    0.000000000000000        0.000000000000000        0.000000000000000
B    0.000000000000000        0.000000000000000        0.000000000000000
""")
    pA = aA if pol else 0.0
    pB = aB if pol else 0.0
    mini = "  dipoles_minimization cg 1.0e-14 500\n" if pol else ""
    damp = DAMP if pol else ""
    open(os.path.join(dirp,"runtime.inpt"),"w").write(
f"""num_steps 0
timestep  41.341
temperature 0.0

num_pbc 3
box
  length {L:.15f} {L:.15f} {L:.15f}

species

  species_type
    name A
    count 1
    charge point 1.0
    mass 1.0
    polarizability {pA:.15f}

  species_type
    name B
    count 1
    charge point -1.0
    mass 1.0
    polarizability {pB:.15f}

{mini}
interactions
  coulomb
    coulomb_rtol 1.0e-15
    coulomb_rcut {rcut:.15f}
    coulomb_ktol 1.0e-15
{damp}
output
  default 1
  energies 1 Total Kinetic Potential Coulomb Coulomb_lr Coulomb_keq0 Coulomb_sr Coulomb_self Coulomb_molecule
  dipoles 1
  forces 1
""")
base="/tmp/mw_oracle_pim-induced-dipole-bulk/scan"
shutil.rmtree(base, ignore_errors=True)
for L in [50.0,100.0,200.0,400.0,800.0]:
    write(f"{base}/L{int(L)}_pol", L, True)
    write(f"{base}/L{int(L)}_nopol", L, False)
print("ok")
