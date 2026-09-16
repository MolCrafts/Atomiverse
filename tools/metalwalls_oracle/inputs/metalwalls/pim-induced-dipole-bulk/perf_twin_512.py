import os, math, random
random.seed(20260729)
BOHR=0.529177210903
a_lat = 5.64          # rocksalt lattice constant [A]
n = 4                 # 4x4x4 conventional cells -> 512 ions (256 A + 256 B)
L = n*a_lat
alphaA, alphaB = 1.0, 1.5     # A^3  (same species as the regression)
pos=[]
for i in range(n):
    for j in range(n):
        for k in range(n):
            o=(i*a_lat,j*a_lat,k*a_lat)
            for d in [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]:
                pos.append(('A',(o[0]+d[0]*a_lat,o[1]+d[1]*a_lat,o[2]+d[2]*a_lat)))
            for d in [(0.5,0,0),(0,0.5,0),(0,0,0.5),(0.5,0.5,0.5)]:
                pos.append(('B',(o[0]+d[0]*a_lat,o[1]+d[1]*a_lat,o[2]+d[2]*a_lat)))
pos=[(s,(p[0]+random.uniform(-0.35,0.35),p[1]+random.uniform(-0.35,0.35),p[2]+random.uniform(-0.35,0.35))) for s,p in pos]
A=[p for p in pos if p[0]=='A']; B=[p for p in pos if p[0]=='B']
print("nA",len(A),"nB",len(B),"L",L)
Lb=L/BOHR; rcut=0.49*Lb
base="/tmp/mw_oracle_pim-induced-dipole-bulk/perf"
for tag,pol in [("pol",True),("nopol",False)]:
    d=f"{base}/{tag}"; os.makedirs(d,exist_ok=True)
    with open(d+"/data.inpt","w") as f:
        f.write(f"# header\nstep                         0\nnum_atoms                  {len(pos)}\nnum_electrode_atoms          0\n# box\n   {Lb:.15f} {Lb:.15f} {Lb:.15f}\n# coordinates\n")
        for s,p in A+B: f.write(f"{s}   {p[0]/BOHR:.12f} {p[1]/BOHR:.12f} {p[2]/BOHR:.12f}\n")
        f.write("# velocities\n")
        for s,p in A+B: f.write(f"{s}   0.0 0.0 0.0\n")
    pA = alphaA/BOHR**3 if pol else 0.0
    pB = alphaB/BOHR**3 if pol else 0.0
    mini = "  dipoles_minimization cg 1.0e-12 500\n" if pol else ""
    damp = "\n  damping\n     tt_pair   A        B  1.0  1  0.0\n     tt_pair   B        A  1.0  1  0.0\n" if pol else ""
    open(d+"/runtime.inpt","w").write(f"""num_steps 0
timestep  41.341
temperature 0.0

num_pbc 3
box
  length {Lb:.15f} {Lb:.15f} {Lb:.15f}

species

  species_type
    name A
    count {len(A)}
    charge point 1.0
    mass 22.9897
    polarizability {pA:.15f}

  species_type
    name B
    count {len(B)}
    charge point -1.0
    mass 35.453
    polarizability {pB:.15f}

{mini}
interactions
  coulomb
    coulomb_rtol 1.6300594548881187e-5
    coulomb_rcut {rcut:.12f}
    coulomb_ktol 1e-7
{damp}
output
  default 0
  energies 1 Total Kinetic Potential Coulomb Coulomb_lr Coulomb_keq0 Coulomb_sr Coulomb_self Coulomb_molecule
  dipoles 1
""")
# ATV geometry dump (A units) for the standalone timing consumer
with open(base+"/geom.txt","w") as f:
    f.write(f"{len(pos)} {L}\n")
    for s,p in A+B:
        f.write(f"{1 if s=='A' else 2} {1.0 if s=='A' else -1.0} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n")
print("written")
