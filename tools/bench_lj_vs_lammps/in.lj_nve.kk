# Fair LJ liquid NVE vs Atomiverse OpenMM CUDA path.
# density=0.8 (sc a0=rho^{-1/3}), cut=2.5, skin=0.3, dt=0.005, eps=sig=1.
# Pass: -var N <atoms> -var WARM <n> -var STEPS <n>  (index-style via CLI)

# CLI -var creates index vars; do not redeclare them.
variable nside equal ceil(v_N^(1.0/3.0))
variable a0 equal (1.0/0.8)^(1.0/3.0)

units           lj
atom_style      atomic
boundary        p p p

lattice         sc ${a0}
region          box block 0 ${nside} 0 ${nside} 0 ${nside}
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create 1.0 87287 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

fix             1 all nve
timestep        0.005

thermo          0
thermo_style    one

run             ${WARM}
reset_timestep  0
timer           full
run             ${STEPS}
