# LAMMPS offline pin — freeze-electrode-atoms

- **Oracle:** LAMMPS `fix setforce 0.0 0.0 0.0` on wall group
- **Version:** develop / patch_30Mar2026-1074-g7f680de296
- **Inputs:** `in.lammps` (1000-step correctness), `in.bench` (20000-step fair wall)
- **Result:** wall positions fixed at create; wall velocities identically 0
- **Consumed by:** `regressions/md/freeze-electrode-atoms/REFERENCE.md`

ctest never launches LAMMPS.
