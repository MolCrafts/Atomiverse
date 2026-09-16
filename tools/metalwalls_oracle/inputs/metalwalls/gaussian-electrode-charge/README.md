# Offline oracle notes — gaussian-electrode-charge

**oracle:** literature (Reed 2007 + MetalWalls JOSS kernel)  
**Why not live MetalWalls / LAMMPS:** agent host has no MetalWalls binary and
no LAMMPS `ELECTRODE` package. Pins are closed-form evaluations of the published
Gaussian-charge Green function (identical physics to MW electrode Gaussians).

When a MetalWalls or LAMMPS ELECTRODE binary becomes available, a fair wall twin
can be added here; until then regressions use analytic pins in
`regressions/md/gaussian-electrode-charge/REFERENCE.md`.
