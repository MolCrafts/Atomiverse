# Pin notes — const-pot-il-edl-box (2026-07-27)

- **Oracle:** MetalWalls-form Gaussian-electrode BO constant-potential solve
  (Scalfi et al. JCP 153, 174704 (2020); MW JOSS 10.21105/joss.02373)
- **Why not LAMMPS:** available `lmp` (30 Mar 2026 development,
  `/home/jicli594/.local/x86_64/bin/lmp`) has **no ELECTRODE package**
  (checked `lmp -h` installed packages 2026-07-27); no `mw` binary on host
- **Script:**
  `tools/metalwalls_oracle/inputs/metalwalls/const-pot-il-edl-box/pin_il_edl.py`
  — independent numpy/scipy implementation of the same algebra as
  `cpu::md::ConstPotBO` with `xy_image_range=0` (Gaussian self
  k_e·η·√(2/π); Gaussian–Gaussian erf((η/√2)r)/r; Gaussian–point
  erf(η·r)/r; min-image; electrode-only neutrality projection), direct
  `numpy.linalg.solve` (vs ATV CG tol 1e-12 in the regression)
- **Command:** `python3 pin_il_edl.py` (numpy 2.5.1, scipy 1.18.0)
- **Geometry:** honeycomb 4×7-cell C walls (112 sites each) at z = 12 / 28 Å,
  Ψ = ∓1 V, η = 1.805132 Å⁻¹ (MW graphite 0.955234 bohr⁻¹); 12+12 CG IL
  ions, deterministic checkerboard + one swap per outer layer; box
  17.04 × 17.216585027234639 × 40 Å
- **Pins [e]:**
  - empty capacitor: Q_left = −4.965149459545e−01, Q_right = +4.965149459545e−01
  - with IL: Q_left = −1.552770528510e+00, Q_right = +1.552770528510e+00
  - screening ratio |Q|_IL/|Q|_empty ≈ 3.127
- **perf_exempt:** no MW twin binary, no LAMMPS ELECTRODE
