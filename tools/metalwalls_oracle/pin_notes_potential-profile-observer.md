# Pin notes — potential-profile-observer

**Date:** 2026-07-24  
**Slug:** `potential-profile-observer`  
**Oracle:** literature (planar Poisson / parallel-plate; MetalWalls potential-profile lineage)

No MetalWalls binary and no LAMMPS built-in planar Poisson \(\Psi(z)\) compute
on the agent host. Offline pin is the closed-form planar Poisson integration
in MD metal units (`units::coulomb`) on a synthetic two-sheet capacitor.

## Procedure

1. Geometry: \(L_x=L_y=10\) Å, \(A=100\) Å², \(z\in[0,20)\), 20 bins,
   \(Q=\pm 1\,e\) at bin centers \(z=5.5\) and \(z=14.5\).
2. \(\sigma = Q/A = 0.01\) e/Å²;
   \(E_{\mathrm{gap}} = 4\pi k_{\mathrm{e}}\sigma\).
3. Discrete piecewise-constant \(\rho\) integration (same as
   `cpu::PotentialProfile`) yields mid-gap \(E\), vacuum \(E=0\),
   \(\Psi(z_{\max}) = -E_{\mathrm{gap}}\times 9\).
4. Recorded in `regressions/md/potential-profile-observer/REFERENCE.md`.

## Perf

`perf_exempt` — analysis micro-kernel; no oracle twin binary for fair wall.
