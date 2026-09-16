# surface-tension-coupling — constant surface-tension NPT (couple xy)

oracle: lammps
oracle_version: 30 Mar 2026 - Development (patch_30Mar2026-1074-g7f680de296); binary `/home/jicli594/.local/x86_64/bin/lmp`

## What is checked

Public `cpu::md::SurfaceTensionNPT` (MTK semi-isotropic barostat with
LAMMPS `fix npt … couple xy` lockstep and NPγT lateral target
\(P_\perp = P_n - n_\mathrm{surf}\,\gamma/L_z\)):

1. **Couple-xy lockstep:** \(L_x = L_y\) after MD steps.
2. **Anisotropic response:** under nonzero \(\gamma\) (equivalently
   \(P_\perp \neq P_n\)), \(L_z\) need not track \(L_x\); box expands from
   a compressed LJ FCC start without runaway volume.
3. **Surface tension diagnostic:**
   \(\gamma_\mathrm{inst} = \frac{L_z}{2}\bigl(P_{zz}-\frac12(P_{xx}+P_{yy})\bigr)\)
   is finite (LAMMPS pressure sign: positive compressive).

## Physical setup (matches offline pin input)

| Quantity | Value | Note |
| -------- | ----- | ---- |
| Potential | LJ \(\varepsilon=0.0104\,\mathrm{eV}\), \(\sigma=3.4\,\mathrm{Å}\), \(r_c=8.5\,\mathrm{Å}\) | argon-like |
| Lattice | FCC \(a=5.1\,\mathrm{Å}\), \(3^3\) cells (108 atoms) | compressed |
| \(T\) | 100 K | |
| \(P_n\) | 0 eV/Å³ | normal target |
| \(\gamma\) | 0.001 eV/Å² | \(n_\mathrm{surf}=2\) |
| \(L_z(0)\) | 15.3 Å | ⇒ \(P_\perp(0)=-1.30719\times10^{-4}\) eV/Å³ |
| \(\tau_T,\tau_P\) | 50 fs, 500 fs | metal: 0.05 ps, 0.5 ps |
| \(dt\) | 1 fs | metal timestep 0.001 ps |
| steps | 200 | e2e production |

## Offline pin (LAMMPS)

Input: `tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling`

LAMMPS cannot take equal-style variables for `Pstart`/`Pstop` on `fix npt`,
so the twin uses **fixed** \(P_\perp = P_n - n\gamma/L_z(0)\) (community
surface-tension path). Atomiverse updates \(P_\perp(L_z)\) each step (true
NPγT); short-run gates are structural, not bit-identical trajectories.

Pin snapshot (`lmp`, OMP=1, 2026-07-24):

| step | \(L_x\) | \(L_y\) | \(L_z\) |
| ---: | ---: | ---: | ---: |
| 0 | 15.3 | 15.3 | 15.3 |
| 200 | **16.527384** | **16.527384** | 16.249165 |

- \(L_x = L_y\) exactly under `couple xy`.
- \(L_z \neq L_x\) under anisotropic targets (surface-tension path).
- Box expanded from compressed start.

Metal conversion used for the pin: \(1\,\mathrm{eV/Å^3} = 1.6021766208\times10^6\,\mathrm{bar}\),
so \(P_\perp=-209.434\,\mathrm{bar}\).

## Dual gate (regression)

1. Structural: \(|L_x-L_y| \le 10^{-9}\max(1,|L_x|)\).
2. Surface-tension path: finite safe volume + edge motion; anisotropy /
   expansion under \(\gamma\neq 0\).

## Fair wall (perf gate)

Comparable workload: **108 atoms, 5000 MD steps**, OMP=1, same LJ + couple-xy
NPT settings as the pin (fixed Plat on LAMMPS; SurfaceTensionNPT on ATV).

| Engine | Measurement | Wall [s] |
| --- | --- | ---: |
| LAMMPS (`lmp`) | process wall, 5000 steps | **0.140** |
| Atomiverse | \(25\times\) full `reg_surface-tension-coupling` (200 steps each = 5000 atom-steps; includes 25× `build()` — conservative vs pure loop) | **0.403** |

`ratio_atv_over_oracle = T_atv / T_lammps = 0.403 / 0.140 ≈ 2.88` (≤ 10).  
within_10x: **true**. Host: same login node, 2026-07-24.

## Provenance

- LAMMPS docs: <https://docs.lammps.org/fix_nh.html> (`couple xy`)
- Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994)
- Queue: `docs/developer/metalwalls-method-queue.md` — `surface-tension-coupling`
