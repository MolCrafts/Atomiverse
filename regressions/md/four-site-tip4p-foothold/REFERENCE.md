# four-site-tip4p-foothold — TIP4P four-site water dimer

oracle: lammps  
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)  
binary: `/home/jicli594/.local/x86_64/bin/lmp`

## What is pinned

Classic **Jorgensen TIP4P** (1983/85) two-water **single-point** energy and forces
with LAMMPS `pair_style lj/cut/tip4p/cut` and Atomiverse
`cpu::PairLJCutTip4pCut` (M virtual site + Feenstra force partition).

| Item | Value |
|------|--------|
| r_OH | 0.9572 Å |
| θ_HOH | 104.52° |
| qdist (OM) | 0.15 Å |
| q_O (on M) | −1.04 e |
| q_H | +0.52 e |
| ε_OO | 0.1550 kcal/mol → eV via `kcal_per_mol_to_ev` |
| σ_OO | 3.15365 Å |
| cut_lj = cut_coul | 10.0 Å |
| LJ shift | yes (`pair_modify shift yes` / ATV V(rc)=0) |
| special_bonds | lj/coul 0 0 0 |
| geometry | O–O = 2.8 Å along x; water2 rotated π about z |
| box | cubic 20 Å, periodic |

## Offline pin path

```text
tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold/
  in.tip4p_dimer_metal
  data.tip4p_dimer_metal
  log.metal
```

Regenerate:

```bash
cd tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold
lmp -in in.tip4p_dimer_metal -log log.metal
```

## Pin values (units metal = eV, eV/Å)

| Quantity | Value |
|----------|--------|
| \(E\) | `0.52924228894955327451` eV |
| \(E_{\mathrm{vdwl}}\) | `0.057185016777344513383` eV |
| \(E_{\mathrm{coul}}\) | `0.4720572721722087195` eV |

Forces (atom order O1 H1 H2 O2 H3 H4), fy = 0:

| atom | fx [eV/Å] | fz [eV/Å] |
|------|-----------|-----------|
| 1 O | −0.094382238074752045076 | 0.33187666441899288383 |
| 2 H | −1.0593571566097745862 | −0.31522207906906996033 |
| 3 H | −0.057823257196248978307 | −0.016654585349923187176 |
| 4 O | 0.094382238074751600987 | 0.33187666441899305037 |
| 5 H | 1.0593571566097748082 | −0.31522207906906962727 |
| 6 H | 0.057823257196249200351 | −0.01665458534992322534 |

## Tolerances (dual-gate)

- Energy: \(\|E - E_{\mathrm{pin}}\| \le 10^{-8}\) eV  
- Geometry: reconstructed OM = qdist within \(10^{-10}\) Å  
- Forces: per-component \(\le 10^{-6}\) eV/Å  

## Perf

Fair wall: same dimer single-point, ATV Evaluate (`reg_four-site-tip4p-foothold`)
vs LAMMPS `in.tip4p_dimer_metal` / `run 0` (both **serial** `OMP_NUM_THREADS=1`,
50 process launches after 3 warmups; includes process start + build).

## Perf result

| Item | Value |
|------|--------|
| host | x86_64 login, Release MD-only build |
| T_atv (mean / launch) | 0.0523 s |
| T_oracle (mean / launch) | 0.1548 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) | **0.338** |
| within_10x | **true** |
| perf_exempt | false |
