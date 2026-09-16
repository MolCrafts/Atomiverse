# virtual-site-multi-topology — general virtual sites (TIP4P + TIP5P)

oracle: lammps  
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)  
binary: `/home/jicli594/.local/x86_64/bin/lmp`

## What is pinned

**TIP4P** two-water single-point energy and **parent** forces using Atomiverse
`cpu::md::VirtualSites` (Weighted3 M-sites as Frame atoms) + `PairLJCut` +
`PairCoulCut`, matched to LAMMPS `pair_style lj/cut/tip4p/cut` (same physics as
`four-site-tip4p-foothold`, different ATV path).

**TIP5P** is dual-gated by geometry only (`r_OL`, `θ_LOL`) via OutOfPlane3
construction (no LAMMPS `tip5p` pair on host binary → no second energy pin).

| Item | Value |
|------|--------|
| r_OH | 0.9572 Å |
| θ_HOH | 104.52° |
| qdist (OM) | 0.15 Å |
| q_M (site) | −1.04 e (O charge 0 on Frame) |
| q_H | +0.52 e |
| ε_OO | 0.1550 kcal/mol → eV via `kcal_per_mol_to_ev` |
| σ_OO | 3.15365 Å |
| cut_lj = cut_coul | 10.0 Å |
| LJ shift | yes |
| special_bonds | lj/coul 0 0 0 |
| geometry | O–O = 2.8 Å along x; water2 rotated π about z |
| box | cubic 20 Å, periodic |

## Offline pin path

```text
tools/metalwalls_oracle/inputs/lammps/virtual-site-multi-topology/
  in.tip4p_dimer_metal
  data.tip4p_dimer_metal
  log.metal
```

(Same twin as `four-site-tip4p-foothold`; regenerate with `lmp -in in.tip4p_dimer_metal`.)

## Pin values (units metal = eV, eV/Å)

| Quantity | Value |
|----------|--------|
| \(E\) | `0.52924228894955327451` eV |

Forces on **parents only** (O1 H1 H2 O2 H3 H4), fy = 0 — after Feenstra
redistribute (M force must be zero):

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
- Geometry TIP4P: OM = qdist within \(10^{-10}\) Å  
- Forces: per-component on parents \(\le 10^{-6}\) eV/Å; site F = 0  
- TIP5P: \(r_{OL}=0.70\) Å and \(\theta_{LOL}=109.47°\) within \(10^{-8}\)  

## Perf

Fair wall: ATV `reg_virtual-site-multi-topology` vs LAMMPS
`in.tip4p_dimer_metal` / `run 0` (serial `OMP_NUM_THREADS=1`).

## Perf result

| Item | Value |
|------|--------|
| host | x86_64 login, Release MD-only build |
| T_atv (mean / launch) | 0.01055 s |
| T_oracle (mean / launch) | 0.1260 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) | **0.0837** |
| within_10x | **true** |
| perf_exempt | false |
| method | 50 process launches after 3 warmups; `OMP_NUM_THREADS=1` |
