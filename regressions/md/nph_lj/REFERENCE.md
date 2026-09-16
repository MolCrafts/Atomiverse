# nph_lj — isotropic NPH enthalpy conservation (LJ)

## What is checked

Public thermostat-free `cpu::md::NphIntegrator` (isotropic NPH):

1. Extended enthalpy
   \(H = E_\mathrm{kin}+E_\mathrm{pot}+P V+\tfrac12 W\dot\varepsilon^2\)
   written to `kConserved` stays within a relative drift gate over production.
2. Volume remains positive and non-runaway.

## Physical setup

| Quantity | Value | Note |
| -------- | ----- | ---- |
| Potential | LJ argon-like | \(\varepsilon=0.0104\,\mathrm{eV}\), \(\sigma=3.4\,\mathrm{Å}\) |
| Lattice | Compressed FCC \(a=5.2\,\mathrm{Å}\), 108 atoms | Dense start |
| Ensemble | NPH (no atom thermostat) | |
| \(P\) | 0 eV/Å³ | Hydrostatic |
| \(\tau_P\) | 1000 fs | Barostat mass only |
| \(dt\) | 1 fs | warm 100 + prod 600 |

## Gates

| Check | Bound |
| ----- | ----- |
| \(\max\|\Delta H\|/\max(1,\|H_0\|)\) | \(\le 0.15\) |
| volume | \(0 < V < 10 V_0\) |

## Literature

Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994) — MTK barostat;
NPH is the thermostat-free subset (conserved extended enthalpy).
