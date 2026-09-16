# pr_npt_lj — Parrinello–Rahman NPT on compressed LJ argon

## What is checked

Public `cpu::md::ParrinelloRahmanNPT` on compressed FCC argon (LJ):

1. Cell volume **increases** from an over-compressed start at \(P_\mathrm{target}=0\)
   (pressure relief).
2. Volume stays within a safe multiple of \(V_0\) (no runaway collapse/explosion).
3. Off-diagonal cell entries remain bounded relative to the edge length under
   hydrostatic target (isotropic liquid limit of full-cell PR).
4. Extended MTK Hamiltonian on `kConserved` is finite.

## Physical setup

| Quantity | Value | Note |
| -------- | ----- | ---- |
| Potential | LJ \(\varepsilon=0.0104\,\mathrm{eV}\), \(\sigma=3.4\,\mathrm{Å}\) | Argon-like |
| Lattice | FCC \(a=5.1\,\mathrm{Å}\), \(3^3\) cells (108 atoms) | Compressed vs Ar eq. |
| \(T\) | 100 K | Thermostat target |
| \(P\) | 0 eV/Å³ | Hydrostatic |
| \(\tau_T,\tau_P\) | 100 fs, 1000 fs | |
| \(dt\) | 1 fs | 800 steps |

## Gates

| Check | Bound |
| ----- | ----- |
| \(V_1 > V_0\) | expansion from compressed start |
| \(V_1 < 5 V_0\) | no runaway |
| shear / edge | < 0.20 |
| \(H\), \(V\) | finite, \(V>0\) |

## Literature

Parrinello & Rahman, *J. Appl. Phys.* **52**, 7182 (1981);
Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994).
