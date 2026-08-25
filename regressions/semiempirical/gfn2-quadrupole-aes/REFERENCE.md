# H2O GFN2-xTB energy with quadrupole AES

```text
oracle: xtb
oracle_version: 6.7.1 (edcfbbe)
```

## Method

GFN2-xTB (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)).

Family-locked numerical oracle: **xtb** (never OpenMOPAC / DFTB+).

Native static in-tree tables. No `libxtb` wrap.

This slug adds **charge–quadrupole AES** and the on-site **qkernel AXC**
on top of the `gfn2-xtb-water` foothold (charge–dipole + dipole–dipole +
dipole AXC). Kernel transcribed from tblite `damped_multipole` molecular
0d (`amat_sq`, packed `(xx,xy,yy,xz,yz,zz)`, traceless Buckingham map).

## Geometry

Water, slightly off-equilibrium (Å) — same ATV H₂O as `gfn2-xtb-water`.
Archived xtb 6.7.1 transcript
`tools/tb_oracle/inputs/xtb/h2o_gfn2/xtb.out` (`mol.xyz --gfn 2 --chrg 0
--uhf 0`); input twin `mol.xyz`.

| Atom | x    | y    | z   |
|------|------|------|-----|
| O    | 0.00 | 0.00 | 0.0 |
| H    | 0.96 | 0.00 | 0.0 |
| H    |−0.24 | 0.93 | 0.0 |

Cubic box 20 Å (isolated). Transcript: 6 AO, 4 shells, 8 electrons,
HOMO–LUMO 14.519 eV, Mulliken q(O)=−0.563, q(H)=0.282 / 0.281.

## Pin

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E_{\mathrm{tot}}\) | **−5.070383095219** | Eh | — |
| \(E_{\mathrm{tot}}\) | −5.070383095219 × `units::kHartreeInEv` | eV | \(1\times10^{-4}\) eV |

xtb summary (same transcript):

| Piece | Eh |
| ----- | -- |
| SCC | −5.104517627658 |
| isotropic ES | 0.031400392671 |
| anisotropic ES | 0.000610908598 |
| anisotropic XC | −0.000805309300 |
| dispersion | −0.000141270483 |
| repulsion | 0.034134532439 |
| **total** | **−5.070383095219** |

Wall (xtb, 1 thread): 0.155 s (SCC 0.115 s).

## Gates

- Finite, non-zero total energy [eV] on `frame.meta[kEnergy]`
- SCC residual < `scc_tol`
- Hard-coded energy pin vs table (tol \(1\times10^{-4}\) eV)

The foothold residual without quadrupole AES was **1.45 meV**. Measured
ATV residual with charge–quadrupole + qkernel AXC: **26 µeV**.

## Perf

`perf_exempt: true` — no xtb binary on the SE build host for a fair wall
ratio. Archived xtb wall is 0.155 s. Revisit when xtb is on PATH.

## Gaps

- Analytic forces: `gfn2-xtb-forces`
- AES inside the SCC Fock (potential), not only post-energy
- Elements beyond H/C/N/O
- ALPB, spin, PBC, IPEA, sTDA
