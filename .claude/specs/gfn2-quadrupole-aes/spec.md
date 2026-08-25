# Spec — `gfn2-quadrupole-aes`

Wave 8 (this-shift wave 2), `tb-method-loop`, Phase C (GFN-xTB, oracle = **xtb**).

## Summary

Add **atomic quadrupole anisotropic electrostatics (AES)** and the
on-site quadrupole AXC kernel to production `cpu::semiempirical::GFN2`.
The GFN2 energy foothold (`gfn2-xtb-water`) is charge–dipole +
dipole–dipole + dipole AXC only; xtb 6.7.1 anisotropic ES includes
charge–quadrupole. Close the **1.45 meV** H₂O residual vs the archived
xtb pin **−5.070383095219 Eh**. Native static tables; **do not** wrap
`libxtb`.

## Domain basis

Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652 (2019),
DOI 10.1021/acs.jctc.8b01176. Family-locked oracle: **xtb only**.

tblite `damped_multipole` (molecular 0d) AES energy is:

- charge–dipole (`amat_sd`, \(r^{-3}\) damped `kdmp3=3`)
- dipole–dipole (`amat_dd`, \(r^{-3}/r^{-5}\) damped `kdmp5=4`)
- **charge–quadrupole** (`amat_sq`, packed \(r_i r_j r^{-5}\) damped `kdmp5`)
- on-site AXC: `dkernel |μ|²` and **`qkernel` Frobenius of packed Θ**
  (off-diagonal scale 2)

No dipole–quadrupole / quadrupole–quadrupole in the 0d kernel (tblite
stops at inverse-cubic order). Packed quadrupole is
`(xx, xy, yy, xz, yz, zz)` after the tblite traceless map
\(\Theta_{ii}=\tfrac32 Q_{ii}-\tfrac12\mathrm{tr}Q\),
\(\Theta_{ij}=\tfrac32 Q_{ij}\) (\(i\neq j\)).

Atomic Θ from Mulliken-partitioned Cartesian second moments, origin-shifted
to the bra atom (same partition as the existing dipole:
\(\tfrac12 P_{\mu\nu}\langle\mu|\cdots|\nu\rangle\), \(\mu\in A\)).
s-only centres (H) carry no CAMM quadrupole. Charge–quadrupole uses the
xtb even-kernel sign (−P·Q) and the 0d Buckingham \(\Theta{:}rr/3\) map.

AES remains a **post-SCC energy** on the isotropic-SCC density (foothold
shape). Forces still throw (`gfn2-xtb-forces`).

## Geometry / pin

Reuse the archived xtb 6.7.1 water twin
`tools/tb_oracle/inputs/xtb/h2o_gfn2/` (`mol.xyz --gfn 2 --chrg 0 --uhf 0`):

| Atom | x     | y    | z   |
|------|-------|------|-----|
| O    | 0.00  | 0.00 | 0.0 |
| H    | 0.96  | 0.00 | 0.0 |
| H    | −0.24 | 0.93 | 0.0 |

Total energy pin: **−5.070383095219 Eh**. Anisotropic ES in the transcript:
**0.000610908598 Eh**; AXC **−0.000805309300 Eh**.

## Design

No new public type. `GFN2` members unchanged. No `*Config`, no factory.
Tables already carry `qkernel` / `mprad` / `mpvcn`.

## Files

| Path | Action |
| ---- | ------ |
| `src/cpu/semiempirical/gfn2.cpp` | quadrupole integrals + AES/AXC |
| `include/atomiverse/cpu/semiempirical/gfn2.h` | comments (quadrupole AES in) |
| `tests/cpu/semiempirical/test_gfn2_params.cpp` | qkernel table + water energy |
| `regressions/semiempirical/gfn2-quadrupole-aes/` | new pin vs same xtb twin |
| `regressions/CMakeLists.txt` | `atv_add_regression(semiempirical gfn2-quadrupole-aes)` |
| `tools/tb_oracle/inputs/xtb/h2o_gfn2/README.md` | note this slug reuses the twin |

## Out of scope

- Analytic forces (`gfn2-xtb-forces`)
- Putting AES in the SCC Fock (potential) — later, with forces
- Elements beyond H/C/N/O (`gfn2-beyond-chno`)
- ALPB / spin / PBC / IPEA / sTDA
- Wrapping libxtb / tblite
- CN `dexp` vs D3-logistic remainder (not this slug)
