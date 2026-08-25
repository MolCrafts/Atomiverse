# CH4 IPEA-xTB energy

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

## Method

IPEA-xTB (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)). GFN1 Hamiltonian
reparametrized for IPs/EAs, with extra \((n+1)s\) functions.

Family-locked numerical oracle: **xtb** (never OpenMOPAC / DFTB+).

Native static in-tree tables. No `libxtb` wrap.

Same tetrahedral CH4 as the GFN1 organic pin, so the two Hamiltonians are
discriminated on one geometry (GFN1 −4.274232224875 Eh; IPEA **−4.670476737294
Eh**).

## Geometry

Methane, tetrahedral, \(r(\mathrm{C{-}H}) = 1.09\) Å. Archived xtb 6.7.1
transcript `tools/tb_oracle/inputs/xtb/ch4_ipea/xtb.out.txt`. Input twin
`mol.xyz`. \(s = 1.09/\sqrt{3}\).

IPEA is selected by pointing `XTBPATH` at a directory whose
`param_gfn1-xtb.txt` is a copy of `param_ipea-xtb.txt` (xtb then prints the
`I P E A - x T B` banner). Command:

```text
XTBPATH=<twin xtbpath> xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --sp
OMP_NUM_THREADS=1
```

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H | \(s\) | \(s\) | \(s\) |
| H | \(s\) | \(-s\) | \(-s\) |
| H | \(-s\) | \(s\) | \(-s\) |
| H | \(-s\) | \(-s\) | \(s\) |

Cubic box 20 Å (isolated). Transcript: 13 AO, 8 electrons, HOMO–LUMO
7.751 eV.

`--vip` is a multi-step ΔSCC IP with an empirical shift; it is **not** this
energy pin.

## Pin

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E_{\mathrm{tot}}\) | **−4.670476737294** | Eh | — |
| \(E_{\mathrm{tot}}\) | −4.670476737294 × `units::kHartreeInEv` | eV | \(0.05\) eV |

xtb summary (same transcript):

| Piece | Eh |
| ----- | -- |
| SCC | −4.766828456418 |
| isotropic ES | 0.001132006161 |
| repulsion | 0.096806468732 |
| dispersion | −0.000454749608 |
| halogen | 0 |
| **total** | **−4.670476737294** |

Measured ATV residual at this geometry: **9 meV**. The 0.05 eV window is the
same kernel-residual floor as `reg_gfn1-xtb-organic` (D3 CN vs xtb ncoord,
Stewart contraction vs `intcut`, Anderson vs Broyden).

Wall (xtb, 1 thread): 0.005 s (xtb clock) / 0.02 s (`/usr/bin/time`).

## Gates

- Finite, non-zero total energy [eV] on `frame.meta[kEnergy]`
- SCC residual < `scc_tol`
- Hard-coded energy pin vs table (tol \(0.05\) eV)

## Perf

Fair 1-thread wall vs the archived xtb clock (0.005 s). Gate
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) at the dual-gate run; otherwise
`perf_exempt` with reason.

## Gaps

- Analytic forces
- Elements beyond H/C/N/O
- Open-shell / `--vip` ΔSCC IP+EA (cation/anion)
- Polar-molecule residual vs water (E3 / charge; not this pin)
- ALPB, PBC, sTDA
