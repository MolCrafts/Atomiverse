# CH4 IPEA-xTB vertical IP/EA (ΔSCC)

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

## Method

IPEA-xTB (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)). Vertical
ionization potential and electron affinity as xtb `--vip` / `--vea` /
`--vipea`:

\[
\mathrm{IP}_v = \bigl(E(N-1)-E(N)\bigr)[\mathrm{eV}] - \Delta_{\mathrm{IPEA}},
\qquad
\mathrm{EA}_v = \bigl(E(N)-E(N+1)\bigr)[\mathrm{eV}] - \Delta_{\mathrm{IPEA}}.
\]

\(\Delta_{\mathrm{IPEA}} = 4.8455\,\mathrm{eV}\) is `ipeashift` from
`param_ipea-xtb.txt` (1.78069 × 0.1 × tblite 27.21138505). Ionized SCCs
use Fermi–Dirac occupations at the xtb default \(T_{\mathrm{el}}=300\,\mathrm{K}\)
(restricted; not UHF). Neutral CH4 stays integer double occupation.

Family-locked numerical oracle: **xtb** (never OpenMOPAC / DFTB+).

Native static in-tree tables. No `libxtb` wrap. Same tetrahedral CH4 as
`reg_ipea-xtb-small`. Polar water is not this pin (0.5 eV kernel hole).

## Geometry

Methane, tetrahedral, \(r(\mathrm{C{-}H}) = 1.09\) Å. Archived xtb 6.7.1
transcript `tools/tb_oracle/inputs/xtb/ch4_ipea_vip/xtb.out.txt`. Input twin
`mol.xyz`. \(s = 1.09/\sqrt{3}\).

IPEA is selected by pointing `XTBPATH` at a directory whose
`param_gfn1-xtb.txt` is a copy of `param_ipea-xtb.txt` (xtb then prints the
`I P E A - x T B` banner). Command:

```text
XTBPATH=<twin xtbpath> xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --vipea
OMP_NUM_THREADS=1
```

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H | \(s\) | \(s\) | \(s\) |
| H | \(s\) | \(-s\) | \(-s\) |
| H | \(-s\) | \(s\) | \(-s\) |
| H | \(-s\) | \(-s\) | \(s\) |

Cubic box 20 Å (isolated). Neutral: 8 electrons, HOMO–LUMO 7.751 eV,
\(E=\) **−4.670476737294 Eh** (same as `reg_ipea-xtb-small`). Cation (7 e)
occupations \(2.0000 + 1.6667\times 3\) (t2 hole). Anion (9 e)
\(2.0000\times 4 + 1.0000\). Cation total **−4.014341473228 Eh**; anion
**−4.795043035470 Eh**.

## Pin

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| empirical shift | **4.8455** | eV | — |
| \(\mathrm{IP}_v\) | **13.0088** | eV | \(0.15\) eV |
| \(\mathrm{EA}_v\) | **−1.4559** | eV | \(0.15\) eV |
| \(E(N)\) | −4.670476737294 | Eh | \(0.05\) eV |

The 0.05 eV energy window is the `reg_ipea-xtb-small` kernel-residual
floor (D3 CN vs xtb ncoord, Stewart contraction vs `intcut`, Anderson
vs Broyden; measured 8.90 meV). IP/EA are ΔSCC of a **charged**
species: the same kernel hole is amplified by \(\Delta q=\pm 1\)
(measured 0.103 eV IP, 0.125 eV EA; cation t2 occupations match xtb
\(2+1.6667\times 3\)). The 0.15 eV IP/EA window is that ionized
residual plus margin — not a missing occupation or shift term.

Wall (xtb `--vipea`, 1 thread): 0.005 s (xtb clock) / 0.02 s (`/usr/bin/time`).

## Gates

- Finite N-electron energy on `frame.meta[kEnergy]`
- Finite `IPEA::last_vip()` / `last_vea()` when the flags are set
- Hard-coded IP/EA pins vs table (tol \(0.15\) eV); N-electron energy 0.05 eV

## Perf

Fair 1-thread wall of `reg_ipea-xtb-vip` (Release, `OMP_NUM_THREADS=1`):
ATV **0.01 s**, xtb 6.7.1 `--vipea` clock **0.005 s**.
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}} = 2.0 \le 10\). Not `perf_exempt`.

## Gaps

- Extra-Z tables (`ipea-beyond-chno`)
- Polar-molecule residual vs water
- UHF / `spin-gfn-xtb`
- ALPB, PBC, sTDA
