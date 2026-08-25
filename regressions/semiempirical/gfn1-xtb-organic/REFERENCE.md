# Reference — CH4 GFN1-xTB energy

## Oracle

```text
oracle: xtb
oracle_version: 6.7.1
```

Offline pin only — ctest never launches xtb.

Family-locked: GFN-xTB → **xtb**. Not OpenMOPAC, not DFTB+, not a DFTB3
successor.

## Method

- **Hamiltonian:** GFN1-xTB (Grimme, Bannwarth, Shushkov, *J. Chem. Theory
  Comput.* **13**, 1989 (2017); DOI
  [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)).
- **Binary:** xtb 6.7.1 (EasyBuild `xtb/6.7.1-gcc-2025b-eb` on Arrhenius),
  `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01`.
- **Twin:** `tools/tb_oracle/inputs/xtb/ch4_gfn1/` (`mol.xyz` + `xtb.out.txt`).
- ATV: `cpu::Evaluate` + `cpu::semiempirical::GFN1`, `with_forces = false`.
  Native static tables, no `libxtb`.

## Geometry

Methane, tetrahedral, \(r(\mathrm{C{-}H}) = 1.09\) Å:

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H | \(s\) | \(s\) | \(s\) |
| H | \(s\) | \(-s\) | \(-s\) |
| H | \(-s\) | \(s\) | \(-s\) |
| H | \(-s\) | \(-s\) | \(s\) |

with \(s = 1.09 / \sqrt{3}\). Isolated molecule.

## Pin

xtb 6.7.1 printed (line `TOTAL ENERGY` of `xtb.out.txt`):

| Quantity | Value | Units |
|----------|-------|-------|
| Total energy (xtb) | **−4.274232224875** | Ha |
| Total energy (public ATV) | **−116.30778398** | eV |

Conversion: \(E[\mathrm{eV}] = E[\mathrm{Ha}] \times 27.211386245988\) (ATV
CODATA). xtb components (documentation only):

| Term | Ha |
|------|-----|
| SCC (electronic) | −4.340066348203 |
| → electrostatic | 0.000089060808 |
| repulsion | 0.066288872935 |
| dispersion | −0.000454749608 |
| halogen | 0 |
| **Total** | **−4.274232224875** |

12 AOs (C 2s2p + 4×H 1s2s), 8 electrons, closed shell, 300 K electronic
temperature (integer occupation on this gap). 1-thread wall **0.02 s**.

## Gate

- Assert `|E_ATV − (−116.30778398)| ≤ 0.05` eV.
- The 0.05 eV window covers remaining kernel residuals (D3 CN vs xtb ncoord,
  Stewart contraction vs xtb's `intcut`, Anderson vs Broyden) on a ~116 eV
  total. Tighten once a later force slug forces the kernels bit-identical.

## Perf (optional, not ctest)

xtb 6.7.1, `OMP_NUM_THREADS=1`: **0.02 s** wall for this molecule. ATV is a
12×12 dense eigenproblem; expected well under 10×.
