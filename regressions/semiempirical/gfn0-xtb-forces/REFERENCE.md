# Reference — off-eq water GFN0-xTB analytic forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

## Method

GFN0-xTB nuclear gradient of the Pracht 2019 non-SCC energy

\[
E = E_{\mathrm{H0}} + E_{\mathrm{rep}} + E_{\mathrm{IES}} + E_{\mathrm{D4}} + E_{\mathrm{SRB}}
\]

already shipped by `reg_gfn0-beyond-toy`. Forces are the **closed-form**
\(\mathbf{F}=-\nabla E\) (Hellmann–Feynman + Pulay + EEQ/CN chains + pair
repulsion + two-body D4 + SRB). The 6N central-FD product path is deleted.

Reference: Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv (2019),
DOI 10.26434/chemrxiv.8326202.v1. Gradient module boundary: pprcht/gfn0
(`build_dSH0`, `chargeEquilibration`, `gfn0_repulsion`, `gfn0_shortranged`).
**Not** DFTB+; **not** OpenMOPAC; **not** GFN2.

## Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0 --grad
```

Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn0_offeq_grad/`
(`mol.xyz` + `xtb.out.txt` + `gradient.txt`). Nothing is executed at ctest
time. xtb prints \(\nabla E\) [Eh/a0]; ATV stores \(F=-\nabla E\) [eV/Å]
with \(a_0=0.52917726\) Å (GFN0) and \(1\,\mathrm{Ha}=27.211386245988\) eV.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−4.366816412151** | Eh |
| \(\|\nabla E\|\) | 0.007179285426 | Eh/a0 |
| Wall (xtb, 1 thread) | 0.007 | s |

### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −3.4755123224283e−4 | −9.4449534348479e−4 | ~0 |
| H1 | −3.8500960780473e−3 | +3.4420757341719e−3 | ~0 |
| H2 | +4.1976473102901e−3 | −2.4975803906871e−3 | ~0 |

## Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** so \(|F|\neq 0\).
Cubic box 20 Å (isolation only).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 0.96  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

## Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−4.366816412151** | Eh | \(1\times10^{-3}\) eV |
| \(F_{O,x}\) | **+0.017871801258** | eV/Å | \(1\times10^{-4}\) |
| \(F_{H1,y}\) | **−0.176998634239** | eV/Å | \(1\times10^{-4}\) |

\(F = -\nabla E \times (27.211386245988\,\mathrm{eV/Eh}) / 0.52917726\,\mathrm{Å}\).

**Tolerance rationale.** Energy floor is the family \(10^{-3}\) eV TB pin.
Force tol \(10^{-4}\) eV/Å is well above xtb's printed \(10^{-16}\) Eh/a0
noise on \(F_z\). Measured on the pin host (Release, `OMP_NUM_THREADS=1`):
\(|E_{\mathrm{ATV}}-E_{\mathrm{pin}}|=3.4\times10^{-8}\) eV,
\(|F_{O,x}-F_{\mathrm{pin}}|=2.6\times10^{-6}\) eV/Å,
analytic vs energy FD \(6.5\times10^{-7}\) eV/Å at \(h=10^{-4}\) Å.

Water has no SRB term (H is outside Z∈[5,9]). Methanol SRB + C–O is covered
by the energy pin `reg_gfn0-beyond-toy` plus the unit FD identity on this
off-eq water (EHT + EEQ + repulsion + D4).

## Perf

xtb wall **0.007 s** (1 thread). ATV wall **0.645 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.092\le 10\). Analytic path is
one EEQ + one diagonalization (replaces 18 energy solves).
