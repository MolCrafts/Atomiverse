# Reference — methanol GFN0-xTB energy vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

## Method

GFN0-xTB (non-self-consistent first-order xTB):

1. Erf coordination number (k=7.5, cut maxCN=8)
2. Electronegativity-equilibration (EEQ) charges and isotropic electrostatics
3. CN- and charge-shifted EHT Hamiltonian; Slater overlap via STO-nG
4. One generalized eigenproblem \(H C = S C\varepsilon\), integer occupation
5. Classical repulsion, two-body D4(BJ) with GFN0 damping, polar-bond SRB

Reference: Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv (2019),
DOI 10.26434/chemrxiv.8326202.v1. **Not** DFTB+; **not** OpenMOPAC;
**not** GFN2 (the on-disk `ch3oh_gfn2` transcript is a different Hamiltonian).

## Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host; compiled
2026-06-09). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0
```

Twin: `tools/tb_oracle/inputs/xtb/ch3oh_gfn0/` (`mol.xyz` + `xtb.out.txt`).
Nothing is executed at ctest time.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−7.597234498685** | Eh |
| H0 (EHT) | −7.632347129465 | Eh |
| Repulsion | +0.079267751055 | Eh |
| Electrostatics (IES) | −0.030781820638 | Eh |
| Dispersion (D4) | −0.000545824477 | Eh |
| Short-range corr. | −0.012827475160 | Eh |
| Wall (xtb, 1 thread) | 0.009 | s |

## Geometry

Gas-phase methanol, charge 0, singlet. Cubic box 30 Å (isolation only).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C    | 0.0000 | 0.0000 | 0.0000 |
| O    | 1.4270 | 0.0000 | 0.0000 |
| H    | −0.3639 | 1.0297 | 0.0000 |
| H    | −0.3639 | −0.51485 | 0.89175 |
| H    | −0.3639 | −0.51485 | −0.89175 |
| H    | 1.7505 | 0.0000 | 0.9059 |

## Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−7.597234498685** | Eh | \(1\times10^{-3}\) eV |

**Tolerance rationale.** \(1\times10^{-3}\) eV is the family floor used on
absolute TB energy pins in this loop. xtb prints \(10^{-12}\) Eh
(\(\approx 3\times10^{-11}\) eV). Observed residual on the pin host:
\(|E_{\mathrm{ATV}}-E_{\mathrm{pin}}| \approx 3.6\times10^{-8}\) eV
(\(1.3\times10^{-12}\) Eh).

## Perf

xtb wall **0.009 s** (1 thread, cold). ATV wall **0.58 ms** on the same
geometry in the regression binary → \(T_{\mathrm{atv}}/T_{\mathrm{xtb}}
\approx 0.065 \le 10\).
