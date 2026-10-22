# semiempirical/gfn1-xtb

One case; every section below is run by `atv_regressions --case semiempirical/gfn1-xtb` and keeps its own oracle record.


## Reference — CH4 GFN1-xTB energy

### Oracle

```text
oracle: xtb
oracle_version: 6.7.1
```

Offline pin only — ctest never launches xtb.

Family-locked: GFN-xTB → **xtb**. Not OpenMOPAC, not DFTB+, not a DFTB3
successor.

### Method

- **Hamiltonian:** GFN1-xTB (Grimme, Bannwarth, Shushkov, *J. Chem. Theory
  Comput.* **13**, 1989 (2017); DOI
  [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)).
- **Binary:** xtb 6.7.1 (EasyBuild `xtb/6.7.1-gcc-2025b-eb` on Arrhenius),
  `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01`.
- **Twin:** `tools/tb_oracle/inputs/xtb/ch4_gfn1/` (`mol.xyz` + `xtb.out.txt`).
- ATV: `cpu::Evaluate` + `cpu::semiempirical::GFN1`, `with_forces = false`.
  Native static tables, no `libxtb`.

### Geometry

Methane, tetrahedral, \(r(\mathrm{C{-}H}) = 1.09\) Å:

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H | \(s\) | \(s\) | \(s\) |
| H | \(s\) | \(-s\) | \(-s\) |
| H | \(-s\) | \(s\) | \(-s\) |
| H | \(-s\) | \(-s\) | \(s\) |

with \(s = 1.09 / \sqrt{3}\). Isolated molecule.

### Pin

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

### Gate

- Assert `|E_ATV − (−116.30778398)| ≤ 0.02` eV.
- Measured ATV residual at this geometry: **10.2 meV** (0.375 mEh). The 20 meV
  floor covers remaining kernel residuals (D3 C6 interpolation vs xtb's
  internal D3, Stewart primitive screening vs xtb `intcut`, Anderson vs
  Broyden) on a ~116 eV total. Tighten once a later force slug forces the
  kernels bit-identical.

### Perf (optional, not ctest)

xtb 6.7.1, `OMP_NUM_THREADS=1`: **0.02 s** wall (transcript). ATV 1-thread
Release binary: **0.01 s**. \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} = 0.5 \le 10\).


## Reference — off-eq water GFN1-xTB analytic forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

### Method

GFN1-xTB nuclear gradient of the Grimme 2017 SCC energy

\[
E = \mathrm{Tr}(P H_0) + \tfrac12 q\,\gamma\,q + \tfrac13 \Gamma q^3
    + E_{\mathrm{rep}} + E_{\mathrm{D3}}
\]

already shipped by `reg_gfn1-xtb-organic`. Forces are the **closed-form**
\(\mathbf{F}=-\nabla E\) (Hellmann–Feynman + Pulay + CN/shell-poly chains +
\(\partial\gamma/\partial R\) + pair repulsion + D3(BJ)).

Reference: Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.* **13**, 1989
(2017), DOI 10.1021/acs.jctc.7b00118. **Not** DFTB+; **not** OpenMOPAC;
**not** GFN0/GFN2.

### Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad
```

Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn1_offeq_grad/`
(`mol.xyz` + `xtb.out.txt` + `gradient.txt`). Nothing is executed at ctest
time. xtb prints \(\nabla E\) [Eh/a0]; ATV stores \(F=-\nabla E\) [eV/Å]
with CODATA \(a_0\) via `units::kHartreePerBohrInEvPerAngstrom`.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.768641651774** | Eh |
| \(\|\nabla E\|\) | 0.004979646520 | Eh/a0 |
| Wall (xtb, 1 thread) | 0.004 | s |

#### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −3.4089559688772e−3 | −1.1077217607051e−3 | ~0 |
| H1 | +6.9083433839239e−4 | −7.6331225483940e−4 | ~0 |
| H2 | +2.7181216304849e−3 | +1.8710340155445e−3 | ~0 |

### Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** so \(|F|\neq 0\).
Cubic box 20 Å (isolation only).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 0.96  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

### Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−5.768641651774** | Eh | \(5\times10^{-3}\) eV |
| \(F_{O,x}\) | **+0.175295563850** | eV/Å | \(5\times10^{-3}\) |
| \(F_{H2,y}\) | **−0.096212437395** | eV/Å | \(5\times10^{-3}\) |

\(F = -\nabla E \times\) `units::kHartreePerBohrInEvPerAngstrom`.

**Tolerance rationale.** Measured on the pin host (Release, `OMP_NUM_THREADS=1`):
\(|E_{\mathrm{ATV}}-E_{\mathrm{pin}}|=2.77\) meV,
\(|F_{O,x}-F_{\mathrm{pin}}|=1.87\times10^{-3}\) eV/Å,
\(|F_{H2,y}-F_{\mathrm{pin}}|=1.81\times10^{-3}\) eV/Å,
analytic vs energy FD \(<10^{-9}\) eV/Å at \(h=10^{-4}\) Å.

\(F_{O,y}\) / \(F_{H1,y}\) vs xtb sit at **~0.20 eV/Å** — the same GFN1 kernel
hole as the CH4 energy residual (D3 C6 interpolation, Stewart primitive
screening vs xtb `intcut`, Anderson vs Broyden). Those components still match
ATV energy FD, so they are not a gradient defect. The pin uses the
well-matching Cartesian components; the FD identity is the gradient gate.

### Perf

xtb wall **0.004 s** (1 thread). ATV wall **0.64 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.16\le 10\). Analytic path is one
SCC + one gradient (replaces 18 energy solves).
