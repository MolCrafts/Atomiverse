# Reference — off-eq CH4 IPEA-xTB analytic forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

## Method

IPEA-xTB nuclear gradient of the Ásgeirsson 2017 SCC energy

\[
E = \mathrm{Tr}(P H_0) + \tfrac12 q\,\gamma\,q + \tfrac13 \Gamma q^3
    + E_{\mathrm{rep}} + E_{\mathrm{D3}}
\]

already shipped by `reg_ipea-xtb-small`. Forces are the **closed-form**
\(\mathbf{F}=-\nabla E\) (Hellmann–Feynman + Pulay + CN/shell-poly chains +
\(\partial\gamma/\partial R\) + pair repulsion + D3(BJ)). Same GFN1
gradient shape, IPEA numbers: \(g_{\exp}=1\), extra \((n+1)s\) on C/N/O,
no H–H / N–H pair \(k\).

Reference: Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
10.1039/c7sc00601b. **Not** DFTB+; **not** OpenMOPAC; **not** GFN1/GFN2.

Off-eq **water** is not this pin. Polar Δq still carries a ~0.5 eV kernel
hole vs xtb (named in `reg_ipea-xtb-small` Gaps). Analytic \(F\) on water
is gated by ATV energy FD in `test_cpu_xtb_ipea`. This product pin uses
the same CH4 that already matches xtb to 8.90 meV, with one C–H stretched.

## Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host, compiled
2026-06-09). IPEA is selected by pointing `XTBPATH` at a directory whose
`param_gfn1-xtb.txt` is a copy of `param_ipea-xtb.txt` (xtb prints
`I P E A - x T B`). Command:

```text
XTBPATH=... OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad
```

Twin: `tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/`
(`mol.xyz` + `xtb.out.txt` + `gradient.txt`). Nothing is executed at ctest
time. xtb prints \(\nabla E\) [Eh/a0]; ATV stores \(F=-\nabla E\) [eV/Å]
with CODATA \(a_0\) via `units::kHartreePerBohrInEvPerAngstrom`.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−4.662477307157** | Eh |
| \(\|\nabla E\|\) | 0.093201061537 | Eh/a0 |
| Wall (xtb, 1 thread) | 0.004 | s |

xtb pieces (same transcript): SCC −4.745157981515 Eh, electrostatic
0.000977581394 Eh, repulsion 0.083136092009 Eh, dispersion −0.000455417651
Eh.

### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| C | −3.5026444745746e−2 | −3.5026444745746e−2 | −3.5026444745746e−2 |
| H1 (stretched) | +4.0716066407528e−2 | +4.0716066407528e−2 | +4.0716066407528e−2 |
| H2 | −1.7357154065799e−3 | −1.9769531276008e−3 | −1.9769531276009e−3 |
| H3 | −1.9769531276008e−3 | −1.7357154065799e−3 | −1.9769531276008e−3 |
| H4 | −1.9769531276008e−3 | −1.9769531276007e−3 | −1.7357154065798e−3 |

## Geometry

Gas-phase methane, charge 0, singlet. Same tetrahedral CH4 as
`reg_ipea-xtb-small` (\(s=1.09/\sqrt{3}\)) except H1 scaled so
\(r(\mathrm{C{-}H1})=1.20\) Å. Cubic box 20 Å (isolation only).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H1 | \(s\cdot 1.20/1.09\) | same | same |
| H2 | \(s\) | \(-s\) | \(-s\) |
| H3 | \(-s\) | \(s\) | \(-s\) |
| H4 | \(-s\) | \(-s\) | \(s\) |

## Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−4.662477307157** | Eh | \(5\times10^{-2}\) eV |
| \(F_{C,x}\) | **+1.801132205114** | eV/Å | \(5\times10^{-3}\) |
| \(F_{H1,x}\) | **−2.093704314111** | eV/Å | \(5\times10^{-3}\) |

\(F = -\nabla E \times\) `units::kHartreePerBohrInEvPerAngstrom`.

**Tolerance rationale.** Measured on the pin host (Release,
`OMP_NUM_THREADS=1`): \(|E_{\mathrm{ATV}}-E_{\mathrm{pin}}|=8.49\) meV
(same kernel hole as eq-CH4 8.90 meV),
\(|F_{C,x}-F_{\mathrm{pin}}|=5.12\times10^{-4}\) eV/Å,
\(|F_{H1,x}-F_{\mathrm{pin}}|=1.86\times10^{-3}\) eV/Å,
analytic vs energy FD \(<2\times10^{-8}\) eV/Å at \(h=10^{-4}\) Å.

The 0.05 eV energy window is the `reg_ipea-xtb-small` kernel-residual floor
(D3 C6 interpolation, Stewart `intcut`, Anderson vs Broyden). The xtb pin
uses the stretched-bond Cartesian components (`F_{C,x}`, `F_{H1,x}`) on
the (111) axis.

Discriminates IPEA from GFN1: eq CH4 GFN1 is −4.274232224875 Eh vs IPEA
−4.670476737294 Eh.

## Perf

xtb wall **0.004 s** (1 thread). ATV wall **1.035 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.26\le 10\). Analytic path is
one SCC + one gradient.

## Gaps

- `--vip` / `--vea` ΔSCC IP/EA (`ipea-xtb-vip`)
- Polar-molecule residual vs water (~0.5 eV; same kernel hole as energy)
- Elements beyond H/C/N/O
- Open shell, PBC, ALPB
- IPEA Hessian (waits on this F)
