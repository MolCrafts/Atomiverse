# semiempirical/ipea-xtb

One case; every section below is run by `atv_regressions --case semiempirical/ipea-xtb` and keeps its own oracle record.


## CH4 IPEA-xTB energy

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

### Method

IPEA-xTB (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)). GFN1 Hamiltonian
reparametrized for IPs/EAs, with extra \((n+1)s\) functions.

Family-locked numerical oracle: **xtb** (never OpenMOPAC / DFTB+).

Native static in-tree tables. No `libxtb` wrap.

Same tetrahedral CH4 as the GFN1 organic pin, so the two Hamiltonians are
discriminated on one geometry (GFN1 −4.274232224875 Eh; IPEA **−4.670476737294
Eh**).

### Geometry

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

### Pin

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

Measured ATV residual at this geometry (Release, dual-gate after rebase
onto `dev` @ `e276a49d`): **8.90 meV** (`E = −127.08124904` eV vs pin
`−127.09014645` eV). The 0.05 eV window is the same kernel-residual floor
as `reg_gfn1-xtb-organic` (D3 CN vs xtb ncoord, Stewart contraction vs
`intcut`, Anderson vs Broyden).

Wall (xtb, 1 thread): 0.005 s (xtb clock) / 0.02 s (`/usr/bin/time`).

### Gates

- Finite, non-zero total energy [eV] on `frame.meta[kEnergy]`
- SCC residual < `scc_tol`
- Hard-coded energy pin vs table (tol \(0.05\) eV)

### Perf

Fair 1-thread wall of `reg_ipea-xtb-small` (Release, `OMP_NUM_THREADS=1`):
ATV **0.01 s**, xtb 6.7.1 clock **0.005 s**.
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}} = 2.0 \le 10\). Not `perf_exempt`.

### Gaps

- Analytic forces: landed as `reg_ipea-xtb-forces` (this energy pin stays energy-only)
- Elements beyond H/C/N/O
- Open-shell / `--vip` ΔSCC IP+EA (cation/anion)
- Polar-molecule residual vs water (E3 / charge; not this pin)
- ALPB, PBC, sTDA


## Reference — off-eq CH4 IPEA-xTB analytic forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

### Method

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

### Offline oracle run

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

#### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| C | −3.5026444745746e−2 | −3.5026444745746e−2 | −3.5026444745746e−2 |
| H1 (stretched) | +4.0716066407528e−2 | +4.0716066407528e−2 | +4.0716066407528e−2 |
| H2 | −1.7357154065799e−3 | −1.9769531276008e−3 | −1.9769531276009e−3 |
| H3 | −1.9769531276008e−3 | −1.7357154065799e−3 | −1.9769531276008e−3 |
| H4 | −1.9769531276008e−3 | −1.9769531276007e−3 | −1.7357154065798e−3 |

### Geometry

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

### Pins

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

### Perf

xtb wall **0.004 s** (1 thread). ATV wall **1.035 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.26\le 10\). Analytic path is
one SCC + one gradient.

### Gaps

- `--vip` / `--vea` ΔSCC IP/EA (`ipea-xtb-vip`)
- Polar-molecule residual vs water (~0.5 eV; same kernel hole as energy)
- Elements beyond H/C/N/O
- Open shell, PBC, ALPB
- IPEA Hessian (waits on this F)


## CH4 IPEA-xTB vertical IP/EA (ΔSCC)

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

### Method

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

### Geometry

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

### Pin

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

### Gates

- Finite N-electron energy on `frame.meta[kEnergy]`
- Finite `IPEA::last_vip()` / `last_vea()` when the flags are set
- Hard-coded IP/EA pins vs table (tol \(0.15\) eV); N-electron energy 0.05 eV

### Perf

Fair 1-thread wall of `reg_ipea-xtb-vip` (Release, `OMP_NUM_THREADS=1`):
ATV **0.01 s**, xtb 6.7.1 `--vipea` clock **0.005 s**.
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}} = 2.0 \le 10\). Not `perf_exempt`.

### Gaps

- Extra-Z tables (`ipea-beyond-chno`)
- Polar-molecule residual vs water
- UHF / `spin-gfn-xtb`
- ALPB, PBC, sTDA


## Reference — CH4 IPEA-xTB FD Hessian vs xtb

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

### Method

IPEA-xTB nuclear Hessian of the Ásgeirsson 2017 SCC energy, assembled as a
**6N central finite difference of the already-shipped analytic gradient**
(\(h = 10^{-3}\) Å, `cpu::vib::Hessian` + `cpu::semiempirical::IPEA` with
`with_forces = true`). Not a closed-form \(\partial^2 E/\partial R_i\partial R_j\)
kernel; not GFN1/GFN2; not `libxtb`. Polar water is **not** this pin
(0.5 eV kernel hole vs xtb; named in `reg_ipea-xtb-small`).

Reference: Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
10.1039/c7sc00601b. xtb 6.7.1 `--hess` with `param_ipea` as GFN1 prints an
**analytical** Hessian (xtb log: “analytical hessian”); ATV stays FD of
analytic \(G\) (same shape as GFN0/1/2). **Not** DFTB+; **Not** OpenMOPAC.

### Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host, compiled
2026-06-09). IPEA is selected by pointing `XTBPATH` at a directory whose
`param_gfn1-xtb.txt` is a copy of `param_ipea-xtb.txt` (xtb prints
`I P E A - x T B`). Command:

```text
XTBPATH=... OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess
```

Twin: `tools/tb_oracle/inputs/xtb/ch4_ipea_hess/`
(`mol.xyz` + `xtb.out.txt` + `hessian.txt` + `vibspectrum.txt`). Nothing is
executed at ctest time. xtb prints \(H\) [Eh/a0²] and projected ω [cm⁻¹];
ATV stores \(H\) [eV/Å²] with CODATA \(a_0\) /
\(1\,\mathrm{Ha}=27.211386245988\) eV from `units.h`.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−4.670476737294** | Eh |
| \(\omega\) (projected, 9 vib) | **1541.68×3, 1705.89×2, 3548.78, 3576.13×3** | cm⁻¹ |
| \(H_{C_x,C_x}\) | **0.6813937129** | Eh/a0² |
| xtb Hessian path | analytical | — |
| Wall (xtb printed, 1 thread) | 0.170 | s |
| Wall (`/usr/bin/time`) | 0.43 | s |

xtb reports Td, \(n_{\mathrm{imag}}=0\), gradient norm \(7.87\times10^{-4}\)
Eh/\(a_0\) (same tetrahedral CH4 as `reg_ipea-xtb-small`). All nine
vibrational eigenvalues are real.

### Geometry

Gas-phase methane, charge 0, singlet. Cubic box 20 Å (isolation only).
Same Cartesian as `reg_ipea-xtb-small` (\(s=1.09/\sqrt{3}\)).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H | \(s\) | \(s\) | \(s\) |
| H | \(s\) | \(-s\) | \(-s\) |
| H | \(-s\) | \(s\) | \(-s\) |
| H | \(-s\) | \(-s\) | \(s\) |

### Pins

\(H[\mathrm{eV}/\mathrm{Å}^2] = H[\mathrm{Eh}/a_0^2]\times
27.211386245988 / (0.52917721092)^2\) (`units::kBohrInAngstrom`).

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−4.670476737294** | Eh | \(5\times10^{-2}\) eV |
| \(\omega\) T2 bend (×3) | **1541.68** | cm⁻¹ | 10 |
| \(\omega\) E bend (×2) | **1705.89** | cm⁻¹ | 10 |
| \(\omega\) A1 stretch | **3548.78** | cm⁻¹ | 10 |
| \(\omega\) T2 stretch (×3) | **3576.13** | cm⁻¹ | 10 |
| \(H_{C_x,C_x}\) | **66.21350** | eV/Å² | 0.2 |

**Tolerance rationale.** Energy floor matches `reg_ipea-xtb-small`
(8.90 meV kernel hole: D3 C6 interpolation, Stewart `intcut`, Anderson vs
Broyden). Frequency / Cartesian-\(H\) tols absorb FD-step vs xtb
**analytical** Hessian plus IUPAC-2013 vs xtb isotope masses. Structural
gates: 15×15 finite symmetric Hessian, nonlinear rotor, 9 real modes,
\(n_{\mathrm{imag}}=0\).

Measured on the pin host (Release, `OMP_NUM_THREADS=1`):
\(|E_{\mathrm{ATV}}-E_{\mathrm{pin}}|=8.90\) meV,
\(\Delta\omega = 0.36/0.64/1.30/0.26\) cm⁻¹ (T2-bend / E / A1 / T2-stretch;
Td degeneracies unsplit),
\(|H_{C_x,C_x}-H_{\mathrm{pin}}|=1.60\times10^{-2}\) eV/Å².

### Perf

xtb printed wall **0.170 s** (1 thread). `/usr/bin/time` process wall
**0.43 s**. ATV `Hessian::run` wall **10.4 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.061\le 10\) (vs printed wall).

### Gaps

- `--vip` / `--vea` ΔSCC IP/EA (`ipea-xtb-vip`)
- Polar-molecule residual vs water (~0.5 eV; same kernel hole as energy)
- Elements beyond H/C/N/O
- Open shell, PBC, ALPB
- Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
