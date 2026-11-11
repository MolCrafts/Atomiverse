# semiempirical/gfn0-xtb

One case; every section below is run by `atv_regressions --case semiempirical/gfn0-xtb` and keeps its own oracle record.


## Reference — methanol GFN0-xTB energy vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

### Method

GFN0-xTB (non-self-consistent first-order xTB):

1. Erf coordination number (k=7.5, cut maxCN=8)
2. Electronegativity-equilibration (EEQ) charges and isotropic electrostatics
3. CN- and charge-shifted EHT Hamiltonian; Slater overlap via STO-nG
4. One generalized eigenproblem \(H C = S C\varepsilon\), integer occupation
5. Classical repulsion, two-body D4(BJ) with GFN0 damping, polar-bond SRB

Reference: Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv (2019),
DOI 10.26434/chemrxiv.8326202.v1. **Not** DFTB+; **not** OpenMOPAC;
**not** GFN2 (the on-disk `ch3oh_gfn2` transcript is a different Hamiltonian).

### Offline oracle run

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

### Geometry

Gas-phase methanol, charge 0, singlet. Cubic box 30 Å (isolation only).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C    | 0.0000 | 0.0000 | 0.0000 |
| O    | 1.4270 | 0.0000 | 0.0000 |
| H    | −0.3639 | 1.0297 | 0.0000 |
| H    | −0.3639 | −0.51485 | 0.89175 |
| H    | −0.3639 | −0.51485 | −0.89175 |
| H    | 1.7505 | 0.0000 | 0.9059 |

### Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−7.597234498685** | Eh | \(1\times10^{-3}\) eV |

**Tolerance rationale.** \(1\times10^{-3}\) eV is the family floor used on
absolute TB energy pins in this loop. xtb prints \(10^{-12}\) Eh
(\(\approx 3\times10^{-11}\) eV). Observed residual on the pin host:
\(|E_{\mathrm{ATV}}-E_{\mathrm{pin}}| \approx 3.6\times10^{-8}\) eV
(\(1.3\times10^{-12}\) Eh).

### Perf

xtb wall **0.009 s** (1 thread, cold). ATV wall **0.58 ms** on the same
geometry in the regression binary → \(T_{\mathrm{atv}}/T_{\mathrm{xtb}}
\approx 0.065 \le 10\).


## Reference — off-eq water GFN0-xTB analytic forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

### Method

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

### Offline oracle run

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

#### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −3.4755123224283e−4 | −9.4449534348479e−4 | ~0 |
| H1 | −3.8500960780473e−3 | +3.4420757341719e−3 | ~0 |
| H2 | +4.1976473102901e−3 | −2.4975803906871e−3 | ~0 |

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

### Perf

xtb wall **0.007 s** (1 thread). ATV wall **0.645 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.092\le 10\). Analytic path is
one EEQ + one diagonalization (replaces 18 energy solves).


## Reference — water GFN0-xTB FD Hessian vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

### Method

GFN0-xTB nuclear Hessian of the Pracht 2019 non-SCC energy, assembled as a
**6N central finite difference of the already-shipped analytic gradient**
(\(h = 10^{-3}\) Å, `cpu::vib::Hessian` + `cpu::semiempirical::GFN0` with
`with_forces = true`). Not a closed-form \(\partial^2 E/\partial R_i\partial R_j\)
kernel; not GFN1/2; not `libxtb`.

Reference: Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv (2019),
DOI 10.26434/chemrxiv.8326202.v1. xtb 6.7.1 `--hess` is the same numerical
shape (analytic G, numerical H; step 0.005 Bohr). **Not** DFTB+; **Not**
OpenMOPAC.

### Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0 --hess
```

Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn0_hess/`
(`mol.xyz` + `xtb.out.txt` + `hessian.txt` + `vibspectrum.txt`). Nothing is
executed at ctest time. xtb prints \(H\) [Eh/a0²] and projected ω [cm⁻¹];
ATV stores \(H\) [eV/Å²] with \(a_0=0.52917726\) Å (GFN0) and
\(1\,\mathrm{Ha}=27.211386245988\) eV.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−4.366816412151** | Eh |
| \(\omega\) (projected) | **1563.14, 3497.87, 3505.03** | cm⁻¹ |
| \(H_{O_x,O_x}\) | **0.4793058695** | Eh/a0² |
| xtb Hessian step | 0.005 | Bohr |
| Wall (xtb printed, 1 thread) | 0.007 | s |
| Wall (`/usr/bin/time`) | 0.02 | s |

xtb reports “incompletely optimized geometry” (same off-eq water as
`reg_gfn0-xtb-forces`); all three vibrational eigenvalues are real.

### Geometry

Gas-phase water, charge 0, singlet. Cubic box 20 Å (isolation only).
Same Cartesian as `reg_gfn0-xtb-forces`.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 0.96  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

### Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−4.366816412151** | Eh | \(1\times10^{-3}\) eV |
| \(\omega_1\) (bend) | **1563.14** | cm⁻¹ | 10 |
| \(\omega_2\) (sym stretch) | **3497.87** | cm⁻¹ | 10 |
| \(\omega_3\) (asym stretch) | **3505.03** | cm⁻¹ | 10 |
| \(H_{O_x,O_x}\) | **46.57588** | eV/Å² | 0.2 |

\(H[\mathrm{eV}/\mathrm{Å}^2] = H[\mathrm{Eh}/a_0^2]\times 27.211386245988 / (0.52917726)^2\).

**Tolerance rationale.** Energy floor is the family \(10^{-3}\) eV TB pin
(same geometry as `reg_gfn0-xtb-forces`). Frequency / Cartesian-\(H\) tols
absorb the FD-step mismatch (xtb \(0.005\,a_0 \approx 0.00265\) Å vs ATV
\(10^{-3}\) Å) plus IUPAC-2013 vs xtb isotope masses (\(\sim 0.3\,\mathrm{cm}^{-1}\)
on H stretches). Measured on the pin host (Release, `OMP_NUM_THREADS=1`):
\(|E_{\mathrm{ATV}}-E_{\mathrm{pin}}|=3.4\times10^{-8}\) eV,
\(\Delta\omega = 0.05/2.73/1.87\) cm⁻¹,
\(|H_{O_x,O_x}-H_{\mathrm{pin}}|=3.8\times10^{-2}\) eV/Å².
Structural gates: 9×9 finite symmetric Hessian, nonlinear rotor, 3 real
modes, \(n_{\mathrm{imag}}=0\).

### Perf

xtb process wall **0.02 s** (1 thread; `/usr/bin/time` of `--hess`). ATV
`Hessian::run` wall **~1.2 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.06\le 10\).
