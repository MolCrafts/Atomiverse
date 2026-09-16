# Reference — water GFN0-xTB FD Hessian vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

## Method

GFN0-xTB nuclear Hessian of the Pracht 2019 non-SCC energy, assembled as a
**6N central finite difference of the already-shipped analytic gradient**
(\(h = 10^{-3}\) Å, `cpu::vib::Hessian` + `cpu::semiempirical::GFN0` with
`with_forces = true`). Not a closed-form \(\partial^2 E/\partial R_i\partial R_j\)
kernel; not GFN1/2; not `libxtb`.

Reference: Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv (2019),
DOI 10.26434/chemrxiv.8326202.v1. xtb 6.7.1 `--hess` is the same numerical
shape (analytic G, numerical H; step 0.005 Bohr). **Not** DFTB+; **Not**
OpenMOPAC.

## Offline oracle run

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

## Geometry

Gas-phase water, charge 0, singlet. Cubic box 20 Å (isolation only).
Same Cartesian as `reg_gfn0-xtb-forces`.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 0.96  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

## Pins

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

## Perf

xtb process wall **0.02 s** (1 thread; `/usr/bin/time` of `--hess`). ATV
`Hessian::run` wall **~1.2 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.06\le 10\).
