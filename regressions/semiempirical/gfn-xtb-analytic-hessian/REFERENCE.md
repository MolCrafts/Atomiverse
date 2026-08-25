# Reference — water GFN1/GFN2-xTB FD Hessian vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

## Method

GFN1-xTB and GFN2-xTB nuclear Hessian of the Grimme 2017 / Bannwarth 2019
SCC energies, assembled as a **6N central finite difference of the
already-shipped analytic gradient** (\(h = 10^{-3}\) Å,
`cpu::vib::Hessian` + `cpu::semiempirical::GFN1` / `GFN2` with
`with_forces = true`). Not a closed-form \(\partial^2 E/\partial R_i\partial R_j\)
kernel; not GFN0 (already `reg_gfn0-xtb-analytic-hessian`); not `libxtb`.

References: Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.* **13**,
1989 (2017), DOI 10.1021/acs.jctc.7b00118; Bannwarth, Ehlert, Grimme,
*J. Chem. Theory Comput.* **15**, 1652 (2019), DOI 10.1021/acs.jctc.8b01176.
xtb 6.7.1 `--gfn 1 --hess` / `--gfn 2 --hess` prints an **analytical**
Hessian (xtb log: “analytical hessian”); ATV stays FD of analytic \(G\)
(same shape as GFN0). **Not** DFTB+; **Not** OpenMOPAC.

Primary pin is **GFN1**. GFN2 AES remains a **post-SCC energy** (angular
\(F\) hole; queue `gfn2-aes-fock`); the GFN2 ω residual carries that hole.

## Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess
# and --gfn 2 on the same xyz
```

Twins: `tools/tb_oracle/inputs/xtb/h2o_gfn1_hess/` and
`tools/tb_oracle/inputs/xtb/h2o_gfn2_hess/`
(`mol.xyz` + `xtb.out.txt` + `hessian.txt` + `vibspectrum.txt`). Nothing
is executed at ctest time. xtb prints \(H\) [Eh/a0²] and projected ω
[cm⁻¹]; ATV stores \(H\) [eV/Å²] with CODATA \(a_0\) /
\(1\,\mathrm{Ha}=27.211386245988\) eV from `units.h`.

### GFN1 (`--gfn 1 --hess`)

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.768641651774** | Eh |
| \(\omega\) (projected) | **1474.96, 3657.78, 3761.56** | cm⁻¹ |
| \(H_{O_x,O_x}\) | **0.5466593694** | Eh/a0² |
| xtb Hessian path | analytical | — |
| Wall (xtb printed, 1 thread) | 0.020 | s |
| Wall (`/usr/bin/time`) | 0.03 | s |

### GFN2 (`--gfn 2 --hess`)

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.070383095219** | Eh |
| \(\omega\) (projected) | **1565.10, 3634.95, 3643.62** | cm⁻¹ |
| \(H_{O_x,O_x}\) | **0.5210291876** | Eh/a0² |
| xtb Hessian path | analytical | — |
| Wall (xtb printed, 1 thread) | 0.012 | s |
| Wall (`/usr/bin/time`) | 0.02 | s |

xtb reports “incompletely optimized geometry” (same 0.96 Å water as
`reg_gfn0-xtb-analytic-hessian` / `reg_gfn1-xtb-forces`); all three
vibrational eigenvalues are real. GFN2 energy matches
`reg_gfn2-xtb-water` (this geometry is **not** the 1.10 Å stretch of
`reg_gfn2-xtb-forces`).

## Geometry

Gas-phase water, charge 0, singlet. Cubic box 20 Å (isolation only).
Same Cartesian as `reg_gfn1-xtb-forces` / `reg_gfn0-xtb-analytic-hessian`.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 0.96  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

## Pins

\(H[\mathrm{eV}/\mathrm{Å}^2] = H[\mathrm{Eh}/a_0^2]\times
27.211386245988 / (0.529177210903)^2\).

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| GFN1 \(E\) | **−5.768641651774** | Eh | \(5\times10^{-3}\) eV |
| GFN1 \(\omega_1\) (bend) | **1474.96** | cm⁻¹ | 10 |
| GFN1 \(\omega_2\) (sym stretch) | **3657.78** | cm⁻¹ | 10 |
| GFN1 \(\omega_3\) (asym stretch) | **3761.56** | cm⁻¹ | 10 |
| GFN1 \(H_{O_x,O_x}\) | **53.12087** | eV/Å² | 0.2 |
| GFN2 \(E\) | **−5.070383095219** | Eh | \(1\times10^{-3}\) eV |
| GFN2 \(\omega\) | **1565.10, 3634.95, 3643.62** | cm⁻¹ | 150 |
| GFN2 \(H_{O_x,O_x}\) | **50.63029** | eV/Å² | 1.0 |

**Tolerance rationale.** Energy floors match the family TB pins
(`reg_gfn1-xtb-forces` 5 meV; `reg_gfn2-xtb-water` 1 meV). GFN1
frequency / Cartesian-\(H\) tols absorb FD-step vs xtb **analytical**
Hessian plus IUPAC-2013 vs xtb isotope masses (the GFN1 angular-\(F\)
kernel hole does **not** move these ω). GFN2 ω / \(H\) tols are the
stated **AES-in-Fock hole** (AES still post-SCC; the water bend is the
angular mode). Structural gates: 9×9 finite symmetric Hessian,
nonlinear rotor, 3 real modes, \(n_{\mathrm{imag}}=0\).

Measured on the pin host (Release, `OMP_NUM_THREADS=1`):

| | GFN1 | GFN2 |
| - | ---- | ---- |
| \(\|E_{\mathrm{ATV}}-E_{\mathrm{pin}}\|\) | \(2.767\times10^{-3}\) eV | \(2.64\times10^{-5}\) eV |
| \(\Delta\omega\) | 0.38 / 0.40 / 5.48 cm⁻¹ | 121 / 89 / 19 cm⁻¹ |
| \(\|H_{O_x,O_x}-H_{\mathrm{pin}}\|\) | \(0.104\) eV/Å² | \(0.751\) eV/Å² |

## Perf

xtb GFN1 process wall **0.03 s** (1 thread; `/usr/bin/time` of `--hess`).
ATV `Hessian::run` wall **~3.2 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.11\le 10\).
xtb GFN2 process wall **0.02 s**; ATV **~1.4 ms** → ratio **≈ 0.07**.
