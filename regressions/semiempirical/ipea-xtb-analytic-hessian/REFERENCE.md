# Reference — CH4 IPEA-xTB FD Hessian vs xtb

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

## Method

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

## Offline oracle run

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

## Geometry

Gas-phase methane, charge 0, singlet. Cubic box 20 Å (isolation only).
Same Cartesian as `reg_ipea-xtb-small` (\(s=1.09/\sqrt{3}\)).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H | \(s\) | \(s\) | \(s\) |
| H | \(s\) | \(-s\) | \(-s\) |
| H | \(-s\) | \(s\) | \(-s\) |
| H | \(-s\) | \(-s\) | \(s\) |

## Pins

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

## Perf

xtb printed wall **0.170 s** (1 thread). `/usr/bin/time` process wall
**0.43 s**. ATV `Hessian::run` wall **10.4 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.061\le 10\) (vs printed wall).

## Gaps

- `--vip` / `--vea` ΔSCC IP/EA (`ipea-xtb-vip`)
- Polar-molecule residual vs water (~0.5 eV; same kernel hole as energy)
- Elements beyond H/C/N/O
- Open shell, PBC, ALPB
- Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
