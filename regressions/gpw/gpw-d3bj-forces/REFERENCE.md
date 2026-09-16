# Reference — periodic water-dimer GPW PBE-D3(BJ) analytic ionic forces

## Method

Γ-point **GPW** (PBE) on a **periodic** water dimer (`PERIODIC XYZ`) with
additive **DFT-D3 Becke–Johnson (rational) damping**, PBE parameters, two-body
only (`CALCULATE_C9_TERM .FALSE.`).

\[
E_{\mathrm{disp}} = -\sum_{A<B}\Bigl[\frac{s_6 C_6^{AB}}{R^6 + f^6}
  + \frac{s_8 C_8^{AB}}{R^8 + f^8}\Bigr],\qquad
f = a_1\sqrt{C_8^{AB}/C_6^{AB}} + a_2
\]

Product ionic forces = electronic Hellmann–Feynman + Pulay (analytic
`gpw_forces_*`) **plus** the geometry-only dispersion gradient
\(-\nabla E_{\mathrm{disp}}\) from `cpu::gpw::dftd3_pbe_bj`, summed over lattice
images inside the CP2K-parity pair cutoff.

Public path: `cpu::gpw::GPWEvaluator` with `d3bj = true`,
`compute_forces = true`, `nonperiodic = false`.

Sibling: `gpw-d3-forces` (isolated cluster, zero damping). Together they cover
both damping functions × both boundary conditions for the dispersion force.

## Why the pin is a *difference* of two CP2K runs

DFT-D3 is geometry-only — it never enters the Kohn–Sham potential, so the
converged density is identical with and without it. Hence

\[
F_A^{\mathrm{CP2K}}(\mathrm{D3BJ}) - F_A^{\mathrm{CP2K}}(\mathrm{plain})
  = -\nabla_A E_{\mathrm{disp}},
\]

which isolates exactly the quantity this capability adds; the much larger
electronic force cancels.

## System

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, **`PERIODIC XYZ`** |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | DFT-D3(BJ), PBE: \(s_6=1.0\), \(s_8=0.7875\), \(a_1=0.4289\), \(a_2=4.4407\) Bohr; CP2K CN form (`cp2k_cn = true`); ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å = 75.589 Bohr |
| Cutoff | 300 Ry (CP2K `CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| SCF | `EPS_SCF 1e-7` / Atomiverse `tol = 1e-7` |

Placement is verbatim CP2K here: a periodic cell wraps by minimum image, so the
absolute position is immaterial. (The isolated MT sibling must be centred — see
`gpw-d3-forces/REFERENCE.md` "Placement".)

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3bj-forces/water_dimer_d3bj_forces.inp` and `water_dimer_plain_forces.inp` |
| Wall | ≈ **11.9 s** (D3BJ run), ≈ **13.5 s** (plain run) |
| Date | 2026-07-26 |

### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain, periodic) | −34.338251977942306 |
| Total (D3BJ, periodic) | −34.339628255637152 |
| **Dispersion energy** | **−0.00137627769481** |

Atomiverse `dftd3_pbe_bj` on the same nuclei: **−1.37627778350509e−03 Ha**
(\(|\Delta| = 8.9\times10^{-11}\) Ha).

### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` table)

D3(BJ) run:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.18231267E−02 | 3.13886473E−08 | −7.96369314E−09 |
| H0 | 4.08872662E−02 | 1.13178277E−08 | 1.31270247E−01 |
| H1 | 4.08872656E−02 | −3.52097593E−08 | −1.31270237E−01 |
| O1 | −8.05124430E−02 | −1.50606809E−08 | 2.18547987E−08 |
| H2 | 3.75636473E−02 | 1.02088685E−08 | 1.35137187E−01 |
| H3 | 3.75636570E−02 | −2.12332662E−09 | −1.35137209E−01 |

Plain run:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.19227425E−02 | 3.13886469E−08 | −7.96370458E−09 |
| H0 | 4.08225109E−02 | 1.13178281E−08 | 1.31264307E−01 |
| H1 | 4.08225103E−02 | −3.52097601E−08 | −1.31264297E−01 |
| O1 | −8.04237663E−02 | −1.50606797E−08 | 2.18548003E−08 |
| H2 | 3.76338722E−02 | 1.02088694E−08 | 1.35133732E−01 |
| H3 | 3.76338819E−02 | −2.12332661E−09 | −1.35133754E−01 |

Derived dispersion force largest component **9.962E−05 Ha/Bohr**; Atomiverse
reproduces it to **1.21E−06 Ha/Bohr**.

## Measured results (2026-07-26, this pin)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1e−8 Ha | **8.9e−11** |
| \(F_{\mathrm{disp}}\) vs CP2K difference | 5e−6 Ha/Bohr | **1.21e−06** |
| Total \(F\) vs CP2K | 1e−2 Ha/Bohr | **4.30e−03** |
| \(\lvert\sum_A F_A\rvert\) (translation invariance) | 3e−3 Ha/Bohr | **1.34e−03** |
| Product wiring | 1e−7 Ha/Bohr | **5.9e−12** |
| Wall | ≤ 60 s | **32.2 s** |

## Tolerance justification

- **\(E_{\mathrm{disp}}\) 1e−8 Ha:** closed-form two-body sum with published
  PBE-BJ parameters and the same CN convention; measured 8.9e−11. Only table
  transcription can move it.
- **\(F_{\mathrm{disp}}\) 5e−6 Ha/Bohr:** the pin is a difference of two
  independently converged CP2K SCFs, so it inherits CP2K's force noise;
  measured floor 1.2e−6 against a 1.0e−4 signal.
- **Total \(F\) 1e−2 Ha/Bohr:** bounded below by the *oracle's* grid error —
  CP2K's own `FORCES| Sum` is 5.43e−3 Ha/Bohr on this twin, so a tighter
  absolute comparison is not physically meaningful at `CUTOFF 300`.
- **\(\lvert\sum F\rvert\) 3e−3 Ha/Bohr:** exact in the continuum, broken by
  grid collocation (egg-box). Atomiverse's 1.34e−3 is 4× tighter than the
  oracle's 5.43e−3 on the same system. The gate's purpose is to catch the
  boundary-condition placement bug class (signature |Σ F| ~ 1–2 Ha/Bohr, see
  `gpw-d3-forces/REFERENCE.md`), so ~500× separation is retained.
- **Wiring 1e−7 Ha/Bohr:** same code on both sides of the difference, so only
  the SCF residual enters; measured 5.9e−12.

## Dual gate

- **Correctness:** the six gates above, plus the analytic-vs-multi-SCF-FD
  gradient identity in `tests/cpu/gpw/test_cpu_gpw_forces.cpp`
  (`GpwForces.AnalyticMatchesFdWithD3Bj`, measured 2.4e−6 Ha/Bohr on cheap MT
  H₂ — 6N SCFs on this cell exceed the e2e budget).
- **Speed:** CP2K `ENERGY_FORCE` twin ≈ 11.9 s; Atomiverse product pass 32.2 s
  on the same host → ratio ≈ **2.7**, inside the 10× fair wall. Note the
  Atomiverse figure includes one extra plain SCF for the wiring gate.

## Notes

- Sibling dispersion-force products: `gpw-d3-forces` (isolated, zero damping),
  `gpw-d4-forces` (periodic, D4/BJ).
- Sibling D3(BJ) energy product: `gpw-pbe-d3bj-ice`.
- The shared `kD3Cp2kPairCutoffBohr` constant (`internal/dftd3.h`) was
  introduced with this pin: the zero-damping path previously used a hard-coded
  40 Bohr, which disagreed with CP2K by 3.9e−7 Ha on a periodic cell. Verified
  against an offline periodic D3-zero CP2K run (`E_disp = −8.20894676620e−04`;
  75.589 Bohr reproduces it to 6.7e−11, 40 Bohr to 3.9e−7).
