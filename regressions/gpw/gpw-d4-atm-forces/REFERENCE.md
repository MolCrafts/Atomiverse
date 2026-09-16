# Reference — periodic water-dimer GPW PBE-D4+ATM analytic ionic forces

## Method

Γ-point **GPW** (PBE) with additive **DFT-D4** including the
Axilrod–Teller–Muto **three-body** term (\(s_9 = 1\), dftd4 `pbe-D4-ATM`) on
a periodic water dimer — the force companion of the `gpw-d4-atm` energy
foothold and the ATM extension of `gpw-d4-forces`.

**Product ionic forces** = electronic Hellmann–Feynman + Pulay (analytic
`gpw_forces_*`) **plus** geometry-only
\(F_{\mathrm{disp}} = -\nabla(E_2 + E_{\mathrm{ATM}})\) from
`cpu::scf::dftd4` with `s9 = 1`. The ATM gradient is **analytic**: explicit
damped Axilrod–Teller triple derivatives plus the
C9 → neutral-C6 → CN\(_\mathrm{d4}\) chain (the \(q = 0\) reweight of dftd4
`disp.f90` is geometry-independent, so the ATM leg has no EEQ adjoint).

Public path: `cpu::gpw::GPWEvaluator` with `d4 = true`, `d4_atm = true`,
`compute_forces = true`.

## System

Same geometry as `gpw-d4-atm` / `gpw-d4-forces`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms** (order **O H H O H H**) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); **ATM on**: \(s_9=1\), alp = 16, disp3 cutoff 40 Bohr, neutral-charge C6 reweight |
| Cutoff | 200 Ry |

## Offline pins (dftd4 3.7.0)

### Hard: E_disp with ATM

**E_disp = −9.9554475834240310×10⁻⁴ Ha** (`pin_periodic_atm_grad.json`;
matches the `gpw-d4-atm` energy pin −9.9554475834240288×10⁻⁴ to 2×10⁻²⁰).

### Hard: D4+ATM gradient (dftd4 3.7.0 `--grad`, s9 = 1)

Printed gradient \(g = \partial E/\partial R\) [Ha/Bohr]; product force
contribution is \(F = -g\). Remapped from POSCAR order (O O H H H H) to ATV
order **O H H O H H** (`pin_periodic_atm_grad.json`, produced 2026-07-31):

| Atom | \(g_x\) | \(g_y\) | \(g_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −5.3595427766884764e−05 | ~0 | ~0 |
| H0 | −5.9463596930597288e−05 | ~0 | −1.0397154303254306e−05 |
| H1 | −5.9463596930597301e−05 | ~0 | +1.0397154303254314e−05 |
| O1 | +1.3610085695186597e−04 | ~0 | ~0 |
| H2 | +1.8210882338106763e−05 | ~0 | −8.1642912818385456e−06 |
| H3 | +1.8210882338106668e−05 | ~0 | +8.1642912818386235e−06 |

The ATM-only force increment vs the two-body `--mbdscale 0` pin
(`gpw-d4-forces` REFERENCE) has max component **4.05×10⁻⁷ Ha/Bohr** — that is
the physics this case gates.

Isolated twin (same nuclei, no cell; ATV atom order directly):
`pin_isolated_atm_grad.json`, E = −9.8819741533958229×10⁻⁴ Ha — pinned at
**1×10⁻⁹ Ha/Bohr** in `tests/cpu/gpw/test_cpu_gpw_dftd4.cpp` (ATV measured
1.7×10⁻¹²).

### Oracle finding: dftd4 3.7.0's periodic ATM gradient is non-conservative

Checked offline (2026-07-31), all numbers printed by dftd4 3.7.0:

- **Isolated**: central difference of its s9=1 energy (h = 10⁻³ Å, O0 x)
  matches its printed gradient to **2.6×10⁻¹⁰** Ha/Bohr — conservative.
- **Periodic (12 Å cell)**: the same FD gives −5.3616541919×10⁻⁵ while its
  printed \(g_x\)(O0) is −5.3595427767×10⁻⁵ — a **2.1×10⁻⁸ Ha/Bohr**
  discrepancy against its *own* energy (max over components ~2.4×10⁻⁸).
- ATV's s9=1 periodic energy equals dftd4's at the base **and** displaced
  geometries to a constant 3.8×10⁻¹² Ha (Bohr-conversion constant; cancels
  in differences), so the two codes share one energy functional, and ATV's
  analytic force is conservative against it to 3×10⁻¹² (self-FD test).

A conservative gradient of the shared functional therefore cannot match the
printed periodic pin beyond that oracle defect. Gates: the **direct** pin
comparison at **5×10⁻⁸** (absorbs the defect, still 8× below the ATM
increment, so it fails if the ATM gradient is dropped), plus a **sharp
1×10⁻⁹ conservative-parity** gate against the FD over the oracle's printed
displaced energies (truncation ≈ 3×10⁻¹⁰):

| Pin | dftd4 3.7.0 s9=1 energy [Ha] |
| --- | --- |
| O0 x = +0.001 Å (`pin_periodic_atm_dispO0x_plus.json`) | −9.9564614957741663×10⁻⁴ |
| O0 x = −0.001 Å (`pin_periodic_atm_dispO0x_minus.json`) | −9.9544350841746518×10⁻⁴ |

### Why not CP2K QS?

**CP2K 2026.1** on Arrhenius is built **without** DFTD4 (`TYPE DFTD4`
aborts). The hard pin is **libdftd4 / dftd4 3.7.0**, the same backend CP2K
uses when DFTD4 is linked — same policy as `gpw-d4-atm` / `gpw-d4-forces`.

### Re-run (offline only — never ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
cd tools/cp2k_oracle/inputs/gpw-d4-atm
dftd4 -f pbe --grad grad_periodic_atm.txt --json pin_periodic_atm_grad.json POSCAR
# POSCAR atom order O O H H H H — remap to O H H O H H for ATV
dftd4 -f pbe --grad grad_isolated_atm.txt --json pin_isolated_atm_grad.json water_dimer.xyz
# Displaced-energy pins (edit POSCAR O0 x to ±0.001 before each run):
dftd4 -f pbe --json pin_periodic_atm_dispO0x_plus.json POSCAR   # x=+0.001 Å
dftd4 -f pbe --json pin_periodic_atm_dispO0x_minus.json POSCAR  # x=−0.001 Å
```

## Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| (standalone and product) | **1×10⁻⁸** Ha |
| max\|F_d4atm + g_pin\| (direct, periodic) | **5×10⁻⁸** Ha/Bohr (oracle defect ~2.4×10⁻⁸; ATM increment 4.05×10⁻⁷) |
| \|F_x(O0) − FD(oracle E±)\| (conservative parity) | **1×10⁻⁹** Ha/Bohr |
| ATM increment max\|(F_atm − F_2b) − Δpin\| | **5×10⁻⁸** Ha/Bohr (same oracle defect); non-vacuity max\|dF\| > 1×10⁻⁸ |
| max\|(F_on − F_plain) − F_standalone\| | **1×10⁻⁷** Ha/Bohr (wiring; SCF residual) |
| Isolated max\|F + g_pin\| (unit suite) | **1×10⁻⁹** Ha/Bohr |
| Unit dual-gate max\|F_an − F_fd\| (s9=1 functional) | **1×10⁻⁷** Ha/Bohr (FD-limited; unit suite) |

## Atomiverse numbers (first green, 2026-07-31)

| Quantity | Value |
| --- | --- |
| E_disp (standalone s9=1) | \|d pin\| = 6.3×10⁻¹¹ Ha (Bohr-conversion constant) |
| max\|F + g_pin\| (periodic, direct) | 2.394×10⁻⁸ Ha/Bohr (= the oracle defect) |
| \|F_x(O0) − FD(oracle E±)\| | 2.65×10⁻¹⁰ Ha/Bohr |
| Isolated max\|F + g_pin\| (unit) | 1.7×10⁻¹² Ha/Bohr |
| ATM increment max\|dF − Δpin\| | 2.391×10⁻⁸ Ha/Bohr (same defect); max\|dF\| = 4.05×10⁻⁷ |
| Self-FD (analytic vs FD of own s9=1 energy, unit) | 3.0×10⁻¹² Ha/Bohr |
| Molecular D4 s9=1 analytic-vs-FD (unit) | 1.9×10⁻¹¹ Ha/Bohr |
| Product E (Γ PBE-D4+ATM 200 Ry) | −34.331326946882 Ha (SCF 11 iters); e_disp \|d pin\| = 2.8×10⁻¹⁴ |
| max\|wiring\| | 1.52×10⁻⁸ Ha/Bohr |
| Wall (product SCF+forces, OMP=8) | 15.7 s |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | Arrhenius CP2K lacks DFTD4 — no fair QS ENERGY_FORCE wall. Hard gate is dftd4 gradient-pin parity (same policy as `gpw-d4-water` / `gpw-d4-forces` / `gpw-d4-atm`). The analytic ATM path replaces the previous 6N-FD ATM leg, strictly faster. |

## What is checked

1. Standalone `dftd4(s9=1)` energy and forces match the offline pins
   (1e−8 Ha / 1e−9 Ha/Bohr).
2. The ATM-only force increment matches the oracle's pin delta (1e−9
   Ha/Bohr) and is non-vacuous (>1e−8).
3. Product GPW `d4 + d4_atm` converges; `e_disp` matches the ATM pin.
4. Additive wiring: product(d4_atm) − product(plain) ≈ standalone (1e−7).
5. Unit suite: isolated + periodic s9=1 pins, ATM increment, and
   analytic-vs-FD of the s9=1 functional
   (`tests/cpu/gpw/test_cpu_gpw_dftd4.cpp`,
   `tests/cpu/scf/test_dispersion_gradient.cpp`).
