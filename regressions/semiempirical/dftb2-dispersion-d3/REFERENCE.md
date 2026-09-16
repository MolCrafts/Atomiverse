# Reference — DFTB2 + D3(BJ) water dimer dispersion

## Method

SCC-DFTB2 electronic + repulsive (ATV static mio-style SK sketch) with
optional **Grimme DFT-D3 Becke–Johnson** two-body correction (`DFTB2::d3`).

Damping parameters = published **DFTB3-D3(BJ)** set (DFTB+ / Grimme list;
`s-dftd3 --bj dftb3`):

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.5883 |
| \(a_1\) | 0.5719 |
| \(a_2\) | 3.6017 Bohr |

CN form: molecular Grimme (simple-dftd3 / PySCF path), **not** CP2K tanh.
ATM off. Non-periodic molecular evaluation.

## Integration note (merge onto dev, 2026-07-28)

This feature was authored against the pre-`sk_set` DFTB2, where the total energy
was assembled inline and forces were a central FD of that total — so folding D3
into the FD evaluator sufficed. On current `dev`, `DFTB2::launch` dispatches to
`launch_mio` / `launch_3ob`, and the closed-shell mio path uses **analytic**
electronic forces with no FD evaluator to hook into. The feature was therefore
**ported**, not merged textually: D3 is now applied once in `DFTB2::launch`
after the SK path returns, contributing

- its energy to `energy_` and `frame.meta[kEnergy]` (deliberately **not** to
  `kEnergyElectronic` — dispersion is an empirical geometric term), and
- its **own analytic gradient** to fx/fy/fz.

This makes `F = −∇E` hold for the total on *every* path (closed-shell analytic,
open-shell FD, 3ob FD) instead of only on the FD ones. Verified out-of-tree by
isolating `F_D3 = F(d3=true) − F(d3=false)` against a central FD of `E_disp`
alone: worst deviation **7.5e-11 eV/Å** on this geometry (max |F_D3| =
0.0067 eV/Å). No in-tree gate covers the D3 force path yet — see Gaps.

Neither gate below depends on the SK parameters, so the 2026-07-27 upgrade of
the H onsite/U rows from sketch values to transcribed mio-1-1 does not move this
pin. Measured after the port: `E_disp = −0.045967885155 eV` against the
s-dftd3 pin `−0.045967885157 eV`, `|Δ| = 1.5e-12 eV`.

**The pinned quantity is unmoved by the 2026-07-28 SCC repair** (D3 is a purely
geometric additive term): `E_disp = −0.045967885155 eV`, `|Δ| = 1.5e-12 eV`.
The printed *total* energies did move — `E(no D3) = −83.6622617788` →
**−188.7287268170 eV**, `E(D3) = −83.7082296640` → **−188.7746947021 eV** —
because the electronic MioSketch part was a non-convergent Picard limit-cycle
snapshot over a double-counting route that was wrong at second order. Neither
total is pinned or claimed against an oracle; the gate is on `E_disp` and on
`E_tot(d3) − E_tot(off) = E_disp` (still 1e-8 eV), both of which held.
Before the repair this water dimer did not converge at all: with linear
damping it needed >50 iterations, which is what drove the switch to Anderson
mixing (it now converges in 14 at `mix_beta = 0.3`).

## Gaps

- ~~**D3 forces are ungated.**~~ **Closed 2026-07-29:**
  `tests/cpu/semiempirical/test_dftb2_dispersion_d3.cpp`
  `ForcesMatchCentralFDOfEDisp` now gates \(F_{D3}\) against a central FD of
  \(E_{\mathrm{disp}}\) in-tree (< 1e-6 eV/Å), molecular and periodic. D3 stress
  is gated too (`StressIncludesD3`, `StressTraceMatchesVolumeDerivative`).
- ~~**No periodic D3.**~~ **Closed 2026-07-29:** `DFTB2::d3_periodic` sums
  lattice images over an orthorhombic cell, pinned against offline s-dftd3 1.2.1
  in `regressions/semiempirical/dftb2-dispersion-d3-periodic/`. This case stays
  the **cluster** pin; the periodic one owns the image term. Note that the
  periodic value depends on `DFTB2::d3_cutoff` (new member, default unchanged at
  94.2 Bohr) — the molecular pin here does not.
- **Triclinic cells still unsupported** — `d3_periodic` throws on a
  non-orthorhombic box.

## Oracle

```text
oracle: literature
oracle_version: Simple-DFTD3/s-dftd3 1.2.1 (offline)
```

Primary numerical pin is **pure \(E_{\mathrm{disp}}\)** from offline
**s-dftd3 1.2.1** (Simple-DFTD3). DFTB+ is the family oracle for the electronic
DFTB2 ladder; for this geometry-only D3 correction the literature/simple-dftd3
engine is the correct pin (same model DFTB+ uses for D3). ctest never launches
s-dftd3 or DFTB+.

## Hard gate — E_disp

**E_disp = −0.0016892886213669401 Ha = −0.04596788515696838 eV**

Source: offline s-dftd3 1.2.1 on the geometry below:

```text
s-dftd3 --bj dftb3 --json pin.json dimer.xyz
# Dispersion energy:      -1.6892886213669E-03 Eh
```

Conversion: CODATA / ATV `kHartreeInEv = 27.211386245988`.

Absolute total DFTB2 energy is **not** pinned to DFTB+ (ATV SK is a mio-style
sketch). Regression gates:

1. `|E_disp_ATV − E_disp_pin| ≤ 1e-4 eV`
2. `E_tot(d3) − E_tot(off) = E_disp` within 1e-8 eV

## Geometry (Å)

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000 | 0.000 | 0.000 |
| H | 0.757 | 0.000 | 0.586 |
| H | −0.757 | 0.000 | 0.586 |
| O | 0.000 | 0.000 | 2.900 |
| H | 0.757 | 0.000 | 3.486 |
| H | −0.757 | 0.000 | 3.486 |

Cubic box 20 Å (isolated; D3 path ignores PBC).

## Offline reproduce

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/Simple-DFTD3/1.2.1-gfbf-2025b/bin:$PATH
# twin: tools/tb_oracle/inputs/dftbplus/dftb2-dispersion-d3/dimer.xyz
s-dftd3 --bj dftb3 --json pin.json dimer.xyz
```

Oracle inputs: `tools/tb_oracle/inputs/dftbplus/dftb2-dispersion-d3/`.

## Perf

`perf_exempt: true` — pure D3 pin is microseconds; fair \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\)
vs s-dftd3 on a 6-atom cluster is not a product gate. Full DFTB2 SCC wall is
electronics-dominated.
