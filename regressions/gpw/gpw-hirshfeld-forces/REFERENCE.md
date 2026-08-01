# Reference — water-dimer Hirshfeld-scaled TS/MBD ionic forces

## Method

**Analytic ionic forces** (and fixed-\(P\) cell stress fold) with
**Hirshfeld volume ratios** \(v_A = V_A/V_A^{\mathrm{free}}\) scaling free-atom
**TS** and **MBD@rsSCS** parameters, on top of Γ-point **GPW PBE**.

Found rot (pre-fix): energy folded post-SCF volumes into `ts_pbe` /
`mbd_rsscs_pbe`, but `gpw_forces` / `gpw_stress` called the free-atom kernels
with empty `vol_ratio` — forces/stress ignored `GpwDft::hirshfeld`.

Product definition (density-chain \(\partial v/\partial R\) out of scope):

\[
F_A = -\frac{\partial E_{\mathrm{disp}}}{\partial R_A}\Big|_{v}
\quad\text{with \(v\) fixed at the SCF density},
\]

same fixed-\(v\) functional rebuilt under orthorhombic strain for analytic
stress.

Public API:

```cpp
GpwDft gpw;
gpw.ts = true;          // or gpw.mbd = true
gpw.hirshfeld = true;   // post-SCF volumes → energy, forces, stress
gpw.build(atoms);
const auto& scf = gpw.run(atoms);
auto F = gpw_forces_from_scf(atoms, gpw, scf);
```

Scaling of free-atom \(\alpha_0\) / \(C_6\) / \(R_{\mathrm{vdW}}\):
\(\alpha\propto v\), \(C_6\propto v^2\), \(R\propto v^{1/3}\).

## Reference values (hard gates)

| Quantity | Source | Value / tol |
| -------- | ------ | ----------- |
| **E_disp isolated MBD@rsSCS + fixed \(v\)** | ATV product kernel = dual (same volumes as `gpw-mbd-hirshfeld`) | **−1.1479717148086300×10⁻³ Ha** (fixed dual volumes below), tol 1×10⁻⁸ |
| **Force dual-gate (TS, MBD)** | product \(F\) vs central FD \(h=10^{-5}\) Bohr of fixed-\(v\) energy | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| **Translation** | \(\|\sum_A F_A\|\) (MBD) | ≤ **1×10⁻⁸** Ha/Bohr |
| **Product wiring** | \(F(\mathrm{ts}+\mathrm{hirsh})-F(\mathrm{plain})\) vs `ts_pbe(…, scf.volume_ratios)` | max\|Δ\| ≤ **1×10⁻⁷** Ha/Bohr |
| **Not free-atom** | max\|F_hirsh − F_free\| and wiring residual vs free-atom | must exceed noise |
| **Stress fold** | analytic \(\sigma(\mathrm{ts}+\mathrm{hirsh})\) vs \(\sigma(\mathrm{ts})\) free-atom | max\|Δσ\| > 1×10⁻¹² Ha/Bohr³ |

Representative fixed volumes (order O,H,H,O,H,H), matching
`reg_gpw-mbd-hirshfeld`:

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784` (fixed dual table; product SCF ratios ≈ these)

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no** stock TS/MBD. Hard pin is **libMBD /
product dual** (force FD of the same fixed-\(v\) energy) plus product wiring.
Twin notes: `tools/cp2k_oracle/inputs/gpw-hirshfeld-forces/`.

**oracle:** literature / libMBD (volume-scaled gradient dual).  
**perf_exempt:** true — no fair CP2K twin algorithm (same policy as
`gpw-ts-forces` / `gpw-mbd-hirshfeld` / `gpw-mbd-forces`).

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (product SCF): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.  
Isolated dual-gates use the same nuclear coordinates without lattice images.

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).  
CUTOFF **200 Ry**, n_occ 8, tol 1e-6, mix 0.4, PBE.

## Perf gate

**perf_exempt = true** — primary deliverable is fixed-\(v\) force dual-gate +
product wiring parity. CP2K has no matched TS/MBD ENERGY_FORCE twin for a fair
wall ratio. Product path wall budget ≤ 60 s on the regression host.
