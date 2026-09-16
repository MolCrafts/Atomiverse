# Reference — periodic water-dimer GPW PBE + rVV10 nonlocal correlation

## Method

**Nonlocal rVV10 correlation** (Sabatini–Gorni–de Gironcoli revised VV10) as a
post-SCF additive on top of Γ-point **GPW PBE**. Ice-relevant H-bonded water
dimer in a periodic 12 Å cell (same geometry as `gpw-ts-ice` / `gpw-vdw-df-ice`).

Public API: `cpu::gpw::GpwDft::rvv10 = true` (and `GPWEvaluator::rvv10`).
Nonlocal energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Defaults: b = 6.3, C = 0.0093.

Formula (Sabatini et al., *Phys. Rev. B* **87**, 041108(R) (2013); RPS FFT
of Román-Pérez & Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)):

\[
E_c^{\mathrm{nl}}
  = \beta\int\rho\,\mathrm d\mathbf r
  + \tfrac12\iint \theta(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\theta(\mathbf r')
\]

with \(q_0 = w_0/\kappa\) saturated to the q-mesh, \(\theta_\alpha = c\,\rho^{3/4}
p_\alpha(q_0)\), and the community rVV10 kernel table (CP2K
`rVV10_kernel_table.dat`). Evaluation is **post-SCF** on the final total
density; spectral \(|\nabla\rho|^2\) from the GPW FFT gradient.

CP2K twin surface: `&VDW_POTENTIAL` / `POTENTIAL_TYPE NON_LOCAL` /
`&NON_LOCAL TYPE RVV10` (PARAMETERS 6.3 0.0093).

## Reference value (hard gate)

**E_disp (PBE density + rVV10 E_c^{nl}, periodic water dimer, 12 Å cell,
CUTOFF 200 Ry, DZVP-MOLOPT-SR-GTH + GTH-PBE) =
+6.11534922097146×10⁻² Ha**

| Source | Geometry / density | E [Ha] |
| ------ | ------------------ | ------ |
| **CP2K 2026.1** TYPE RVV10 (SC, OT SCF) | water dimer | **+6.116467960721×10⁻²** |
| Atomiverse GPW PBE + `rvv10` (this regression) | post-SCF ρ | **+6.11534922097146×10⁻²** |

Δ(ATV post-SCF − CP2K SC) ≈ **−1.1×10⁻⁵ Ha** (expected: post-SCF vs
self-consistent density; float32 kernel tables).

### Offline notes

`tools/cp2k_oracle/inputs/gpw-rvv10-ice/`.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Tolerance

**5 × 10⁻⁵ Ha** on E_disp (regression dual-frozen pin).

## Perf gate

**perf_exempt = true**: primary deliverable is E_disp pin parity (same policy
as `gpw-vdw-df-ice` / `gpw-ts-ice`). CP2K wall ≈ 121 s (OMP=8) for full QS;
ATV SCF+kernel is dominated by electronic SCF.
