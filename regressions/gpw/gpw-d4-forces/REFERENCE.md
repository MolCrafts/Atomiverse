# Reference — periodic water-dimer GPW PBE-D4 analytic ionic forces

## Method

Γ-point **GPW** (PBE) with additive **DFT-D4** (Becke–Johnson, two-body,
PBE parameters; ATM off) on a periodic water dimer.

**Product ionic forces** = electronic Hellmann–Feynman + Pulay (analytic
`gpw_forces_*`) **plus** geometry-only
\(F_{\mathrm{disp}}^{\mathrm{D4}} = -\nabla E_{\mathrm{disp}}\) from
`cpu::scf::dftd4` (exact gradient of the same two-body energy as
`dftd4_energy` / `gpw-d4-water`).

Oracle twins:

1. **Hard (geometry-only):** offline **dftd4 3.7.0** `-g` gradients on the
   same nuclei (force = −gradient).
2. **Wiring (product):** \(F^{\mathrm{d4}}-F^{\mathrm{plain}}\approx F_{\mathrm{disp}}\)
   (geometry-only D4 does not change the SCF density).
3. **Unit dual-gate:** multi-SCF `gpw_forces_fd` with `d4=true` on cheap MT
   H₂ lives in `tests/cpu/gpw/test_cpu_gpw_forces.cpp` (6N SCFs on the water
   dimer exceed the ≤60 s regression budget).

Public path: `cpu::gpw::GPWEvaluator` with `d4=true`, `compute_forces=true`.

## System

Same geometry as `gpw-d4-water` / `gpw-d4-stress`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms** (order **O H H O H H**) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); ATM off |
| Cutoff | 200 Ry |

## Offline pins

### Hard: E_disp (dftd4 3.7.0)

**E_disp = −0.00099458842745846774 Ha** (two-body, periodic, `--mbdscale 0`).

Source: offline **dftd4 3.7.0** on Arrhenius (same as `gpw-d4-water`).

### Hard: D4 gradient (dftd4 3.7.0 `-g`)

Printed **gradient** \(g = \partial E/\partial R\) [Ha/Bohr]. Product force
contribution is \(F = -g\).

Remapped from POSCAR order (O O H H H H) to ATV order **O H H O H H**:

| Atom | \(g_x\) | \(g_y\) | \(g_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −5.3738961271913915e−05 | ~0 | ~0 |
| H0 | −5.9550148958160894e−05 | ~0 | −1.0810592129949165e−05 |
| H1 | −5.9550148958160921e−05 | ~0 | +1.0810592129949182e−05 |
| O1 | +1.3609823922940861e−04 | ~0 | ~0 |
| H2 | +1.8370509979413634e−05 | ~0 | −8.5497124425277334e−06 |
| H3 | +1.8370509979413539e−05 | ~0 | +8.5497124425278096e−06 |

Gradient norm ≈ **1.719e−04** Ha/Bohr.

### Why not CP2K QS ENERGY_FORCE?

**CP2K 2026.1** on Arrhenius is built **without** DFTD4 (`TYPE DFTD4` aborts).
Twin input retained under `tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts
with libdftd4. Hard pin is **libdftd4 / dftd4 3.7.0**.

### Optional: re-run pin (offline only — never ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --grad pin_grad.txt --json pin.json POSCAR
# note POSCAR atom order O O H H H H — remap to O H H O H H for ATV
```

## Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| | **1×10⁻⁸** Ha |
| max\|F_d4 + g_dftd4\| | **1×10⁻⁷** Ha/Bohr |
| max\|(F_d4_on − F_plain) − F_d4\| | **1×10⁻⁷** Ha/Bohr (wiring; SCF residual) |
| Unit dual-gate max\|F_an − F_fd\| | **5×10⁻⁵** Ha/Bohr (H₂, not this e2e) |

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| --- | --- |
| **E (Γ PBE-D4 200 Ry)** | **−34.3313259463 Ha** (SCF 11 iters) |
| E_disp | **−9.9458849003×10⁻⁴ Ha** (\|d pin\| = 6.3×10⁻¹¹) |
| max\|F_d4 + g_dftd4\| | **3.0×10⁻¹¹** Ha/Bohr |
| max\|(F_d4 − F_plain) − F_standalone\| | **2.9×10⁻⁹** Ha/Bohr |
| max\|F_product\| | **0.136** Ha/Bohr |
| T_product (OMP=8, Release) | **17.6 s** |
| T_plain twin (OMP=8) | **17.1 s** |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | Arrhenius CP2K build lacks DFTD4 (no fair QS ENERGY_FORCE wall). Primary hard gate is dftd4 E_disp + gradient pin parity + product wiring (same policy as `gpw-d4-water` / `gpw-d4-stress`). |

## What is checked

1. Product analytic forces with `d4=true` finish without throw.
2. `e_disp` within 1e-8 Ha of the dftd4 pin.
3. Standalone D4 forces match offline −gradient pin.
4. Additive wiring: product(d4) − product(plain) ≈ standalone D4.
5. Unit suite dual-gate (H₂ + D4 analytic vs multi-SCF FD).
