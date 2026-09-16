# Reference — periodic water-dimer GPW **optB88-vdW**

## Method

**optB88-vdW** (Klimeš, Bowler, Michaelides, *J. Phys.: Condens. Matter*
**22**, 022201 (2010)):

| Piece | Implementation |
| ----- | -------------- |
| Exchange | Optimized B88 (`libxc` `GGA_X_OPTB88_VDW`, β = 0.003368671161034469, γ = 6.981317007977318) |
| Local correlation | PW92 LDA only (CP2K `&PW92`) |
| Nonlocal correlation | DRSLL / vdW-DF DF1 post-SCF (`GpwDft::vdw_df`) |

Public API:

```cpp
gpw.xc = cpu::gpw::XcKind::OptB88VdW;
gpw.vdw_df = true;
```

Shared pointwise engine: `cpu::scf::evaluate_xc(XCType::OptB88VdW, …)` on the
GPW grid (`xc_grid_optb88_vdw`). Nonlocal energy on `GpwResult::e_disp` [Ha],
folded into `e_total`. Public Frame energy remains eV via `GPWEvaluator`.

CP2K twin surface (Hutter recipe):

```
&XC_FUNCTIONAL
  &LIBXC
    FUNCTIONAL XC_GGA_X_OPTB88_VDW
  &END LIBXC
  &PW92
  &END PW92
&END XC_FUNCTIONAL
&vdW_POTENTIAL
  DISPERSION_FUNCTIONAL NON_LOCAL
  &NON_LOCAL TYPE DRSLL
  &END NON_LOCAL
&END vdW_POTENTIAL
```

Input: `tools/cp2k_oracle/inputs/gpw-optb88-vdw/water_dimer_optb88_vdw.inp`.

## Geometry

Same ice-relevant water dimer as `gpw-vdw-df-ice` / `gpw-rvv10-ice`:

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
CUTOFF **200 Ry**, Γ-point, mix 0.4, tol 1e−7, max_iter 40, \(n_{\mathrm{occ}}=8\).

## Offline oracle status

**CP2K** binary present on pin host (`cp2k.psmp`) but **built without LIBXC**
(`unknown subsection LIBXC of section XC_FUNCTIONAL`). Absolute CP2K
\(E_{\mathrm{tot}}\) / dispersion digits are therefore **not** invented.

Pin strategy (binding):

1. **Semi-local dual gate.** Same geometry, `vdw_df=false`:
   \[
   |E_{\mathrm{OptB88VdW}} - E_{\mathrm{PBE}}| > 10^{-4}~\mathrm{Ha}.
   \]
   Measured **0.303 Ha**.
2. **Absolute first-green ATV freeze** for full optB88-vdW (`OptB88VdW` +
   `vdw_df`):
   - \(E_{\mathrm{disp}}\) = **1.5207282287995946×10⁻¹ Ha**
   - \(E_{\mathrm{tot}}\) = **−3.4481701207152035×10¹ Ha**
3. **Unit kernel:** optB88 exchange parameters match libxc `GGA_X_OPTB88_VDW`
   (pyscf/libxc reverse-fit max |Δexc| &lt; 1e−7 on representative points);
   DRSLL kernel already pinned vs libvdwxc in `gpw-vdw-df-ice`.

When a LIBXC-enabled CP2K is available offline, re-run the twin input and
replace absolute pins (cite CP2K version), tighten tol.

### Atomiverse numbers (verified 2026-07-26, first green)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{disp}}\) (optB88 density + DRSLL) | **1.5207282287995946×10⁻¹ Ha** |
| \(E_{\mathrm{tot}}\) (optB88-vdW) | **−34.481701207152035 Ha** |
| \(E_{\mathrm{OptB88VdW}}\) semi-local | −34.633774030031404 Ha |
| \(E_{\mathrm{PBE}}\) semi-local | −34.330331404832023 Ha |
| \(\lvert\Delta E_{\mathrm{semi}}\rvert\) | 0.303 Ha (\(> 10^{-4}\)) |
| SCF iters (full) | 11 (converged) |

Note: \(E_{\mathrm{disp}}\) is close to the PBE+DRSLL pin of `gpw-vdw-df-ice`
(1.519439×10⁻¹ Ha) — density-driven difference only; the **semi-local**
shift is the activation gate for optB88 exchange + PW92.

## Tolerance

- Dual-gate floor: \(10^{-4}\) Ha (**hard**).
- \(E_{\mathrm{disp}}\): \(5\times 10^{-5}\) Ha.
- \(E_{\mathrm{tot}}\): \(1\times 10^{-3}\) Ha vs first-green ATV until offline
  CP2K replaces it.

## Perf gate

**perf_exempt = true**: primary deliverable is XC combo + E_disp pin parity
(same policy as `gpw-vdw-df-ice` / `gpw-rvv10-ice`). No fair CP2K QS wall
(LIBXC unavailable on pin host).
