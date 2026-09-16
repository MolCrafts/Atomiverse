# Reference — constant-charge (conq) electrode control, parallel plate

oracle: lammps  
oracle_version: 3 Nov 2022 (ELECTRODE package: madelung example log
`log.1Dec2022.madelung-ewald-ew3dc.g++.1` + `graph-il/in.conq`
`fix electrode/conq` semantics)

## What this pins

Constant-**charge** electrode control (LAMMPS ELECTRODE `fix electrode/conq`,
Ahrens-Iwers et al., *J. Chem. Phys.* **157**, 084801 (2022), DOI
[10.1063/5.0099239](https://doi.org/10.1063/5.0099239)): the total charge of
each electrode is prescribed, the per-site distribution is solved from the
Gaussian elastance matrix, and the uniform electrode potential \(\Psi_e\) is
the **response**. LAMMPS exposes the response as `f_conq[e]` and its example
observable is `v_dv = f_conq[2] - f_conq[1]`; the input notes *"symm on and
off give different electrode potentials, but identical potential
difference"* — i.e. conq is the exact inverse of the conp solve up to a
uniform shift. That inverse relation is the oracle contract used here.

## System (LAMMPS ELECTRODE madelung twin)

Same geometry as `regressions/md/const-pot-bo-parallel-plate/`:

| Species | Type | z [Å] | q [e] |
| --- | --- | --- | --- |
| Bottom electrode | 1 | 8 | solved |
| Top electrode | 2 | 12 | solved |
| Electrolyte + | 3 | 9 | +0.5 |
| Electrolyte − | 3 | 11 | −0.5 |

Box \(1\times1\times20\) Å, \(\eta = 2.0\) Å⁻¹, 2D \(xy\) Gaussian image
range 250 (v1 real-space elastance — not full `ewald/electrode`).

## Offline pins

1. **LAMMPS conp anchor** (madelung log, `run 0`,
   `fix electrode/conp 0 2 couple top 1 symm on`):
   \(\Delta\Psi = 1\) V \(\Leftrightarrow Q_{\mathrm{top}} =
   +0.279214485147238\) e.
2. **conq inverse contract** (`graph-il/in.conq` semantics): feeding the conp
   totals \(\pm Q^{*}\) back as conq targets must return the conp potential
   difference exactly (uniform-shift-invariant).
3. **Literature Madelung-sum plate capacitor** (empty cell, no electrolyte,
   \(Q = \pm1\) e, \(A_{\mathrm{cell}} = 1\) Å², \(d = 4\) Å,
   \(\eta = 2\) Å⁻¹): the potential difference between the two **discrete**
   Gaussian-charge lattices is

   \[
   \Delta\Psi = 2 k_e\Bigl[\,2\pi\sigma d
     + \eta\sqrt{2/\pi} + M_\square + g(\eta)\Bigr]
   = \mathbf{651.8103\ V},
   \]

   with \(k_e = 14.399645\) eV·Å/e², sheet term \(2\pi\sigma d\)
   (Griffiths, *Introduction to Electrodynamics*, 4th ed., §2.5.4), the
   square-lattice 2D Madelung constant \(M_\square = -3.90026492\) (unit
   spacing; Glasser & Zucker, *Theoretical Chemistry: Advances and
   Perspectives* **5**, 67 (1980), lattice sums) and the rapidly convergent
   Gaussian correction \(g(\eta) = -\sum_{n\neq0}
   \operatorname{erfc}(\tfrac{\eta}{\sqrt2}\rho_n)/\rho_n = -0.195386\).
   Same Madelung-sum route as `plate_cap.py` in the LAMMPS madelung example.
   The bare-sheet textbook value \(4\pi k_e\sigma d = 723.88\) V is **not**
   the pin — it omits the on-site discrete-lattice term (−72.07 V here).
   \(R\to\infty\) numeric extrapolation of the v1 lattice sum reproduces
   651.81 V; at \(R = 250\) the real-space tail leaves −0.80%.

## What Atomiverse asserts

Public API: `cpu::md::ConstPotBO::add_electrode_charge` (Component on
`cpu::md::MD` at `pre_force`; `electrode_potential(e)` / `delta_psi()` = the
LAMMPS `f_conq[e]` / `v_dv` observables).

| Check | Tol |
| --- | --- |
| conp \(Q^{*}_{\mathrm{top}}\) vs LAMMPS pin 0.279214485147238 e | ≤ 1% (v1 A-matrix gate, same as conp regression) |
| conq group totals = targets ("Q fixed") | ≤ 1e−12 e |
| conq \(\Delta\Psi\) with targets \(\pm Q^{*}\) = 1 V (inverse contract) | ≤ 1e−8 V |
| conq per-site charges = conp per-site charges (duality) | ≤ 1e−10 e |
| conq \(\Delta\Psi\) with the raw LAMMPS \(Q_{\mathrm{pin}}\) | 1 ± 0.25 V (see error budget) |
| Empty-capacitor \(\Delta\Psi\) at \(Q=\pm1\) e vs 651.8103 V | ≤ 1.5% (R=250 tail ≈ 0.8%) |
| MD-bound path (1 Verlet step): totals exact, \(\Delta\Psi\) | ≤ 1e−10 e / ±0.05 V |

## Error budget (why the raw-\(Q_{\mathrm{pin}}\) gate is 0.25 V)

The plate stiffness is \(d\Delta\Psi/dQ = 2(a-c) \approx 4\pi k_e d /
A_{\mathrm{cell}} \approx 724\) V/e. The v1 real-space 2D-image elastance
differs from LAMMPS `ewald/electrode` by ~0.07% in \(Q\) at fixed \(V\)
(≈ 2×10⁻⁴ e), which the stiffness amplifies to ≈ 0.14 V at fixed \(Q\).
The well-conditioned oracle observable is therefore the inverse-contract
\(\Delta\Psi(\pm Q^{*}) = 1\) V (checked to 1e−8) with \(Q^{*}\) itself gated
against the LAMMPS pin at 1%; the raw-\(Q_{\mathrm{pin}}\) response is gated
coarsely at ±0.25 V to catch sign/scale regressions.

## Perf

```
perf_exempt: true
perf_exempt_reason: local lmp lacks PKG_ELECTRODE (verified 2026-07-27: installed = EXTRA-*, KSPACE, MANYBODY, MISC, MOLECULE, OPENMP, QEQ, REPLICA, RIGID); no fair fix electrode/conq twin binary
ratio_atv_over_oracle: n/a
within_10x: exempt
```

## Repro of pin (offline only — never ctest)

```bash
# Requires LAMMPS built with -D PKG_ELECTRODE=ON
cd $LAMMPS_DIR/examples/PACKAGES/electrode/madelung
lmp -in in.ewald-ew3dc            # conp: Q_top at dV = 1 V (run 0)
cd ../graph-il && lmp -in in.conq # conq: f_conq[e] potentials, v_dv
```

See also `tools/metalwalls_oracle/inputs/lammps/in.conq_madelung` (the exact
conq twin of this regression) and
`tools/metalwalls_oracle/pin_notes_const-charge-electrode.md`.
