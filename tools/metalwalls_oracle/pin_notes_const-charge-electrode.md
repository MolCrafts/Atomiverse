# Pin notes — const-charge-electrode

**Date:** 2026-07-27  
**Slug:** `const-charge-electrode`  
**Oracle:** lammps (ELECTRODE package, 3 Nov 2022)

Local `lmp` (checked 2026-07-27 via `lmp -h`) is built **without**
`PKG_ELECTRODE` (installed: EXTRA-*, KSPACE, MANYBODY, MISC, MOLECULE,
OPENMP, QEQ, REPLICA, RIGID), so `fix electrode/conq` cannot be launched
locally. The pin therefore uses the **shipped official LAMMPS ELECTRODE
logs** plus the package's documented conq↔conp inverse contract — the same
sourcing as `pin_notes.md` for `const-pot-bo-parallel-plate`.

## Procedure

1. **conp anchor** (madelung example,
   `log.1Dec2022.madelung-ewald-ew3dc.g++.1`, `run 0`,
   `fix electrode/conp 0 2 couple top 1 symm on`):
   \(\Delta\Psi = 1\) V ⇔ \(Q_{\mathrm{top}} = +0.279214485147238\) e.
2. **conq contract** (`graph-il/in.conq`): `fix electrode/conq` prescribes
   the per-electrode total charge and reports the resulting potentials
   `f_conq[e]`; the shipped input's observable is
   `v_dv = f_conq[2]-f_conq[1]` and its comment pins the invariant used
   here: *"symm on and off give different electrode potentials, but
   identical potential difference"* — conq is the exact inverse of conp up
   to a uniform shift.
3. Combining 1+2: conq with targets \(\pm Q^{*}\) (the self-consistent conp
   totals) must return \(\Delta\Psi = 1\) V. Asserted to 1e−8 V in
   `reg_const-charge-electrode`, with \(Q^{*}\) itself gated at 1% against
   the LAMMPS log pin.
4. **Literature cross-check** (empty cell): discrete Gaussian-lattice plate
   capacitor \(\Delta\Psi = 2k_e[2\pi\sigma d + \eta\sqrt{2/\pi} + M_\square
   + g(\eta)] = 651.8103\) V for \(Q = 1\) e, \(A = 1\) Å², \(d = 4\) Å,
   \(\eta = 2\) Å⁻¹ — sheet term (Griffiths, *Introduction to
   Electrodynamics*, 4th ed., §2.5.4) plus the square-lattice 2D Madelung
   constant \(M_\square = -3.90026492\) (Glasser & Zucker 1980 lattice sums)
   and Gaussian correction \(g(2) = -0.195386\). Verified against
   \(R\to\infty\) extrapolation of the lattice sum (651.81 V). Gated at
   1.5% (R=250 real-space image tail ≈ 0.8%). NB: the bare-sheet
   \(4\pi k_e\sigma d = 723.88\) V is **not** the site potential — the
   on-site lattice term contributes −72.07 V.
5. Exact conq twin input preserved at `inputs/lammps/in.conq_madelung` for
   re-pinning on any host with `PKG_ELECTRODE`.

## Error budget

Plate stiffness \(d\Delta\Psi/dQ \approx 4\pi k_e d/A \approx 724\) V/e
amplifies the v1 real-space-A vs `ewald/electrode` charge difference
(~2×10⁻⁴ e) to ≈ 0.14 V; the raw-\(Q_{\mathrm{pin}}\) V-response is
therefore only coarse-gated (±0.25 V) while the inverse contract carries the
tight gate. Details in the regression `REFERENCE.md`.

## Perf

`perf_exempt` — no `fix electrode/conq` in the local binary; no fair twin.
