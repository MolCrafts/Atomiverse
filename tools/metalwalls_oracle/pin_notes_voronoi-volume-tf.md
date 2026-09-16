# Pin notes — `voronoi-volume-tf`

Slug: `voronoi-volume-tf` (wave 5) — Voronoi / site-volume companion for
Thomas–Fermi electrode metallicity (map §1 "Voronoi volume (TF)").

## Oracle decision (binding priority rule)

- LAMMPS (incl. ELECTRODE) has **no** Thomas–Fermi / Voronoi-volume electrode
  twin (perfect conductor only) → oracle binds to **MetalWalls**.
- No MetalWalls binary is installed on the agent host → pins are taken from
  the **MetalWalls TF model** (Scalfi et al., *J. Chem. Phys.* **153**, 174704
  (2020), DOI 10.1063/5.0028232; MW per-species electrode inputs
  `thomas_fermi_length` + `voronoi_volume`, gitlab `ampere2/metalwalls`
  branch `release`) evaluated offline by the discrete script below →
  `perf_exempt`.

## How the pins were taken (offline, never ctest)

```bash
python3 tools/metalwalls_oracle/inputs/metalwalls/voronoi-volume-tf/pin_voronoi_tf.py
# numpy 2.5.1, python3 on the x86_64 login node, 2026-07-27
```

The script mirrors the `ConstPotBO` Gaussian elastance (η = 1/0.56 Å⁻¹,
2D xy image half-range 40) on the wave-2 gold capacitor toy and applies the
MW TF diagonal with **per-site volumes**:

\[ A_{ii} \mathrel{+}= 4\pi k_e\,l_{\mathrm{TF}}^2 / v_i,\qquad
   l_{\mathrm{TF}} = 2\ \text{Å}. \]

## Pinned numbers (hard-coded in the regression)

| Quantity | Value | Units |
| --- | --- | --- |
| FCC Voronoi volume \(v = a_0^3/4\), \(a_0 = 4.07\) Å | **16.854785750000** | Å³ |
| \(Q_{\mathrm{top}}\) with \(v_i = a_0^3/4\) (case A) | **+3.934601758679e−03** | e |
| \(Q_{\mathrm{top}}\) with \(v_i = a_0^3/8\) (case B) | **+3.509853715789e−03** | e |
| \(Q_{\mathrm{top}}\) perfect metal \(l_{\mathrm{TF}}=0\) (wave-2 pin) | **+4.620945546104e−03** | e |

Case A equals the wave-2 fixed-density pin
(`pin_notes` for `thomas-fermi-electrode`, \(d = 4/a_0^3\)) to all printed
digits — supplying the bulk FCC Voronoi volume is algebraically identical to
the fixed-density foothold, which the regression asserts as an equivalence
invariant (≤ 1e−10 rel), plus the metallicity ordering
\(C(v/2) < C(v) < C(\text{perfect metal})\).

The FCC Voronoi volume itself is exact geometry (4 sites fill the
conventional cube \(a_0^3\); the FCC Voronoi cell is the rhombic
dodecahedron), so the `VoronoiVolume` pin needs no oracle binary.

## Perf

`perf_exempt: true` — no MetalWalls binary on the agent host for a fair wall
twin; LAMMPS has no TF/Voronoi electrode feature. The whole regression is a
sub-second algebraic solve + a 32-site tessellation.
