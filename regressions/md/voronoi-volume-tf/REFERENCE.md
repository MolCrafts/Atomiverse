# voronoi-volume-tf

oracle: metalwalls
oracle_version: MetalWalls release 20.05, git **c8ad175d** (release branch,
2026-06-18), binary at
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
oracle_kind: real binary (re-pinned 2026-07-29; supersedes the wave-5
`pin_voronoi_tf.py` self-golden)

## Physics

Thomas–Fermi electrode metallicity with **per-site Voronoi volume** as the
local volume that enters the TF diagonal correction (Scalfi et al., J. Chem.
Phys. **153**, 174704 (2020); MetalWalls `voronoi_volume` species keyword).
ATV: `ConstPotBO::site_volume` + `cpu::VoronoiVolume` for geometry-derived
\(v_i\).

## Geometry (matched twin of `thomas-fermi-electrode`)

Empty Au capacitor: 3+3 collinear sites, \(a_0 = 4.07\) Å, plane spacing
\(a_0/2\), gap 20 Å, \(\Delta\Psi = 1\) V, \(\eta = 1/0.56\) Å⁻¹,
`charge_neutrality true`, `num_pbc 2`.

Twin inputs (templates with `@LTF@` / `@VORONOI@`):
`tools/metalwalls_oracle/inputs/metalwalls/voronoi-volume-tf/mw_twin/`.

| case | `thomas_fermi_length` [bohr] | `voronoi_volume` [bohr³] |
| ---- | --------------------------- | ------------------------ |
| perfect metal | 0 | \(a_0^3/4\) (unused at \(l_{\mathrm{TF}}=0\)) |
| TF bulk \(v\) | \(2\,\mathrm{Å}/a_0\) = 3.779452249251540 | 113.741732075823890 |
| TF half \(v\) | same | 56.870866037911945 |

## Pin (numbers MetalWalls printed)

From `total_charges.out` step 0 (`|Q_top|` column):

| case | \(Q_{\mathrm{top}}\) [e] |
| ---- | ------------------------- |
| \(l_{\mathrm{TF}}=0\) | **4.354124119406114×10⁻³** |
| \(l_{\mathrm{TF}}=2\) Å, \(v=a_0^3/4\) | **3.707555065364343×10⁻³** |
| \(l_{\mathrm{TF}}=2\) Å, \(v=a_0^3/8\) | **3.321360199637453×10⁻³** |

These match the real-binary `thomas-fermi-electrode` re-pin for the
fixed-density perfect-metal / TF cases. Halved Voronoi volume hardens the
diagonal and lowers \(Q\) (metallicity ordering).

## Superseded self-golden

Wave-5 pins \(4.6209\times10^{-3}\) / \(3.9346\times10^{-3}\) /
\(3.5099\times10^{-3}\) came from `pin_voronoi_tf.py`, a numpy transcription
of ATV's own truncated elastance at `xy_image_range = 40` — **~6.13 % high**
vs the real binary (same defect as the pre-repin `thomas-fermi-electrode`
numbers; see `.claude/notes/oracle-pin-provenance.md`).

## Assertions

| check | tolerance |
| ----- | --------- |
| FCC Voronoi \(v_i = a_0^3/4\) | 1e-9 rel (geometry) |
| \(Q_{\mathrm{top}}\) vs MW pins above | 1 % rel |
| \(Q(v) = Q(d=4/a_0^3)\) self-consistency | 1e-10 rel |
| \(C(v/2) < C(v) < C(l_{\mathrm{TF}}=0)\) | ordering |

`xy_image_range = 1280` so the truncated image sum approaches MW's 2D-Ewald
elastance (same treatment as the TF re-pin).

## Perf

`perf_exempt` — tiny elastance; wall ratio would measure Ewald setup.
EOF