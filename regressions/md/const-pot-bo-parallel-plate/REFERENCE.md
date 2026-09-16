# Reference — constant-potential BO solve (parallel plate / madelung)

oracle: lammps  
oracle_version: 3 Nov 2022 (ELECTRODE package example `madelung`, log `log.1Dec2022.madelung-ewald-ew3dc.g++.1`)

## System

LAMMPS ELECTRODE **madelung** twin (Ahrens-Iwers et al., *J. Chem. Phys.* **157**,
084801 (2022), DOI [10.1063/5.0099239](https://doi.org/10.1063/5.0099239)):

| Species | Type | z [Å] | initial q [e] |
| --- | --- | --- | --- |
| Bottom electrode | 1 | −2 | 0 (solved) |
| Top electrode | 2 | +2 | 0 (solved) |
| Electrolyte + | 3 | −1 | +0.5 |
| Electrolyte − | 3 | +1 | −0.5 |

Box: \(L_x=L_y=1\) Å, \(L_z=20\) Å (slab-like; electrode solve uses **2D \(xy\)
Gaussian lattice images**, v1 — not full `ewald/electrode`). Atom \(z\) values are
the madelung relative coordinates shifted into \((0,L_z)\) so fully periodic ATV
boxes do not wrap the slab under VelocityVerlet (LAMMPS example uses
`boundary p p f`).

Applied potentials: \(\Psi_{\mathrm{bot}}=0\) V, \(\Psi_{\mathrm{top}}=1\) V,
\(\Delta\Psi=1\) V. Gaussian width \(\eta=2.0\) Å⁻¹. Electroneutrality on
(`symm`).

## Offline pin (LAMMPS ELECTRODE log)

From `~/work/lammps/examples/PACKAGES/electrode/madelung/log.1Dec2022.madelung-ewald-ew3dc.g++.1`
(`run 0` after `fix electrode/conp 0 2 couple top 1 symm on`,
`kspace_style ewald/electrode 1e-8`, `kspace_modify slab 8.0`):

| Observable | Value |
| --- | --- |
| \(Q_{\mathrm{bot}}\) [e] | **−0.279214485147238** |
| \(Q_{\mathrm{top}}\) [e] | **+0.279214485147238** |
| PE (real units, kcal/mol) | 139.943964815502 (not compared — unit system differs) |

## Literature cross-check (Madelung / plate_cap)

`plate_cap.py` in the same example directory: pure real-space 2D Gaussian
lattice sum of the BO matrix (Siepmann–Sprik Gaussian charges). With metal
units (`units::coulomb = 14.399645`) and image range ≳ 200:

\[
Q_{\mathrm{top}}^{\mathrm{Madelung}} \approx +0.27941\,e
\]

(relative difference vs LAMMPS Ewald-electrode pin ≈ **0.07%**).

## What Atomiverse asserts

Public API: `cpu::md::ConstPotBO` on `md::Phase::pre_force`, writes
`kAtom_charge`.

| Check | Tol |
| --- | --- |
| \(\lvert\sum Q_i\rvert\) | \(\le 10^{-10}\) e |
| \(\lvert Q_{\mathrm{top}} - 0.279214485147238\rvert / 0.279214485147238\) | **≤ 1%** (v1 2D-image Gaussian \(A\) vs full Ewald electrode) |
| \(\mathrm{sign}(Q_{\mathrm{top}})=+1\) for \(\Delta\Psi=+1\) V | exact |

## Approximation note (map E1)

v1 builds \(A\) from Gaussian real-space + 2D \(xy\) images only. Full
`ewald/electrode` / MetalWalls reciprocal electrode matrix is **E2**
(`slab-2d-ewald-foothold`). The 1% gate is deliberate for this foothold.

## Perf

```
perf_exempt: true
perf_exempt_reason: local LAMMPS binary lacks ELECTRODE package; no fair twin binary for fix electrode/conp
ratio_atv_over_oracle: n/a
within_10x: exempt
```

## Repro of pin (offline only — never ctest)

```bash
# Requires LAMMPS built with -D PKG_ELECTRODE=ON
cd $LAMMPS_DIR/examples/PACKAGES/electrode/madelung
lmp -in in.ewald-ew3dc
# thermo: pe c_qbot c_qtop at step 0
```

Or literature Madelung:

```bash
python3 plate_cap.py   # writes plate_cap.csv (real units)
```
