# Reference — water-dimer multi-k GPW PBE + Hirshfeld-scaled MBD@rsSCS

## Method

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) from the final **multi-k**
(Monkhorst–Pack 2×2×2, time-reversal) **GPW PBE** valence density, used to
density-scale free-atom **MBD@rsSCS** parameters (Tkatchenko & Scheffler,
*Phys. Rev. Lett.* **102**, 073005 (2009); Ambrosetti et al., *J. Chem. Phys.*
**140**, 18A508 (2014)). Extends Γ-only Hirshfeld (`reg_gpw-mbd-hirshfeld`).

Public API:

```cpp
GpwDft gpw;
gpw.mbd = true;
gpw.hirshfeld = true;
gpw.k_mesh[0] = gpw.k_mesh[1] = gpw.k_mesh[2] = 2;
```

Dispersion energy on `GpwResult::e_disp` [Hartree]; ratios on
`GpwResult::volume_ratios`. Free-atom multi-k path (`hirshfeld=false`) remains
geometry-only pre-SCF. Isolated MBD (default `mbd_k_mesh={0,0,0}`).

Free-atom valence densities (in-tree exponential model,
`basis_data/hirshfeld_tables.h`): same as `gpw-mbd-hirshfeld`.

Scaling of free-atom α₀ / C₆ / R_vdW: \(\alpha\propto v\), \(C_6\propto v^2\),
\(R\propto v^{1/3}\).

## Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS, multi-k Hirshfeld volumes, water dimer, CUTOFF 200 Ry,
k = 2×2×2, TR fold) = −1.1479849553026611×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Note | E_disp [Ha] |
| ------ | ---- | ----------- |
| **ATV product** (`GpwDft` mbd+hirshfeld, k=2×2×2, CUTOFF 200, tol 1e-6) | dual gate | **−1.1479849553026611×10⁻³** |
| Γ Hirshfeld MBD (same geom, no k-mesh) | `reg_gpw-mbd-hirshfeld` | −1.1479852469005181×10⁻³ |
| Free-atom MBD (volume_ratio = 1) | `reg_gpw-mbd-ice` | −1.4928804171407961×10⁻³ |

On this 12 Å molecular cell the multi-k vs Γ density difference is tiny
(\(\Delta E_{\mathrm{disp}}\sim 3\times 10^{-10}\) Ha); the hard pin freezes the
multi-k product path (n_kpts = 4 with TR fold).

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Offline twin notes under
`tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoints/`. Free-atom / Γ Hirshfeld
already pinned; this slug pins the **k-summed density → volume-ratio** layer.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**, **k-mesh 2×2×2**
(Monkhorst–Pack, time-reversal fold → 4 irreducible k).

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).
CUTOFF **200 Ry**, max_iter 16, tol 1e-6, mix 0.4, PBE.

Representative volume ratios at the pin (order O,H,H,O,H,H):

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784`

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests: multi-k path reports
volumes + attractive E_disp; UKS + hirshfeld still throws.

## Perf gate

**perf_exempt = true** — primary deliverable is multi-k E_disp + volume-ratio
parity. SCF dominates wall (same policy as `gpw-mbd-hirshfeld` / `gpw-mbd-ice`).
