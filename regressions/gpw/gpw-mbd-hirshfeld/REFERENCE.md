# Reference — water-dimer GPW PBE + Hirshfeld-scaled MBD@rsSCS

## Method

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) from the final Γ-point
**GPW PBE** valence density, used to density-scale free-atom **MBD@rsSCS**
parameters (Tkatchenko & Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009);
Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014)).

Public API:

```cpp
GpwDft gpw;
gpw.mbd = true;
gpw.hirshfeld = true;   // post-SCF volumes → mbd_rsscs_pbe(…, vol_ratio)
```

Dispersion energy on `GpwResult::e_disp` [Hartree]; ratios on
`GpwResult::volume_ratios`. Free-atom path (`hirshfeld=false`) remains
geometry-only (`reg_gpw-mbd-ice`).

Free-atom valence densities (in-tree exponential model,
`basis_data/hirshfeld_tables.h`):

\[
\rho_Z^{\mathrm{free}}(r)=A\exp(-\alpha_Z r),\quad
A=N_{\mathrm{val}}\alpha_Z^3/(8\pi),\quad
V_Z^{\mathrm{free}}=60\,N_{\mathrm{val}}/\alpha_Z^3
\]

H: \(N=1\), \(\alpha=2.0\); O: \(N=6\), \(\alpha=2.2\) (Bohr⁻¹).

Scaling of free-atom α₀ / C₆ / R_vdW (libMBD tables): \(\alpha\propto v\),
\(C_6\propto v^2\), \(R\propto v^{1/3}\). MBD@rsSCS is **isolated** on the
nuclear coordinates (periodic Ewald MBD follow-on).

## Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS, Hirshfeld volumes, water dimer, CUTOFF 200 Ry) =
−1.1479852469005181×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Note | E_disp [Ha] |
| ------ | ---- | ----------- |
| **ATV product** (`GpwDft` mbd+hirshfeld, CUTOFF 200, tol 1e-6) | dual gate | **−1.1479852469005181×10⁻³** |
| Free-atom MBD (volume_ratio = 1) | `reg_gpw-mbd-ice` / libMBD Fortran | −1.4928804171407961×10⁻³ |
| Dual pure-Python Hirshfeld (same α tables) | free-atom limit on isolated O | \(v\approx 1\) (unit test) |

Hirshfeld scaling reduces |E_disp| vs free-atom (~23% less attractive on this
geometry), as expected for compressed molecular volumes (\(v_A\in[0.58,0.74]\)).

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Offline twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-hirshfeld/`. Free-atom MBD already pinned vs
libMBD Fortran (`gpw-mbd-ice`); this slug pins the **volume-ratio layer**.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
MBD pin is isolated on the same nuclear coordinates with Hirshfeld \(v_A\).

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).
CUTOFF **200 Ry**, max_iter 16, tol 1e-6, mix 0.4, PBE.

Representative volume ratios at the pin (order O,H,H,O,H,H):

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784`

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests: free-atom limit \(v\approx 1\)
to 3e-2; volume-scaled MBD dual to 1e-15; product path physical bounds.

## Perf gate

**perf_exempt = true** — primary deliverable is E_disp + volume-ratio parity.
Hirshfeld grid integral is \(O(N_{\mathrm{grid}} N_{\mathrm{atoms}})\); SCF dominates
wall (same policy as `gpw-mbd-ice` / `gpw-ts-ice`).
