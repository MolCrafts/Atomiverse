# Spec — `gpw-bands-si`

## Goal

GPW band-structure post-process on a high-symmetry Bloch path after multi-k
closed-shell SCF: freeze the self-consistent density / \(V_\mathrm{eff}\),
rebuild \(H_k\) on the path, and diagonalize for \(\varepsilon_n(\mathbf{k})\).
Matched twin: CP2K Quickstep `&PRINT &BAND_STRUCTURE` on GTH-GPW diamond Si.

## Public API

```cpp
cpu::gpw::GpwDft dft;          // multi-k RKS members (k_mesh, cutoff, …)
dft.build(atoms);
cpu::gpw::Bands bands;         // members: n_bands, kfrac / set_diamond_path
auto br = bands.compute(dft, atoms);
// br.eigenvalues_ha[ik][ib], gap_ha, e_vbm_ha, e_cbm_ha, scf energy
```

No `*Config`, no factories. Parameters are members of `Bands` / `GpwDft`.

## Algorithm

1. **SCF** — `method.run(atoms)` with the method’s Monkhorst–Pack / explicit
   mesh (product: fixed integer occupations, closed-shell RKS, single-grid,
   local PP on grid).
2. **Freeze \(V_\mathrm{eff}\)** — collocate \(\rho=\sum_k w_k\rho_k\) from the
   converged \(P_k\), Poisson + XC + \(V_\mathrm{loc}^\mathrm{SR}\) once.
3. **NSCF path** — for each fractional path k: rebuild \(S_k,T_k,V_\mathrm{nl}^k\),
   integrate frozen \(V_\mathrm{eff}\) into \(V^k\), diagonalize
   \(H_k=T_k+V_\mathrm{nl}^k+V^k\).
4. **Gap** — VBM/CBM from `n_occ` occupied bands across the path spectrum
   (Hartree).

## Product scope (this slug)

- Closed-shell RKS, fixed occupations, single-grid, periodic orthorhombic.
- Rejects: UKS, metallic smear, multigrid, nonperiodic, `local_pp_on_grid=false`,
  DFT+U, dispersion flags (out of scope for the twin).

## System / twin

- Diamond Si, bct 4-atom cell (same as `gpw-kpoints-si`): \(a=5.43\) Å,
  cell \((a/\sqrt2,a/\sqrt2,a)\), DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE,
  CUTOFF 150 Ry, MP \(2\times2\times2\), EPS_SCF \(10^{-8}\).
- Path: Γ–X–W–L–Γ–K fractional (same SPECIAL_POINT list as PW diamond path),
  `n_seg=4`.
- Offline CP2K: `tools/cp2k_oracle/inputs/gpw-bands-si.inp` — ENERGY +
  `&BAND_STRUCTURE` with matching SPECIAL_POINT set and ADDED_MOS.

## Acceptance

| Gate | Check |
| --- | --- |
| A1 | SCF converges; path eigenvalues sorted ascending per k |
| A2 | `n_bands` bands per path k; path includes Γ and X |
| A3 | Positive min local gap; positive path gap (insulating Si) |
| A4 | \(\|E_\mathrm{tot}-E_\mathrm{CP2K}\|\le 10^{-3}\) Ha |
| A5 | Gap and selected \(\varepsilon\) (Γ VBM, X CBM) within twin window |
| Perf | Fair wall: ATV SCF(+NSCF) vs same CP2K ENERGY+BAND_STRUCTURE job, ≤ 10× |

## Out of scope

GPW DOS (`gpw-dos-si`), spin/metallic bands, multigrid NSCF, Frame I/O of bands.
