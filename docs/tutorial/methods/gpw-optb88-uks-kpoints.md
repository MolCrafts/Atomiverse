# Multi-k UKS GPW — optB88-vdW (OptB88 + DRSLL)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-optb88-uks-kpoints` |
| Regression | `regressions/gpw/gpw-optb88-uks-kpoints/` |
| Public API | `cpu::gpw::GpwDft` (`n_alpha`/`n_beta`, `XcKind::OptB88VdW`, `vdw_df`, `k_mesh` / `k_points`) |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion — multi-k UKS optB88-vdW |
| Fair wall | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx \mathbf{0.73}\) (≤10×) |

## Summary

**Collinear unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** on a
**Monkhorst–Pack k-mesh** with **optB88-vdW**: optimized B88 exchange
(`GGA_X_OPTB88_VDW`) + **PW92** local correlation + **self-consistent DRSLL /
vdW-DF** nonlocal correlation on the k-summed total density. Pins the dual-spin
k-path that previously only had Γ UKS (`gpw-optb88-uks`) and multi-k RKS
(`gpw-optb88-kpoints`) product pins.

Public path: set `n_alpha`/`n_beta`, `xc = XcKind::OptB88VdW`, `vdw_df = true`,
and `k_mesh` (or explicit `k_points`) on `GpwDft`, then `build` / `run`.
Hartree and \(E_c^{\mathrm{nl}}\) via `GpwResult::{e_total,e_disp}`. Product
gate: `reg_gpw-optb88-uks-kpoints` — Γ-limit identity, OptB88-vs-PBE dual gate,
absolute \(E_{\mathrm{tot}}\) + tight \(E_{\mathrm{disp}}\) vs offline CP2K, and
closed-shell multi-k UKS≡RKS identity.

Related: `gpw-optb88-uks`, `gpw-optb88-kpoints`,
`gpw-vdw-uks-kpoints`. Narrative: [Ch. 9 GPW](../ch09-gpw.md).

## Method

| Piece | Form |
| ----- | ---- |
| Exchange | optB88 (`XcKind::OptB88VdW` → `xc_grid_optb88_vdw_spin`) |
| Local correlation | PW92 LDA only |
| Nonlocal | DRSLL SC \(V_{\mathrm{nl}}\) on \(\rho_\alpha+\rho_\beta\) (`build_uks_k_energy`) |
| Post-SCF \(E_c^{\mathrm{nl}}\) | `fold_dispersion_k_uks` → `e_disp` [Ha] |

```cpp
ATV::cpu::gpw::GpwDft gpw;
// cell, cutoff_ry, atoms ...
gpw.n_alpha = 8;
gpw.n_beta = 8;
gpw.xc = ATV::cpu::gpw::XcKind::OptB88VdW;
gpw.vdw_df = true;
gpw.k_mesh = {2, 1, 1};
gpw.k_time_reversal = false; // 1:1 with CP2K full MP mesh
gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_total, res.e_disp [Ha]
```

## Offline CP2K twin

Water dimer, LSD MULTIPLICITY 1, MP 2×1×1, NGRIDS 4 / REL_CUTOFF 40,
`GGA_X_OPTB88_VDW` + `PW92` + `NON_LOCAL TYPE DRSLL`. Input:
`tools/cp2k_oracle/inputs/gpw-optb88-uks-kpoints/water_dimer_uks_optb88_vdw_kp.inp`.

| Quantity | CP2K 2026.1 (`git:5e54ba2`) |
|----------|----------------------------|
| E_tot [Ha] | −34.487914403936173 |
| E_disp [Ha] | +0.15206184094414 |
| wall (1×8 OMP) | 13.29 s |

Full provenance and ATV gates: [`REFERENCE.md`](../../../regressions/gpw/gpw-optb88-uks-kpoints/REFERENCE.md).

## Build & run

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-optb88-uks-kpoints
  OMP_NUM_THREADS=8 ./build/regressions/reg_gpw-optb88-uks-kpoints
'
```

## Citations

- J. Klimeš, D. R. Bowler, A. Michaelides, *J. Phys.: Condens. Matter*
  **22**, 022201 (2010).
- M. Dion et al., *Phys. Rev. Lett.* **92**, 246401 (2004).
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009).
