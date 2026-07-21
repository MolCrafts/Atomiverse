# Example results (literature-facing numbers)

Machine-readable outputs captured from the numbered tutorial binaries under
`examples/`. Documentation embeds these values in tables and MolPlot charts —
**do not invent numbers for the docs**. When an example changes, re-run it and
refresh the matching JSON:

```bash
cmake -B build -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON
cmake --build build --target 02_rhf 03_uhf 04_uks 07_md_lj \
  11_koopmans 12_delta_scf 13_mp2 14_fci 15_am1 -j
# then re-run each binary and update the JSON below
```

| File | Source binary | What it holds |
|------|---------------|---------------|
| `02_rhf.json` | `02_rhf` | H₂ / STO-3G RHF vs Szabo & Ostlund |
| `03_uhf_pes.json` | `03_uhf` | H₂ RHF/UHF dissociation scan |
| `04_uks.json` | `04_uks` | Atomic UKS-LSDA + H₂ UHF/UKS scan |
| `05_geom_opt.json` | `05_geom_opt` | Aspirin L-BFGS + RHF/STO-3G trajectory |
| `07_md_lj.json` | `07_md_lj` | NVE LJ argon energy drift |
| `08_pw.json` | `08_pw` | Si plane-wave LDA (tutorial ecut/k) |
| `09_gpw.json` | `09_gpw` | H₂O GPW-PBE vs CP2K |
| `11_koopmans.json` | `11_koopmans` | Koopmans IP vs experiment |
| `12_delta_scf.json` | `12_delta_scf` | ΔSCF IP vs Koopmans |
| `13_mp2.json` | `13_mp2` | H₂O MP2 correlation |
| `14_fci.json` | `14_fci` | H₂ FCI vs RHF |
| `15_am1.json` | `15_am1` | AM1 water energy, ΔH_f, charges |
| `16_density_cube.json` | `16_density_cube` | H₂ density/HOMO cubes + ∫ρ |
| `h2_density.cube` / `h2_homo.cube` | `16_density_cube` | Gaussian cube files for MolVis |

All energies are from Atomiverse on the machine that last refreshed this
directory; the `reference` fields cite literature / experiment, never a golden
snapshot of our own prior output.
