# Spec — `gfn2-xtb-water`

Wave 7, `tb-method-loop`, Phase C (GFN-xTB, oracle = **xtb**).

## Summary

Production **GFN2-xTB** energy on the public `cpu::semiempirical` surface.
Pin H₂O against the archived xtb **6.7.1** single-point
`tools/tb_oracle/inputs/xtb/h2o_gfn2/xtb.out`
(**−5.070383095219 Eh**). Native static in-tree tables; **do not** wrap
`libxtb` (the leftover `ATV_XTB_ROOT` CMake path is deleted with this slug).

## Domain basis

Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652 (2019),
DOI 10.1021/acs.jctc.8b01176. Family-locked oracle: **xtb only**.

Hamiltonian (molecular, closed shell, energy-only this slug):

- STO-*n*G valence basis (H: 1s; C/N/O: 2s2p)
- CN-shifted H⁰ with electronegativity and shell-polynomial scaling
- Shell-resolved isotropic Klopman–Ohno SCC (gexp = 2, arithmetic Hubbard)
- On-site third-order
- Anisotropic electrostatics + AXC kernels
- Screened Coulomb repulsion
- Grimme D4 (GFN2 BJ: s8=2.7, a1=0.52, a2=5.0, s9=5.0)

Parameters transcribed from the published GFN2-xTB set
(`param_gfn2-xtb.txt` / tblite `xtb/gfn2.f90`). Internal conversion
`eV → Eh` uses xtb/tblite’s `1/27.21138505`; Frame energy is written in
ATV eV (`units::kHartreeInEv`).

## Geometry convention

The archived xtb transcript does not reprint `mol.xyz` (gitignored `*.xyz`
at the time of the run). The geometry is the **same off-equilibrium water**
used by every in-tree SE H₂O pin (Å):

| Atom | x     | y    | z   |
|------|-------|------|-----|
| O    | 0.00  | 0.00 | 0.0 |
| H    | 0.96  | 0.00 | 0.0 |
| H    | −0.24 | 0.93 | 0.0 |

This is consistent with the transcript: slightly split H Mulliken charges
(0.282 / 0.281), dipole in the *xy* plane, nonzero gradient norm
(0.02286 Eh/a₀). Input twin: `tools/tb_oracle/inputs/xtb/h2o_gfn2/mol.xyz`.

## Design

Public type `cpu::semiempirical::GFN2` (kind `cpu::Semiempirical` +
`cpu::QmEngine`). Members: `charge`, `scc_tol`, `max_scc`, `mix_beta`,
`with_forces` (default false; this slug is energy-only — forces throw
naming `gfn2-xtb-forces`). No `*Config`, no factory. Frame in / Frame out
via `Evaluate` + `on(evaluate::Phase::evaluate).add`.

Unsupported Z throws with the supported set (H/C/N/O this foothold).
Open shell / PBC / ALPB throw naming the pending slug.

## Files

| Path | Action |
| ---- | ------ |
| `include/atomiverse/cpu/semiempirical/gfn2.h` | new |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn2_xtb_tables.h` | new |
| `src/cpu/semiempirical/gfn2.cpp` | new |
| `tests/cpu/semiempirical/test_gfn2_params.cpp` | new |
| `regressions/semiempirical/gfn2-xtb-water/` | new |
| `CMakeLists.txt` | compile `gfn2.cpp` under SE; drop libxtb wrap |
| `tests/CMakeLists.txt` | register unit test |
| `regressions/CMakeLists.txt` | `atv_add_regression(semiempirical gfn2-xtb-water)` |
| `tools/tb_oracle/inputs/xtb/h2o_gfn2/` | README + `mol.xyz` beside archived `xtb.out` |

## Out of scope

- Analytic forces (`gfn2-xtb-forces`)
- ALPB/GBSA (`alpb-xtb`)
- GFN0 / GFN1 (sibling slugs)
- Spin / PBC / IPEA / sTDA
- Wrapping libxtb / libtblite
