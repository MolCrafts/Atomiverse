# gpw/pbe-family

Archived oracle record: periodic GPW PBE-family functionals compared with the
actual output of independent CP2K runs. The pins in `cp2k_rks_pins.h` and
`cp2k_uks_pins.h` were parsed from CP2K's own stdout, never generated from
Atomiverse.

## Common setup

Public `cpu::gpw::GpwDft` (and `cuda::gpw::GpwDft` when built with CUDA),
periodic orthorhombic 12 × 11 × 10 Bohr cell, DZV-GTH-PADE basis,
GTH-PBE-q1 potential, one 80 Ry density grid, local pseudopotential on the grid
(`CORE_PPL GRID`), integer occupations (a zero-temperature energy). CP2K names
the LibXC exchange/correlation pairs explicitly so the PBEsol/revPBE/RPBE
parameters are unambiguous:

| Variant | Exchange | Correlation |
| ------- | -------- | ----------- |
| PBE | GGA_X_PBE | GGA_C_PBE |
| PBEsol | GGA_X_PBE_SOL | GGA_C_PBE_SOL |
| revPBE | GGA_X_PBE_R | GGA_C_PBE |
| RPBE | GGA_X_RPBE | GGA_C_PBE |

| Item | Value |
| ---- | ----- |
| Engine | CP2K 2026.2 (`cp2k.psmp`, EasyBuild `CP2K/2026.2-gmpflf-2026.03`) |
| Binary SHA256 | `454f88e4070ce3679e0e2d10db9b90f2ff1ec5fc74f2afea18fd2d801c8b6f79` |
| Inputs | `cp2k-inputs/<case>/input.inp`, exactly as run |

CP2K 2026.2 aborts analytical stress for `CORE_PPL GRID`
("Virial not debuged for CORE_PPL"), so the pins use CP2K's
`STRESS_TENSOR NUMERICAL` (independently converged strained calculations, same
Hamiltonian). CP2K prints compression-positive stress in bar, converted to GPa
by 1e−4. The public orthorhombic GPW method supports only the diagonal
xx / yy / zz stress, so only those three lanes are compared; no shear support
is claimed.

Gates (all cases): energy 5e−6 Ha, force 5e−5 Ha/Bohr per component, diagonal
stress 5e−5 Ha/Bohr³ per lane. Every lane of every variant is measured before
the case reports failure.

## pbe-family-cp2k-rks — H₂ RKS, PBEsol / revPBE / RPBE (captured 2026-10-27)

Tilted H–H geometry in the cell above.

The CP2K PBEsol energy is −1.158385101258297 Ha. An earlier native energy
(−1.15839043956 Ha) missed CP2K's Gaussian-core overlap term (+5.33892824e−6 Ha,
just above the gate); with that term included the CPU case agrees to
6.3e−10 Ha (checked 2026-11-05 on this archive's sources).

## pbe-family-cp2k-uks — H₃ Γ UKS, PBE / PBEsol / revPBE / RPBE (captured 2026-10-28)

Three H atoms, integer Nα = 2 / Nβ = 1, at the PW H₃ geometry (in Å) in the
same cell; the single 80 Ry grid is 36 × 32 × 30. Each external
numerical-stress calculation converged twenty SCFs and printed spin counts 2/1
throughout. Besides the energy, all nine force components and the three
diagonal stress lanes, the case checks the electron counts from the public
density payload: Tr(PαS) = 2 and Tr(PβS) = 1.

The fixture does not qualify negative raw spin densities or the empty-β
endpoint; those derivative domains are separate open questions.

## Status

Last checked 2026-11-05 on CPU (Apple M4): all seven pins pass, energy within
1.1e−9 Ha, forces within 8.3e−8 Ha/Bohr, diagonal stress within 4.1e−9 Ha/Bohr³.
The CUDA backend has not been rechecked against this archive.

## References

- CP2K XC input: <https://github.com/cp2k/cp2k/blob/master/src/input_cp2k_xc.F>
- Perdew, Burke & Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE;
  Perdew et al., *Phys. Rev. Lett.* **100**, 136406 (2008) — PBEsol;
  Zhang & Yang, *Phys. Rev. Lett.* **80**, 890 (1998) — revPBE;
  Hammer, Hansen & Nørskov, *Phys. Rev. B* **59**, 7413 (1999) — RPBE.
