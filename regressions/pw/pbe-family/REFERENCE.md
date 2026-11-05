# pw/pbe-family

Archived oracle record: periodic plane-wave PBEsol / revPBE / RPBE compared with
the actual output of an independent Quantum ESPRESSO run. The pins in
`qe_pins.h` were parsed from QE's own stdout, never generated from Atomiverse.

## pbe-family-qe — Si₂ RKS and H₃ UKS against QE 7.6

### Method

Public `cpu::pw::PW` (and `cuda::pw::PW` when built with CUDA) with
`XCType::PBEsol`, `RevPBE` or `RPBE`, analytic GTH pseudopotentials,
Fermi–Dirac smearing of width 0.02 Ry, Γ only. Each case runs `forces()` and
then `stress()` at the same geometry.

| Case | System | Cell | Cutoff (Ry) | Spin |
| ---- | ------ | ---- | ----------- | ---- |
| `si2-rks-{pbesol,revpbe,rpbe}` | Si₂, GTH-PBE-q4 | cubic 6 Å | 12 / 48 | restricted, 6 bands |
| `h3-uks-{pbesol,revpbe,rpbe}` | H₃, GTH-PBE-q1 | skew cell (4.2, .3, .2 / 4, .1 / 3.8 Å) | 6 / 24 | collinear doublet, initial magnetization 1/3, 5 bands |

The same GTH-PBE pseudopotentials are used unchanged across the three XC
variants; this compares one Hamiltonian across programs and makes no claim of
pseudopotential transferability. QE's `input_dft` overrides the XC stated in the
pseudopotential, and the printed XC label of every run matched the requested
variant.

### Oracle

| Item | Value |
| ---- | ----- |
| Engine | Quantum ESPRESSO 7.6 (`pw.x`, EasyBuild `QuantumESPRESSO/7.6-gmpflf-2026.03`) |
| Binary SHA256 | `5e03776776ed71bf3f6e1cc01f46a8216a680a0179eaf41fdb3a8e99d6f89f5b` |
| Captured | 2026-10-27, one `srun` step per run, 4 OpenMP threads, single-threaded BLAS |
| Inputs | `qe-inputs/<case>/input.in` and the `.gth` file beside it, exactly as run |

Compared quantities and conversions:

- **Energy:** the finite-temperature Mermin free energy including QE's printed
  entropy term, in Ha (QE Ry / 2).
- **Forces:** every Cartesian ionic-force component, Ha/Bohr.
- **Stress:** all six physical components, compression-positive
  σ = −∂E/∂ε / Ω; QE prints Ry/Bohr³, divided by 2. Public `kStress` is the
  extensive virial σΩ in eV; `stress()` returns eV/Å³.

### Gates

| Quantity | Tolerance |
| -------- | --------- |
| Energy | 1e−5 Ha |
| Force, per component | 1e−4 Ha/Bohr (the force pin must exceed 1e−3 Ha/Bohr) |
| Stress, per component | 2e−3 eV/Å³ |
| Public energy publication | `kEnergy` = energy (Ry) × QE's Rydberg-to-eV constant, 1e−9 |
| Occupied electrons | 8 (Si₂) / 3 (H₃), 1e−8; Mermin entropy term < −1e−5 |
| Spin | \|magnetization\| > 0.1 for H₃ |

The energy publication uses QE's documented Rydberg-to-eV conversion while
forces and stress use the core unit conversion; mixing the two constants
produces a false sub-micro-eV publication failure.

### Status

Last checked 2026-11-05 on CPU (Apple M4): all six pins pass, energy within
1.5e−6 Ha, forces within 1.1e−5 Ha/Bohr, stress within 1.2e−5 eV/Å³. The CUDA
backend has not been rechecked against this archive.

### References

- QE 7.6 XC dispatch: <https://github.com/QEF/q-e/blob/qe-7.6/Modules/funct.f90>
- Perdew et al., *Phys. Rev. Lett.* **100**, 136406 (2008) — PBEsol;
  Zhang & Yang, *Phys. Rev. Lett.* **80**, 890 (1998) — revPBE;
  Hammer, Hansen & Nørskov, *Phys. Rev. B* **59**, 7413 (1999) — RPBE.
