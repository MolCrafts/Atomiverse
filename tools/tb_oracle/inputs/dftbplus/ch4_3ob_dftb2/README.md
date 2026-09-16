# Offline DFTB+ pin — CH4 / 3ob / DFTB2

```text
oracle: dftb+
oracle_version: 24.1
```

## How the pin was taken

1. Build DFTB+ 24.1 (module `buildenv-gcc/2026.03-mpich`).
2. SK files: DFTB.org 3ob-3-1 (`H-H`, `C-C`, `C-H`, `H-C`).
3. Geometry: CH4 tetrahedral \(r_{\mathrm{CH}}=1.09\) Å (`geo.gen`).
4. `dftb_in.hsd`: `Hamiltonian = DFTB { SCC = Yes; … }` — **no** ThirdOrder.
5. Result: `Total Energy: -3.2291103259 H` (−87.8686 eV).

Inputs and `detailed.out` may live under `/tmp/tb_oracle_dftb2_3ob_organic/`
on the build host; the hard-coded numbers are in
`regressions/semiempirical/dftb2-3ob-organic/REFERENCE.md`.
