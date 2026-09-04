# DFTB+ 24.1 twin — `spin-dftb2-o2`

```text
oracle: dftb+
oracle_version: 24.1
```

- **Binary:** `molcrafts/software/dftbplus-24.1/bin/dftb+` (`source tools/tb_oracle/setup_dftbplus_env.sh`)
- **SK:** mio-1-1 `O-O.skf` (`ATV_DFTBPLUS_SK_MIO`)
- **Geometry:** cluster (`C`) O₂, r = 1.21 Å (`geo.gen`)
- **Spin:** `SpinPolarisation = Colinear { UnpairedElectrons = 2 }`,
  `SpinConstants { ShellResolvedSpin = No; O = { -0.028 } }`
  (HOAO \(W_{\mathrm{pp}}\) of the published PBE table
  \(\{-0.035,-0.030,-0.030,-0.028\}\) Ha)
- **Archived output:** `dftb.out.txt` = `detailed.out` verbatim
- **Total energy:** −6.5047954387 H = −177.004501133607 eV
- **Energy SPIN:** −0.0280000000 H = −0.761918814888 eV

Re-run:

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/spin-dftb2-o2 mio
```

ctest never launches DFTB+; numbers are hard-coded in
`regressions/semiempirical/spin-dftb2-o2/`.
