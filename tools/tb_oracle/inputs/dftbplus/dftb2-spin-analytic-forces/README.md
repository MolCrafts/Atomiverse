# DFTB+ 24.1 twin — `dftb2-spin-analytic-forces`

```text
oracle: dftb+
oracle_version: 24.1
```

- **Binary:** DFTB+ 24.1 (`source tools/tb_oracle/setup_dftbplus_env.sh`)
- **SK:** mio-1-1 `O-O.skf`
- **Geometry:** cluster (`C`) O₂, r = 1.30 Å (`geo.gen`)
- **Spin:** same block as `spin-dftb2-o2` (`ShellResolvedSpin = No`,
  `O = { -0.028 }`)
- **Forces:** `Analysis { PrintForces = Yes }`
- **Archived output:** `dftb.out.txt` = `detailed.out` verbatim
- **Total energy:** −6.4963341184 H = −176.774256878772 eV
- **Forces (Ha/Bohr):** \(F_1=(+0.095512870310,0,0)\),
  \(F_2=(-0.095512870310,0,0)\)

Re-run:

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh \
  tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-forces mio
```
