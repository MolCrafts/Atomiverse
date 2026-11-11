# DFTB+ 24.1 non-SCC twins for `dftb0-analytic-forces`

Offline pins for `regressions/semiempirical/dftb0/`.
Each subdirectory is one `SCC = No` mio-1-1 single point with
`PrintForces = Yes`. `dftb.out.txt` is the verbatim `detailed.out`.
`Prefix = "./"` is rewritten to the mio-1-1 tree by
`tools/tb_oracle/run_dftbplus_pin.sh`.

| Dir | Geometry | Why |
| --- | -------- | --- |
| `h2_074` | H₂, R = 0.74 Å | energy must stay on the published −18.36635052 eV pin; force is not zero |
| `h2_090` | H₂, R = 0.90 Å | stretched ss + repulsive force |
| `h2o` | off-eq water | heteronuclear; an SCC run is not a DFTB0 pin |
| `co` | C(0,0,0), O(1.20, 0.15, 0) Å | ppσ/ppπ and a transverse force |

Run (not from ctest):

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/dftb0-analytic-forces/h2_090 mio
```
