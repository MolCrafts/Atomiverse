# Offline pin: dftb2-dispersion-d3

Geometry-only D3(BJ) energy for the water dimer used by
`reg_dftb2-dispersion-d3`.

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/Simple-DFTD3/1.2.1-gfbf-2025b/bin:$PATH
s-dftd3 --bj dftb3 --json pin.json dimer.xyz
# E_disp = -1.6892886213669401e-3 Ha
```

`oracle: literature` / Simple-DFTD3 1.2.1. DFTB+ is not required for the pure
dispersion pin (same D3 model DFTB+ uses for D3).
