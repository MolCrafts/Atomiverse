# External XC reference output

Oracle: PySCF 2.14.0, actually executed with
`/tmp/atv-xc-oracle/bin/python` (Python 3.13, macOS arm64),
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```sh
python tools/pyscf_oracle/pin_scf.py rks --geom water --basis sto-3g --xc '0.8*LDA_X,' --grad > water.json
python tools/pyscf_oracle/pin_scf.py uks --geom h2-1A --basis sto-3g --charge 1 --spin 1 --xc '0.8*LDA_X,' --grad > h2plus.json
```

JSON `e_tot_ha` and every `grad_ha_bohr` entry are the oracle output lines.
Inputs and tolerances are recorded in `examples/xc-extension/README.md` and
`regressions/scf/{rks,uks}/REFERENCE.md`. No Atomiverse value was used to
construct these pins. Tests read frozen constants and do not launch PySCF.
