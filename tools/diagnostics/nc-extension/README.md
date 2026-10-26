# External NC extension: actual QE reference

`qe-raw.txt` is the completed QE7.5 execution of `si2.in`, using
`Si_GTH_PBE_q4.gth`. The published block is transcribed verbatim in
`published-si-cp2k.txt` from the installed CP2K GTH_POTENTIALS. The full
parameter-source and executable hashes, exact invocation/environment, real
printed energy/forces and predeclared native bounds are in `reference.json`.
The provider example is `examples/nc-extension`; no native acceptance result
is recorded here yet.

Reproduce from this directory (or use the script, which sets the two requested
thread environment variables and extracts only the total force block):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /path/to/qe7.5/pw.x -in si2.in > qe-raw.txt
python capture.py /path/to/qe7.5/pw.x --parse-existing
# Or run+capture once:
python capture.py /path/to/qe7.5/pw.x
```

QE's native `Modules/read_pseudo.f90` dispatches `.gth` to
`upflib/gth.f90::readgth`. The ABINIT-format input declares Z14/Zval4,
pspcod10 (scalar NC, no NLCC), pspxc11 (PBE), lmax1 and local channel2;
the extra zero following the p block is its zero spin-orbit coefficient.
The GTH coefficients are in Hartree in this input format. QE's own local
analytic Fourier transform and analytic GTH projector transform handle their
internal Ry normalization; the file is not manually rescaled. In particular,
QE multiplies its projector transform by e²=2 and stores h/e², giving the
correct overall2 Ha-to-Ry factor. No implementation of that operator is added
to this directory.

The accepted execution converged in35 iterations: final total F at line700,
total forces at lines715–716. Verbosity=high also prints separate force
contributions; they are deliberately excluded from the pin extraction.
Energy includes the printed -TS. All six Cartesian force entries are nonzero.
Every scientific setting and original bound is documented in the example
README; no stress or live performance-ratio qualification is claimed.

`si2-fixed-diagnostic.in`/`qe-unconverged-fixed.txt` document the earlier
fixed-occupation failure in200 iterations. `unconverged-fixed.json` marks it
unaccepted; it contributes no numerical pin. Explicit finite-temperature
occupations were selected for this near-degenerate molecular system, rather
than taking a nonconverged fixed-occupation energy as a reference.
