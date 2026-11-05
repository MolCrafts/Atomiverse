# QE twin for the external NC extension example

Pinned inputs for the energy and forces in
[`examples/extensions/nc/pins.h`](../../../../examples/extensions/nc/pins.h):

- `si2.in`: periodic cubic 6 Å cell, Si(2,2,2) and Si(3.8,3.2,3.0) Å, PBE,
  Γ, ecutwfc 12 Ry / ecutrho 48 Ry, 40³ dense and smooth grids, restricted
  Fermi–Dirac smearing 0.02 Ry, six bands, conv_thr 1e-11 Ry.
- `Si_GTH_PBE_q4.gth`: the published Si GTH-PBE-q4 block in ABINIT format.
- `published-si-cp2k.txt`: the same block transcribed verbatim from the
  installed CP2K `GTH_POTENTIALS`.

Capture (runs QE once, serial, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`):

```sh
python tools/qe_oracle/inputs/nc-extension-si2/capture.py /path/to/qe-7.5/bin/pw.x --out OUT
```

`OUT` must not exist. It receives the raw stdout (`qe-raw.txt`) and
`reference.json` with the command, executable and input hashes and the
printed energy and forces. `--parse-existing` re-reads `OUT/qe-raw.txt`
without running QE. Neither file is committed; transcribe the printed numbers
into `pins.h`.

The current pin is QE 7.5 (reported commit
`770a0b2d12928a67048e2f3da8d10d057e52179e-dirty`). It converged in 35
iterations; the pinned values are the final Mermin free energy (including
-TS) and the total-force block. QE dispatches `.gth` to
`upflib/gth.f90::readgth` and uses its own analytic local and projector
Fourier transforms. The input declares Z 14 / Zval 4, pspcod 10 (scalar NC, no
NLCC), pspxc 11 (PBE), lmax 1 and local channel 2, with coefficients in
Hartree. QE multiplies its projector transform by e² = 2 and stores h/e², so
the file isn't rescaled by hand.

An earlier fixed-occupation run of the same geometry didn't converge in 200
iterations and contributes no number. The finite-temperature occupations
above were chosen for this near-degenerate system instead.
