# CPU PW XC and response validation

These files record native validation, not external oracle pins or replacement
assertion constants. The external spectra, their qualification limits, and the
unchanged acceptance bounds remain in `regressions/response/dfpt/REFERENCE.md`.

The accepted implementation uses private 3/2-padded GGA XC quadrature with a
weighted-adjoint potential projection, and one CPU Pulay mixer for restricted
and joint alpha/beta densities. Spin residuals retain both unmixed channels.
No acoustic sum-rule projection is applied to pass the row-sum check.

Reproduce with a Release CPU build enabling HF, PW, GPW, VIB, tests and
regressions, then run:

```sh
ctest --test-dir build -LE regression -j1 --output-on-failure
ctest --test-dir build -R '^(response/dfpt|pw/rks|pw/collinear-spin)$' -j1 -V --output-on-failure
```

The registered response case declares one OMP/OpenBLAS thread. The raw output
includes every section, public displaced-supercell SCF convergence, same-q
spin identity, energy/free-energy curvature and whole-section times. The JSON
records source hashes and selected measurements for traceability; no test
reads it. Original numerical and 60-second bounds are retained. Historical
CP2K timings in the reference are not refreshed fair-performance comparisons.

`native-force-validation.txt` records the original single-H symmetry gate,
including standalone and nested method calls. Fixed-occupation SCF skips the
unused orbital solve of an exactly empty spin channel; its density is exactly
zero by the occupation definition. This prevents unconverged empty-channel
orbitals from replacing an occupied-channel initial guess. Force outputs are
not projected, and the original tol=1e-4, max_iter=8 and 1e-9 eV/Å gate remain.

`native-o2-validation.txt` records the current O₂ energy/magnetization and GGA
activation sections. The physical density FFT stays 36³; the private GGA XC
quadrature is 54³. The external QE energy and the original 1e-3 Ha acceptance
bound are unchanged. These runs are correctness/runtime checks, not fresh
matched-engine performance ratios.
