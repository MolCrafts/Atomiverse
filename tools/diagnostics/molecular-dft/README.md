# Molecular DFT numerical qualification

Standard CAM-B3LYP was subsequently repaired and passed six live PySCF
energy/full-force comparisons. See the [standard CAM repair and current
qualification](cam-standard-20261026/README.md). The first-run failures below
remain historical evidence; the numerical artifacts were not overwritten.

`validate.py` compares freshly constructed public Python `RKS`/`UKS` methods
with live PySCF, checking total energy and every analytic force component.
Run it on a compute allocation with a freshly compiled HF-enabled extension,
NumPy, molpy, PySCF and threadpoolctl. The quadrature is explicit:
`grid="MatchPySCF"` and PySCF level 3, including the analytic grid response in
the oracle gradient. Both solvers must converge. Energy and force gates are
1e-8 Ha and 1e-6 Ha/Bohr, respectively; nonfinite or missing results fail.

```sh
PYTHONPATH="$PWD/python/src:$PWD/build-hf-auto:$PWD" \
  .claude/scratch/hf-auto/env/bin/python \
  tools/diagnostics/molecular-dft/validate.py \
  --out /path/to/fresh/dft-validation --threads 1
```

`--system`, `--basis` and `--xc` select subsets. Defaults cover PBE, PBE0,
BLYP and B3LYP. `--native-fd` adds full own-energy finite differences as a
separately labeled consistency check; they never replace the live oracle. The default matrix exercises
non-symmetric water (RKS), H2+ (UKS with an empty beta channel), and triplet O2
(UKS with both channels), with STO-3G and spherical def2-SVP. The supplied
geometries exercise nonzero Cartesian forces. The JSON retains inputs, full
force vectors, solver convergence, errors, gates, Python/native library
identities, source hashes, threadpool and allocation metadata. It is numerical
qualification rather than a performance comparison. `build_verified=false`
explicitly leaves formal source-to-binary build attestation open.

This diagnostic was added while repairing two independent force defects:

- Spherical-basis XC gradients now contract the Cartesian shells with
  `P_cart = C2S P_sph C2S^T`, including every Cartesian component. Previously
  they indexed the spherical density as if it were Cartesian and omitted the
  extra Cartesian d/f components. The defect affected semilocal and hybrid
  functionals and both spin treatments even when total energies were correct.
- The first repair of the then-named CAM-B3LYP implementation added the nuclear
  derivative of its long-range erf exchange
  integral, scaled by the additional long-range exchange fraction. The same
  analytic angular-momentum recurrence evaluates bare and attenuated ERI
  derivatives. The UKS Fock/energy path now carries the same long-range term,
  which it previously omitted. The native integral unit tests independently check finite
  differences, all centers, shell swaps, translation invariance, the zero
  range parameter convention and the large-range bare limit.

These checks qualify only the recorded method/basis/geometry combinations.
They do not establish periodic hybrid DFT, meta-GGA, a full B2PLYP MP2
correction, or CPU/CUDA parity.

The first-run `CAMB3LYP` functional reused the B3LYP semilocal mixture
instead of standard CAM's short-range B88 form. It was excluded from the
original default passing matrix. The archived run retained standard PySCF
CAM-B3LYP failures alongside native-energy/force consistency. A passing
consistency check did not qualify that earlier model as standard CAM-B3LYP.
The later repair above replaces the semilocal recipe; rerunning
`--xc CAMB3LYP` now checks the standard functional without a custom oracle.

`pin_erf.py` and `erf-pyscf-pin.json` retain the independent PySCF/libcint
four-center attenuated ERI and all three nuclear derivatives on its first
center. The `TestRysEriErf.OffCenterValueAndDerivativeVsPyscf` unit consumes
these numbers. It detects a missing transformation of the Rys roots that the
older coincident-center ssss energy pin could not expose: attenuating the
Coulomb parameter also requires mapping `u=t^2/(1-t^2)` to
`theta*u/(1+(1-theta)*u)`. Both the value and derivative kernels now apply this
map. The zero-range-parameter bare-operator convention is retained.

The 2026-10-26 CPU run (Slurm 4475911, n213, one math thread) passed all 24
default comparisons. Maximum energy and full-force-component errors were
1.083e-9 Ha and 1.348e-9 Ha/Bohr. [Complete observations](cpu-validation.json)
include all force vectors and both solver convergence flags;
[compressed native log](cpu-validation.log.gz) retains the solver output.
All 28 affected integral/gradient conversion unit tests passed:
[unit log](unit.log), [compile/link and binary/source provenance](unit-build.json).
The independent libcint pin records native library paths and SHA256 values,
as does the molecular oracle record. These are scoped numerical checks;
formal build attestation remains explicitly false.

The same run retained all four standard CAM-B3LYP failures (water and H2+,
both bases): total-energy differences span 0.0751–0.7439 Ha and maximum
force-component differences span 0.00108–0.00656 Ha/Bohr. All four full
native-energy finite-difference checks passed at the unchanged 1e-6 Ha/Bohr
consistency gate; their maximum residual was 5.101e-9 Ha/Bohr. This verifies
the repaired derivative of the native model, while its standard CAM
qualification stays failed. [Complete failed oracle comparisons and separate
consistency checks](cam-standard-mismatch.json),
[compressed native log](cam-standard-mismatch.log.gz).
