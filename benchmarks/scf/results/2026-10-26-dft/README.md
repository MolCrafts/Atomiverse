# Molecular DFT acceptance, 2026-10-26

The public RKS/UKS benchmark passed **35 scientific workloads, 70/70 required
CPU/reference cells**, in Slurm job **4475712**. Every workload has a cold and
a fresh-state warm observation, with live PySCF rather than archived goldens.
The 35 unavailable CUDA cells remain explicitly unavailable: Gaussian CUDA KS
has no implementation. CUDA PW uses its own representation and
[separate GH200 evidence](../../../../tools/diagnostics/cuda-pw-public/README.md).

| Quantity | Maximum error | Unchanged acceptance gate |
| --- | ---: | ---: |
| Total energy | 1.095e-9 Ha | 1e-8 Ha |
| Analytic force component | 1.200e-9 Ha/Bohr | 1e-6 Ha/Bohr |

[Full matrix](matrix.md) · [HTML](matrix.html) ·
[Verification and scope](verification.json) ·
[Immutable observations](measurements.json.gz).

The 21 rows cover LSDA/PBE/BLYP/B3LYP/PBE0 RKS energies, PBE/PBE0 RKS forces,
and LSDA/PBE/PBE0 UKS energies and forces. PBE/PBE0 also use spherical
def2-SVP. Systems include water, stretched H₂, fully polarized H₂⁺ and triplet
O₂. The asymmetric-water def2-SVP force rows require all nine components to
be nonzero. XC recipes and integration controls are explicit; PySCF forces
include moving-grid response. Both solvers must converge, and forces are
compared component by component.

This matrix exercises the spherical XC-gradient repair and the public grid
control. Separate [molecular diagnostics](../../../../tools/diagnostics/molecular-dft/README.md)
record 28 affected native units, another 24 live energy/full-force comparisons,
independent libcint ERI/derivative pins and the long-range exchange repair.
**Standard CAM-B3LYP remains a model mismatch** and is excluded from this
passing matrix; all four failed standard-CAM comparisons are retained there.
Native-energy finite differences do not replace those failed external checks.

The allocation supplied eight CPUs; each engine used one requested math
thread. The campaign uses one cold and one warm call, with no warmup, and is
acceptance evidence rather than a performance campaign. Its dirty source tree
changed during the run and `build_verified=false`. The selected production,
binding, benchmark and comparison sources still match the measurement
manifest, and every native observation uses the same SHA256 as the final
extension:
`a0854aec15a0232eb1e979b117b643788fe432142135e7f56eebe984502a5c77`.
These checks do not establish a clean-source build attestation, historical
performance baseline or device-residency qualification.

The earlier attempt, job 4475288, was cancelled after the molpy Frame capsule
ABI guard rejected its separately built extension. No guard was bypassed and
no acceptance is claimed for that attempt. Its partial observations and logs
are preserved in `initial-abi-failure.json.gz` and
`initial-abi-failure-artifacts.tar.gz`. The successful run uses the matching
molpy/native build and retains the exact Slurm-submitted script. The 51 Python
grid/framework contract checks also passed.

Reproduce on a CPU compute allocation with a matching molpy/Atomiverse build:

```sh
python -m benchmarks run --method rks --method uks --tier acceptance \
  --threads 1 --repeats 1 --warmup 0 --timeout 120 --no-snapshot \
  --out /path/to/fresh/dft-acceptance
```

The result and payload projections, CSV and original per-observation requests,
outputs and logs are retained. Extract `artifacts.tar.gz` beside a decompressed
result bundle to resolve its `artifacts/...` links. Regenerate the report with
`python -m benchmarks report --input results.json --out /path/to/report --no-snapshot`.
