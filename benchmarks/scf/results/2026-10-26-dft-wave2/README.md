# Molecular DFT follow-up acceptance, 2026-10-26

**67/67 scientific workloads passed, 134/134 required CPU/reference cells**
in Slurm job **4476975**. This extends the preceding
[35-workload record](../2026-10-26-dft/README.md), retaining its functionals
and adding standard CAM-B3LYP, PBEsol, revPBE and RPBE. All runs use the public
RKS/UKS Python methods and fresh live PySCF references, with one cold and one
fresh-state warm observation. The 67 Gaussian CUDA cells remain explicitly
unavailable; these results do not stand in for CUDA PW or GPW.

| Observable | Maximum error | Declared gate |
| --- | ---: | ---: |
| Total energy | 1.2861e-9 Ha | 1e-8 Ha |
| Every analytic force component | 1.2078e-9 Ha/Bohr | 1e-6 Ha/Bohr |

[Full matrix](matrix.md) · [HTML](matrix.html) ·
[Verification](verification.json) · [Original observations](measurements.json.gz).

CAM-B3LYP is now compared to standard libxc functional 433 through PySCF.
Its semilocal expression is .35 B88 + .46 short-range ITYH-B88 + .19 VWN5
plus .81 LYP, with exact exchange .19 K[1/r] + .46 K[erf(.33r)/r]. Both energy
and full analytic forces pass for RKS water/H₂ and UKS H₂⁺/O₂, including
spherical def2-SVP. The old model's failed standard-CAM records remain
preserved in the [molecular diagnostics](../../../../tools/diagnostics/molecular-dft/README.md).

The [separate CAM record](../../../../tools/diagnostics/molecular-dft/cam-standard-20261026/README.md)
includes 62 native units, six additional live molecular energy/force
comparisons and genuine libxc scalar/ITYH samples. Three extreme spin/density
points retain explicit B88/LYP/VWN numerical differences; passing full
molecular forces does not imply universal scalar parity at those boundaries.

The GGA-family additions cover RKS energies and asymmetric-water forces,
plus UKS H₂⁺/O₂ energies and forces in STO-3G. They do not qualify GPW or
CUDA variants, other bases, response kernels or periodic hybrids. Every
energy and force comparison uses explicit XC names, MatchPySCF/level-3
quadrature and full PySCF moving-grid force response.

The run used an AMD EPYC allocation with eight CPUs and one requested math
thread per engine. The native extension was copied into an isolated directory
before measurement; all native observations and both before/after hashes
agree on `867e7c257d325f31799c5e2fe95b341919587d0b849bed12b6a341b394e57874`.
The worktree was dirty and changed during measurement; full build attestation
remains false. This is numerical acceptance, without a performance ratio,
historical performance baseline or clean-source qualification. The 54 Python
grid/framework/parser contract checks also passed.

The exact submitted job, per-observation input/output/log archive, compressed
bundle and report projections are retained. Extract `artifacts.tar.gz` beside
a decompressed result bundle to resolve its `artifacts/...` paths. Reproduce
with a matching molpy/native build on a compute allocation:

```sh
python -m benchmarks run --method rks --method uks --tier acceptance \
  --threads 1 --repeats 1 --warmup 0 --timeout 300 --no-snapshot \
  --out /path/to/fresh/dft-acceptance
```

The timeout budgets each newly extended complete-call workload; the numerical
gates above are unchanged. The report can be regenerated without rerunning
science using `python -m benchmarks report --input results.json --out /path/to/report --no-snapshot`.
