# GAFF API alignment: isolated CPU benchmark

Slurm job **4473906**, FAT partition, node **n1**, **AMD EPYC 9745 128-Core Processor**, physical CPU affinity **[2]**, one thread. The allocated core was **1.02% busy** in the one-second preflight. Both routes and all subprocesses inherited the same CPU binding and OpenMP/BLAS/Rayon limit.

**All 576 observations completed. 46 of 48 molecule/model combinations passed every required numerical check.** The two GAFF2 DMSO comparisons retain the proven numbering-dependent tleap proper-torsion anomaly and have **no speedup**. The resulting 94/96 participant-cell verdict and job exit code 1 are expected numerical disagreement, not an execution failure.

The campaign covers 12 fixed canonical molecules (7–50 atoms), GAFF/GAFF2 crossed independently with BCC/ABCG2, one cold observation plus five fresh warm observations per participant, and one untimed fresh-state warmup before each warm sample. Raw inputs, logs, outputs, every timing and parameter record are retained. No outlier or failed comparison was dropped.

## Full-route timing

Ratios below use median(reference)/median(native) for accepted cases only. They include graph/input conversion, AM1/BCC, typing/parameter assignment and common output materialization. The external route includes antechamber, parmchk2 and tleap startup/file costs. **These are full Python-versus-CLI route ratios, not isolated AM1 solver speedups.** Cold process timing is also retained in the results.

| Model | Accepted cases | Minimum ratio | Median ratio across cases | Maximum ratio |
| --- | ---: | ---: | ---: | ---: |
| `gaff.bcc` | 12/12 | 6.50× | 65.09× | 160.86× |
| `gaff.abcg2` | 12/12 | 6.83× | 63.23× | 138.85× |
| `gaff2.bcc` | 11/12 | 6.30× | 42.42× | 112.56× |
| `gaff2.abcg2` | 11/12 | 6.18× | 43.40× | 117.28× |

The range reflects different molecule sizes and chemistry. It is not a confidence interval. Within-case ranges and median absolute deviations (MAD) are available for all five samples. There is no historical before/after performance baseline.

GAFF2 + BCC representative cases:

| Molecule | Atoms | Native median ± MAD (ms) | Reference median ± MAD (ms) | Ratio |
| --- | ---: | ---: | ---: | ---: |
| acetamide | 9 | 7.055 ± 0.444 | 601.354 ± 59.214 | 85.24× |
| acetate | 7 | 5.658 ± 0.369 | 537.259 ± 108.488 | 94.95× |
| dmso | 10 | 6.828 ± 0.184 | 528.775 ± 47.761 | ineligible: torsion mismatch |
| ethanol | 9 | 5.091 ± 0.343 | 409.500 ± 83.589 | 80.43× |
| halogenated | 8 | 7.793 ± 0.270 | 463.165 ± 49.971 | 59.43× |
| hexadecane | 50 | 119.224 ± 0.189 | 751.566 ± 108.583 | 6.30× |
| methylammonium | 8 | 5.333 ± 0.191 | 600.240 ± 40.550 | 112.56× |
| octane | 26 | 15.402 ± 0.187 | 616.009 ± 163.381 | 40.00× |
| phenol | 13 | 15.286 ± 0.018 | 565.628 ± 79.283 | 37.00× |
| pyridine | 11 | 12.871 ± 0.214 | 473.175 ± 47.652 | 36.76× |
| thiophene | 9 | 15.558 ± 0.245 | 660.024 ± 108.151 | 42.42× |
| trimethylphosphate | 17 | 17.431 ± 0.080 | 598.522 ± 68.361 | 34.34× |

## Component diagnostics

Component medians are separate profiling scopes and need not sum to the full-route median. The native charge stage includes AM1, equivalencing and BCC; antechamber likewise includes SQM and its surrounding pipeline.

| GAFF2+BCC molecule | Native AM1/BCC (ms) | Native GAFF (ms) | antechamber (ms) | parmchk2 (ms) | tleap (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ethanol | 2.107 | 1.257 | 314.054 | 38.533 | 59.916 |
| hexadecane | 112.021 | 3.844 | 592.469 | 70.039 | 98.500 |
| trimethylphosphate | 13.314 | 2.236 | 412.293 | 69.135 | 68.406 |

## Numerical scope

GAFF atom-type labels match exactly. The maximum per-atom charge deviation is **0.00086944266 e**, below the predeclared 0.001 e limit. Bond parameters, atom masses, accepted improper parameters and 1–4 scales agree exactly. The maximum angle difference is **5.6108461e-05 degree**, attributable to Amber's frozen radian/degree conversion; accepted proper coefficients differ by at most **4.4444445e-10** at prmtop decimal precision. The maximum Lennard-Jones parameter difference is **3.825837e-09** in its declared sigma/epsilon units.

The [DMSO anomaly evidence](../2026-10-26-reference-anomaly/README.md) demonstrates that swapping chemically equivalent methyl carbon indices moves tleap's extra torsion terms to the other physical methyl. Native parameterization retains the symmetric exact GAFF2 zero-coefficient row. The comparison remains a failure; no parameter tolerance was relaxed to hide it.

This is a parameter-level comparison against independently assembled live AmberTools outputs. It does not establish independent force/energy-evaluator agreement, geometry optimization, dynamics, RESP, GPU support or arbitrary organic-chemistry coverage.

## Provenance and artifacts

ATV source hashes stayed unchanged during the campaign. The full sibling molpy tree was independently captured before and after and matched. All 288 native observations used the same source receipt, sibling tree and both actual loaded native binaries. [Provenance verification](provenance-verification.json) passed. The source tree is local/dirty and has no official build attestation; the roadmap remains provisional.

The CI snapshot envelope was explicitly disabled with `--no-snapshot` because installed molci lacks the framework's required 0.3 protocol. Scientific result JSON, checks, CSV, JUnit and matrix are complete.

- [HTML matrix](matrix.html), [Markdown matrix](matrix.md), [full authoritative results](results.json), [compact generated summary](summary.json).
- [Resource allocation](resource-allocation.json), [Slurm submission](submission.sbatch), [exact run and receipt-verification script](run-isolated.sh), [controller log](controller.log).
- [Sibling source receipt](molpy-source-receipt.json), [pre/post verification](provenance-verification.json), complete raw AmberTools jobs under `raw/` and framework observations under `artifacts/`.
- [Actual API example tleap output](example-tleap.log): valid MOL2/frcmod accepted with 0 errors, 0 warnings, 0 notes.
