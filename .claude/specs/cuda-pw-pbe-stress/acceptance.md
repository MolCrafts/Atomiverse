# acceptance — cuda-pw-pbe-stress

- [x] Unit: `test_cuda_pw_cell_stress` green on GH200 (empty/open-shell + H₂ LSDA + H₂ PBE twins)
- [x] Regression `reg_cuda-pw-stress-pbe --device=both`:
  - CPU and CUDA σ finite + cubic symmetry (soft 2.5e-2 eV/Å³)
  - max\|Δσ\| = 5.585e-12 eV/Å³ < 5e-2
  - mean\|diag_cpu\| = 0.273 > 1e-3 (non-vacuous)
- [x] GPU log evidence: **GH200 job 1122143** (wave-1); re-verify after
      rebase onto `dev@d701a1cd`: **GH200 job 1122705**; re-verify after
      rebase onto `dev@4913b33b` (Assign wave 3): **GH200 job 1123959**
      (same max|Δσ|=5.585e-12 eV/Å³, E=-13.90403280 Ry, BUILDS_WIPED_OK)
- [x] perf_exempt (correctness foothold: 1 SCF + 12 NSCF; small cell)
- [x] build* wiped after gate (`BUILDS_WIPED_OK`)

`acceptance_met=true`
