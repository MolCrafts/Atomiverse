# Acceptance — cuda-pw-nondefault-stream

- [x] Spec names lever (file/function/metric)
- [x] Shared non-default stream: density owns, band solver borrows
- [x] All density/band kernels + library handles on that stream
- [x] Unit: `stream() != nullptr`; density stage parity tols unchanged
- [x] GH200 **1049681** (`HEAD=7a8b3ece`, branch `wf/escuda-cuda-pw-nondefault-stream`):
  ctest 9/9 PASS (8× `test_cuda_pw_*` + `test_cuda_scf_rhf`);
  `diag_band_parity` MAX_EIG_DIFF **5.274e-15 Ha**;
  `reg_pw-cuda-lda-si` + `reg_cuda-pw-nondefault-stream` |ΔE|=**1.243e-14 Ry**;
  `GPU_JOB_OK` + `BUILDS_WIPED`
- [x] Builds wiped after gate (job + post-worker scan)
