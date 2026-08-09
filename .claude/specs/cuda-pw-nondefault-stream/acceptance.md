# Acceptance — cuda-pw-nondefault-stream

- [x] Spec names lever (file/function/metric)
- [x] Shared non-default stream: density owns, band solver borrows
- [x] All density/band kernels + library handles on that stream
- [x] Unit: `stream() != nullptr`; density stage parity tols unchanged
- [x] GH200 log **1048455**: ctest `test_cuda_pw_*` 8/8 PASS; diag MAX_EIG_DIFF 5.274e-15 Ha; both regs |ΔE|=1.243e-14 Ry
- [x] Builds wiped after gate
