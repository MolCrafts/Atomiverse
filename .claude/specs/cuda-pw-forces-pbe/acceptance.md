---
slug: cuda-pw-forces-pbe
acceptance_met: true
---

# Acceptance checklist

- [x] Spec lever named (DeviceDensity `method.xc` force rebuild; reg dual-gate)
- [x] `reg_cuda-pw-forces-pbe` lands with `XCType::PBE` Si dimer stretch
- [x] CMake target `run_reg_cuda-pw-forces-pbe`
- [x] Unit: public API accepts PBE (no throw on empty-frame gate before SCF)
- [x] GH200 job **1050253**: units PASS + dual-gate PASS max|ΔF|=2.615e-13
- [x] `builds_wiped=true` on job exit
- [x] Commits on `wf/cuda-pw-forces-pbe` only (no push)

Also fixed force-path stream race (`den.stream()` shared with band solver).
