---
slug: cuda-pw-forces-pbe
acceptance_met: false
---

# Acceptance checklist

- [ ] Spec lever named (DeviceDensity `method.xc` force rebuild; reg dual-gate)
- [ ] `reg_cuda-pw-forces-pbe` lands with `XCType::PBE` Si dimer stretch
- [ ] CMake target `run_reg_cuda-pw-forces-pbe`
- [ ] Unit: public API accepts PBE (no throw on empty-frame gate before SCF)
- [ ] GH200 job log: units PASS + dual-gate PASS with stated tols
- [ ] `builds_wiped=true` on job exit
- [ ] Commit on `wf/cuda-pw-forces-pbe` only (no push)

Set `acceptance_met: true` only after GH200 green evidence is in REFERENCE.md.
