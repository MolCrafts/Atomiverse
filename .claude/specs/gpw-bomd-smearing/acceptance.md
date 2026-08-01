# Acceptance — gpw-bomd-smearing

| Gate | Criterion |
| ---- | --------- |
| Product | Multi-k BOMD under FD/MP smear advances ions with free-energy SCF and fractional \(P_k\) warm start |
| Regression | `reg_gpw-bomd-smearing` green ≤ 60 s with dual-gate vs stated pins |
| Oracle | Real CP2K 2026.1 AIMD+SMEAR+k twin in REFERENCE.md (matched GPW class) **or** honest `oracle: literature` only if CP2K cannot express the propagator |
| Perf | Fair wall on matched twin job \(\le 10\times\) (no `perf_exempt` for missing binary) |
| API | No public `*Config` / factories; Frame for MD state |

`acceptance_met=true` only when all gates pass.
