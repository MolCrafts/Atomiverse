# ES CPU + CUDA opt — acceptance report (harness wave 5)

**Date:** 2026-08-10  
**Parent:** `dev` @ `721ac286` (post-merge of both wave-5 staff)  
**Staff:** CUDA only (`cpu_pending=0`). No push.

## Waves

| Wave | Landed | Notes |
| ---: | --- | --- |
| 1–4 | yes | See `docs/developer/es-cpu-cuda-queue.md` done |
| **5** | **yes (2/2)** | `cuda-pw-forces-pbe` + `cuda-gpw-ks-gga` |

## CPU track

**cpu_done this wave: 0.** No pin-safe wall lever on the fair twins.
worst_ot_ratio / worst fair ratio remains **MG MAIN 2.33×** (authority
`gpw-perf-baseline.md`; ATV 1.6 s / CP2K quiet 0.688 s). UKS O₂ 1.077× closed;
OT water OT ~1.20× / SCF body ≤1.0×. Blocked residual: `mg-main-structural`.

## CUDA track (this wave)

| Slug | Merge | Evidence |
| --- | --- | --- |
| `cuda-pw-forces-pbe` | `a92d1af0` | GH200 **1050675** rebased max\|ΔF\|=2.634e-13; post-merge **1050801** unit PASS + dual-gate max\|ΔF\|=**2.628e-13** eV/Å (tol 5e-3); E both −13.89672028 Ry; stream-share fix on force path |
| `cuda-gpw-ks-gga` | `721ac286` | GH200 **1050680** rebased dual-gate e_H/e_xc ~1e-15, max\|ΔV_eff\|=8.0e-16, max\|Δacc\|=3.9e-14; post-merge **1050825** (unit + reg) |

**Not landed:** full GPW product; collinear spin; metallic smear; MG KS chain;
RHF non-default stream. No classical MD CUDA. No new electronic methods.

## Loop tallies (es-cpu-cuda-opt cumulative)

| Metric | Value |
| --- | ---: |
| cpu_done (loop) | 10 (`…` + 2center setup/angular) |
| cuda_done (loop) | 18 (wave-5 +2: forces-pbe, ks-gga) |
| pending_remaining | **4** CUDA open |
| worst_ot_ratio / fair | **2.33** MG MAIN |

## Pending remaining (cuda_pending=4)

1. `cuda-pw-collinear-spin` (P1) — tip feat only; needs GH200  
2. `cuda-pw-metallic-smear` (P2)  
3. `cuda-gpw-ks-multigrid` (P2)  
4. `cuda-rhf-nondefault-stream` (P3)

Blocked: `cuda-gpw-full-product`, `mg-main-structural`, `global-xtensor-xsimd`.

## Clearup

- Done SPECs deleted (`cuda-pw-forces-pbe`, `cuda-gpw-ks-gga`)
- Builds wiped on GPU jobs (`builds_wiped=true`); parent tree no `build*`
- Fully-merged worker branches may be deleted in clearup; no force-merge of
  foreign MD tips

## Next staff (wave 6)

Prefer `cuda-pw-collinear-spin` on `wf/escuda-cuda-pw-collinear-spin` off
`721ac286`+ with **GH200 green required** before merge. Optional second:
`cuda-gpw-ks-multigrid` or `cuda-pw-metallic-smear` on a disjoint surface.
