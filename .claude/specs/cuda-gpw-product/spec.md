---
slug: cuda-gpw-product
status: in-progress
created: 2026-08-14
domain: cuda / gpw
queue: claim-completion
deferred: true
depends_on: []
unblocks: [cuda-gpw-product]
grilled: false
---

# Spec: `cuda-gpw-product` — finish the frozen CUDA GPW product shape

**Deferred:** GH200 + CUDA ES open gate. **This spec adds no new design** —
the public shape is frozen to CPU `GpwDft` on `dev`
(`.claude/notes/cuda-gpw-public-shape.md`, merge `0147fe02`) and the work
items live on the es-cpu-cuda queue. This spec exists to carry the claim
bar and the close-out condition; the shape note remains the authority.

## Summary

CUDA GPW is ◐: Γ RKS DIIS PADE/PBE + Γ UKS PADE energy + multi-k foothold,
with host residuals (χ^k, Löwdin) and hybrid force/stress legs. Finish the
staffable product under the frozen contract: water σ, multi-k F/σ, UKS
F/σ, MG production — and replace CPU-twin-only anchors with independent
ones (CP2K twin decks per the strict loop law) so ◐ can become ✓.

## Design

Authority documents (do not fork them into this spec):

- Shape contract: `.claude/notes/cuda-gpw-public-shape.md`
- Landed inventory: `.claude/notes/cuda-gpw-product-inventory.md`
- Queue + wave order: es-cpu-cuda queue (`.claude/notes/es-optimization-order.md`)
- Anchor law: `.claude/notes/oracle-pin-provenance.md` (strict CP2K)

Claim bar addition (from `docs/developer/claim-completion-queue.md`): every
CUDA GPW case that today anchors only on the ATV CPU leg gains a CP2K pin
(same settings class) before its row is marked ✓ — cross-backend
equivalence alone stays ◐.

## Tasks

- [ ] Work the es-cpu-cuda queue items under the frozen shape (water σ →
      multi-k F/σ → UKS F/σ → MG production, or the queue's current order)
- [ ] Independent-anchor sweep: add CP2K pins to the ~17 CPU-twin-only CUDA
      cases (`regressions/gpw/cuda-gpw-*`, `regressions/pw/cuda-*`)
- [ ] Dual-gate green per item on GH200; inventory + roadmap ◐/✓ updates
- [ ] Close when the queue's product matrix for the frozen shape is done
      and no CUDA GPW ✓ rests on a self-anchor

## Testing

Per the ES dual-gate law: CUDA↔CPU stated tolerances + external CP2K pin
per quantity with a twin. Perf statements only on large systems.

## Out of scope

- Any public-shape change (frozen)
- Hybrids/meta-GGA/GAPW (CPU engine gaps first)
- New slugs for follow-on capabilities (GapCensus owns discovery)
