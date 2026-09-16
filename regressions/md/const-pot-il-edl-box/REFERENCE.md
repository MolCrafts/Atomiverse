# const-pot-il-edl-box

oracle: metalwalls
oracle_version: MetalWalls release 20.05, git **c8ad175d** (2026-06-18)
oracle_kind: real binary empty capacitor (re-pinned 2026-07-29); with-IL
absolute Q not expressible under ATV truncated image-sum elastance

## Geometry

Honeycomb C walls (112 sites each) at z = 12 / 28 Å, Ψ = ∓1 V,
η = 1.805132 Å⁻¹; 12+12 CG IL ions; box ≈ 17.04 × 17.22 × 40 Å.

Twin: `tools/metalwalls_oracle/inputs/metalwalls/const-pot-il-edl-box/mw_twin/`.

## Pins (MetalWalls printed)

| case | Q_left [e] | Q_right [e] |
|------|------------|-------------|
| empty | **−2.076457628129008×10⁻¹** | **+2.076457628129015×10⁻¹** |
| with-IL | −4.818386149307364 | +4.818386149307384 |

ATV empty at `xy_image_range=80` matches MW to ~0.5 %. Absolute with-IL Q
under image-sum (~1.36 e) does **not** approach MW Ewald (~4.82 e) —
gate screening order `|Q|_IL > |Q|_empty` + polarity + neutrality instead.

## Superseded

Wave-5 `pin_il_edl.py` numpy BO at `xy_image_range=0` was a self-golden of
ATV's own min-image elastance (forbidden by pin-provenance rule).

## Perf

`perf_exempt` — product box; wall ratio deferred.
