# periodic-scs-mp2-kpoints

## Goal

Dedicated dual-gate for **multi-k RI-GPW SCS-MP2** on product
`GpwDft::mp2` — Grimme scales `mp2_os_scale=1.2` / `mp2_ss_scale=1/3` on the
momentum-conserving multi-k path (same members as Γ `periodic-scs-mp2-si`
and molecular RHF). Capability already product-shaped after
`periodic-mp2-kpoints` + scale members; this slug is **pin-only** vs a
matched CP2K multi-k `MP2_GPW` twin with `SCALE_S` / `SCALE_T`.

## System

- Diamond Si bct 4-atom cell (same as `periodic-mp2-kpoints` / `gpw-kpoints-si`)
- a = 5.43 Å; a_t = a/√2, c = a
- DZVP-MOLOPT-GTH / GTH-PBE-q4, PBE, cutoff 80 Ry
- MP 2×1×1, `k_time_reversal=false` (equal weights)
- `mp2_ri=true`, `mp2_n_virt=2` (e2e budget)

## Public API

```cpp
cpu::gpw::GpwDft m;
// ... multi-k PBE GPW members; build(); run(atoms);
m.mp2_ri = true;
m.mp2_n_virt = 2;
m.mp2_os_scale = 1.2;
m.mp2_ss_scale = 1.0 / 3.0;
const cpu::gpw::Mp2Result& r = m.mp2(atoms);
// r.e_corr_os / r.e_corr_ss unscaled; r.e_corr = 1.2*OS + (1/3)*SS
```

## Oracle

- Offline CP2K 2026.1 `cp2k.psmp` git:5e54ba2 **actually run** on
  `tools/cp2k_oracle/inputs/periodic-scs-mp2-kpoints.inp`.
- Discovery: multi-k SCF converges; `METHOD MP2_GPW` aborts under DFT
  `&KPOINTS` (`Method not implemented for k-points`).
- Pin path: **`oracle: literature`** (Grimme SCS 2003 + Del Ben RI-GPW 2012)
  with algebraic dual gates; never invent a CP2K SCS energy.
- Never launch from ctest.

## Acceptance

See `acceptance.md`.
