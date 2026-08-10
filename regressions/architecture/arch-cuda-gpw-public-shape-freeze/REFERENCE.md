# arch-cuda-gpw-public-shape-freeze

## Intent

Architecture freeze: full CUDA GPW method product **must equal** CPU
`GpwDft` public shape (members + `build` / `run` / `last_result` +
forces/stress free-function family). Stage footholds stay internal.
CPU OT free-loop residual does not block the freeze.

## Oracle

`oracle: harness` — public-shape assertions only (no energy pin, no SCF).
Contract text: `.claude/notes/cuda-gpw-public-shape.md`.

## Acceptance

- `GpwDft` is `cpu::Driver<gpw::Phase>` with five SCF slots
- Public surface markers present: `build`, `run`, `last_result`, `is_uks`
- No multi-iter SCF in this binary
