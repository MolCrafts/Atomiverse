# pin notes — `pim-induced-dipole-bulk` (re-pinned against the real MetalWalls binary)

Date: 2026-07-29 · Oracle: **MetalWalls git `c8ad175d`** (release branch,
2026-06-18), built at
`/nobackup/.../work/opt/metalwalls/mw`, run `mpirun -n 1`.
Verdict: **repinned**. No Atomiverse source was changed; no tolerance widened.

Full numbers, box ladder, unit conversions and the discrepancy analysis live in
`regressions/md/pim-induced-dipole-bulk/REFERENCE.md`. This file records how to
re-run it and what the traps were.

## How to re-run

```bash
bash -lc 'module load buildenv-gcc/2026.03-mpich
          export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
          cd <case-dir>
          mpirun -n 1 /nobackup/.../work/opt/metalwalls/mw > run.stdout 2>&1'
```

- `inputs/metalwalls/pim-induced-dipole-bulk/pol/` — L = 50 Å twin, α > 0.
- `inputs/metalwalls/pim-induced-dipole-bulk/nopol/` — same geometry, α = 0.
- `inputs/metalwalls/pim-induced-dipole-bulk/box_ladder.py` — writes the
  L = 50…3200 Å ladder into `/tmp/mw_oracle_pim-induced-dipole-bulk/scan/`.
- `.../extrapolate.py` — collects the ladder, prints the Richardson table.
- `.../closed_form_check.py` — independent closed-form two-site Applequist SCC
  (damped and undamped) used to cross-check both sides.

`E_pol` and the polarization force are **differences** of the α > 0 and α = 0
runs at identical geometry/box/Ewald settings (MetalWalls prints no separate
polarization channel; its variational PIM functional collapses to
`U_qq − ½Σμ·E^perm` at the minimum).

## Traps hit (record these — they cost the most time)

1. **`damping` block is mandatory under PIM.** `system%tt` is allocated only by
   `read_tt_parameters` (`src/configuration.F90`), but `coulomb_sr_pim` reads
   `tt%order/b/c` unconditionally when `pimrun` is true. Omit the block and
   MetalWalls SIGSEGVs in `qmelt2mumelt_electricfield` *after* printing its
   whole setup banner, so it looks like a physics failure and is not.
   Workaround: declare the block with intensity `c = 0.0`, which is exactly
   undamped. (Upstream robustness bug at c8ad175d; not ours to fix.)
2. **MetalWalls has no Thole damping.** `grep -ri thole` over the whole tree:
   zero hits. Its only short-range multipole screening is Tang–Toennies on the
   charge→dipole field, with a per-ordered-pair intensity `c` that
   `PimInducedDipole::tt_b` does not have. Any `thole_a > 0` pin is therefore
   *not* a MetalWalls number, whatever a previous REFERENCE.md claimed.
3. **MetalWalls is Ewald-periodic, `PimInducedDipole` is min-image/cutoff.**
   At L = 50 Å the two differ by 8e-4 (μ) to 2e-3 (E_pol) — pure
   neutralising-background lattice-sum effect, exactly ∝ 1/L³. Comparing at a
   single box size would have read as a real disagreement. The ladder +
   Richardson extrapolation is what makes the comparison meaningful.
4. **Everything left after that is `units::coulomb`.** ATV ships
   `14.399645 eV·Å/e²`; exact is `14.399645478456`. The 3.3227e-08 relative
   deficit reproduces the entire residual in E and F, and is absent from μ
   (the SCC runs in Gaussian units where the constant never appears).

## Fair wall (perf gate)

The regression's two-site case is below timer resolution on both sides, so the
wall twin is a **512-ion disordered rocksalt PIM single point** built with the
*same species as the regression* (q = ±1, α = 1.0 / 1.5 Å³, 4×4×4 NaCl-type
lattice, a = 5.64 Å, every ion displaced uniformly in ±0.35 Å, seed 20260729,
L = 22.56 Å). Generators, committed next to the L = 50 Å twin:
`inputs/metalwalls/pim-induced-dipole-bulk/perf_twin_512.py` (writes both
MetalWalls cases plus `geom.txt`) and `perf_twin_512_atv.cpp` (the Atomiverse
consumer; compile with the link line of any `reg_*` target).

Method — both sides isolate the induced-dipole solve, single rank, single
thread (`mpirun -n 1`, `OMP_NUM_THREADS=1`), best of 7 cold runs:

- MetalWalls: its own internal timer, `run.out` →
  *Additional degrees of freedom computation* → **`Melt dipoles computation`**.
  Nothing else is counted: no MPI start-up, no Ewald setup, no file I/O.
- Atomiverse: `std::chrono` around `cpu::Evaluate::run` with only
  `PimInducedDipole` bound — permanent field + SCC + forces + virial.

| | wall (best of 7) | iterations | E_pol |
| --- | --- | --- | --- |
| MetalWalls c8ad175d `Melt dipoles computation` | **8.0124e-02 s** | 17 (preconditioned CG, Ewald) | −0.723372 Ha |
| Atomiverse `PimInducedDipole` launch | **6.17e-02 s** | 35 (Jacobi, cutoff O(N²)) | −0.745275 Ha |

**ratio_atv_over_mw = 0.77** (gate: ≤ 10 — passes with a wide margin).

**Caveat that must be quoted with the ratio:** the two are not doing identical
work — the E_pol column differs by 3.0 %. MetalWalls solves the *Ewald* PIM
problem (real + reciprocal, preconditioned CG); Atomiverse solves the *cutoff
min-image* PIM problem (11 Å cutoff, O(N²) all-pairs Jacobi, no reciprocal
space). Atomiverse is doing strictly *less* physics per iteration, so 0.77 is a
lower bound on the cost of actual parity, not an equal-accuracy benchmark. It
is reported anyway because "no binary" is no longer a reason to skip the gate.
The two facts worth carrying forward: Atomiverse needs **2×** the iterations
(Jacobi vs preconditioned CG — a real algorithmic gap, `PimInducedDipole` has
no preconditioner and no CG), and it has no reciprocal-space dipole path at
all, so a like-for-like Ewald PIM would land somewhere above 1.0.
