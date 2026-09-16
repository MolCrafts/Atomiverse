# Reference — MPI PW k-mesh round-robin scaling gate (4 ranks)

## Method

Spin-unpolarized plane-wave Kohn–Sham LDA on bulk silicon — the **identical
twin** to `regressions/pw/pw-lda-si` (same geometry, pseudopotential, cutoff,
k-mesh, tolerance) — executed at **np = 1** and **np = 4** MPI ranks. Inside
`cpu::pw::PW::run()` k-points are distributed round-robin over
`MPI_COMM_WORLD`; density and band-energy sums are Allreduced every SCF
iteration, and the Davidson initial guess is seeded per subspace index (not
per rank), so the SCF trajectory is rank-count invariant by construction.
This regression is the shipped **industrial gate** for that claim plus the
scaling wall ratio (queue metric).

Public path: `cpu::pw::PW` (members) → `build(frame)` / `run(frame)`;
distribution observables `PWResult::n_kpts` / `PWResult::n_kpts_local`.

The gate binary re-launches itself through the build-configured `mpiexec`
(MPICH hydra from `buildenv-gcc/2026.03-mpich`) — **no MPI launcher is
invoked at ctest time**; the case runs only under the explicit
`run_reg_mpi-pw-scaling-4rank` target like every regression.

## Pseudopotential (same in-tree file as pw-lda-si)

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013); M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |

## Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| \(E_\mathrm{cut}\) | 15 Ry |
| k-mesh | \(2\times2\times2\) Monkhorst–Pack (`k_mesh = 2`, 8 k-points) |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 50 |
| Ranks × OMP threads | 1×4 and 4×4 (equal threads per rank in both runs) |

## Pinned acceptance

1. **External correctness pin (both rank counts):** PseudoDojo ships an
   ABINIT deltafactor EOS for this exact ONCV LDA Si pseudopotential
   (`Si.djrepo`, deltafactor key `"8.0"`, `num_sites = 2`):
   \(E_\mathrm{min} = -230.158202\) eV/cell \(= -16.916316\) Ry/cell.
   The under-converged twin must sit in the variational incomplete-basis
   window \([-16.916316,\, -16.916316 + 1.50]\) Ry — the same cited band as
   `pw-lda-si` (lower edge external; headroom covers ecut 15 / k = 2).
2. **Rank-count invariance:** \(|E(np{=}1) - E(np{=}4)| \le 10^{-8}\) Ry and
   identical SCF iteration counts (deterministic seeding ⇒ identical
   trajectory; observed agreement is exact to output precision).
3. **Real 4-way split:** at np = 4 the reduced per-rank owned-k counts must
   be exactly \([2, 2]\) (8 k-points / 4 ranks); at np = 1, \([8, 8]\).
4. **Scaling wall ratio (queue metric):**
   \(T_{\mathrm{run}}(np{=}1) / T_{\mathrm{run}}(np{=}4) \ge 1.2\) at equal
   OMP threads per rank. The floor is deliberately weak (shared login node;
   grid/density work is replicated on every rank — only the per-k Davidson
   parallelizes at this problem size).

## Measured on this host (2026-07-25, Release, MPICH, 4 OMP threads/rank)

| Quantity | np = 1 | np = 4 |
| -------- | ------ | ------ |
| \(E\) [Ry] | −15.704816618 | −15.704816618 |
| SCF iterations | 6 | 6 |
| owned k per rank | 8 | 2 |
| \(T_\mathrm{run}\) [s] | 3.801 | 1.495 |

\(|\Delta E| = 0.0\) Ry (bit-identical); speedup **2.54×** (floor 1.2);
total regression wall ≈ 12 s — inside the 60 s budget.

## CP2K note (absolute energy + wall)

CP2K Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH** (different PP and
discretization): absolute total energies are **not** comparable to ONCV
plane-wave, and the fair CP2K science comparison is EOS-level (see
`regressions/pw/pw-lda-si/REFERENCE.md`). An offline wall-twin input
skeleton lives at `tools/cp2k_oracle/inputs/si_lda_pw_note.inp`.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary or module on the build host (`module avail cp2k` / `which cp2k{,.psmp}` empty, checked 2026-07-25); fair wall twin cannot run offline this shift |
| Internal scaling gate | speedup 2.54× at 4 ranks (floor 1.2, enforced by the regression) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness + internal scaling gate only) |

When a CP2K binary is available: time the same geometry with the oracle
input at 4 MPI ranks, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) here,
require \(r \le 10\).
