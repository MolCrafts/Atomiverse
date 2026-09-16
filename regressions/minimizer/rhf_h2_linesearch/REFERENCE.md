# Reference — H2 geometry optimization with a line-searched nested RHF

L-BFGS with the **default StrongWolfe** line search, over an RHF/STO-3G force
source **nested at `minimizer::Phase::force`**. Two things are pinned at once:
the relaxed geometry (literature) and the number of iterations it took (the
witness that every line-search trial carried a real SCF energy and slope).

## Literature value

**NIST Computational Chemistry Comparison and Benchmark Database (CCCBDB)**,
NIST Standard Reference Database 101, <https://cccbdb.nist.gov> — *Calculated
bond lengths*, hydrogen H₂ (¹Σg⁺), HF/STO-3G:

| Quantity | Literature | This case |
|---|---|---|
| r_e(H–H), HF/STO-3G | **0.712 Å** (0.7122 Å at table precision) | 0.712223 Å |
| E_total, HF/STO-3G at r_e | −1.11751 Hartree | −30.408884 eV = −1.117506 Ha |

The same STO-3G equilibrium bond length is the one originally reported with the
basis: W. J. Hehre, R. F. Stewart, J. A. Pople, *Self-Consistent Molecular-Orbital
Methods. I. Use of Gaussian Expansions of Slater-Type Atomic Orbitals*,
J. Chem. Phys. **51**, 2657 (1969), DOI 10.1063/1.1672392 — r(H–H) = 0.712 Å.

Only the bond length is asserted; the total energy is printed and quoted here as
corroboration (it is the *same* SCF that is pinned by
`regressions/scf/h2o_rhf_sto3g` and friends, so re-asserting it here would add
no independent information).

### Tolerance: 5 × 10⁻³ Å

Physical, not statistical. Three independent arguments, all coarser than the
measured deviation of 2.3 × 10⁻⁵ Å:

1. CCCBDB quotes the geometry to 10⁻³ Å; the tolerance is 5 × that resolution.
2. The convergence criterion sets the geometric residual: the measured H₂
   stiffness near the minimum is k = |f| / |Δr| ≈ 0.016441 eV/Å ÷ 3.05 × 10⁻⁴ Å
   ≈ 54 eV/Å² (from the printed trace, step 4 → final). With `ForceConverged`
   at ftol = 0.01 eV/Å the geometry can be at most |Δr| ≈ 2 × 10⁻⁴ Å off the
   stationary point — 25 × inside the tolerance. The criterion, not the
   tolerance, is what decides the answer.
3. The tolerance is 40 × smaller than the STO-3G basis error itself
   (0.7122 Å vs the experimental 0.7414 Å), so it cannot be satisfied by
   anything but the right minimum on the right surface.

## What this case pins about the *engine*

`Runtime::run_slot_` — the replay path behind `run_stage`, hence behind
`Minimize`'s `reeval_forces` line-search callback — dispatches **nested driver
nodes**. `cpu::scf::RHF` is a `Driver<scf::Phase>` node, so before that fix each
StrongWolfe trial replayed `ZeroForces → ZeroEnergy → (nothing) → Reduce` and
read back E = 0, f = 0. This is the first geometry optimization in the tree
whose line search provably re-evaluates a QM force source at trial geometries.

Unit-level cover for the same behaviour (stubs, no SCF):
`tests/cpu/driver/test_driver_nesting.cpp` (`RunStageReplaysNestedDriver`) and
`tests/cpu/minimize/test_line_search_nested_node.cpp`.

## Iteration bound: measured 5, frozen at 7

`iterations <= 7`. Measured **5**, frozen at 5 + 2 headroom (budget 30, so a
regression prints the real count rather than silently truncating).

Measured trace (single-threaded; identical at `OMP_NUM_THREADS=8`):

| step | E [eV] | f_max [eV/Å] | accepted max_disp [Å] |
|---|---|---|---|
| 1 | −29.71249472 | 6.2689e+00 | 0 (initial state) |
| 2 | −30.40461616 | 7.0669e−01 | 1.000000e−01 |
| 3 | −30.40711713 | 4.3649e−01 | 1.013092e−02 |
| 4 | −30.40888184 | 1.6441e−02 | 3.868214e−03 |
| 5 | −30.40888427 | 4.0276e−04 | 1.513987e−04 |

### Why 7 is only reachable with a working line search

The pre-fix binary cannot be rebuilt from this branch (the dispatch is
committed), so the bound is justified from the measured post-fix count plus two
pieces of recorded evidence about the degenerate search it replaced.

**(a) Recorded unit evidence (U5, 2026-07-26).** In the RED state of
`tests/cpu/minimize/test_line_search_nested_node.cpp` the nested node's call
counter equalled the iteration count exactly — *no* trial ever re-entered the
force source — and every accepted step's `max_disp` equalled the initial trial
step exactly: the search accepted α_init without ever testing it.

**(b) Offline replay of the pre-fix trial data through the public search.**
`strong_wolfe` (`cpu/minimize/line_search.h`) is header-only and public, so the
pre-fix inputs can be replayed without the pre-fix binary. Fed φ(α) ≡ 0,
φ′(α) ≡ 0 (what `ZeroEnergy`/`ZeroForces` leave behind) with *this case's*
measured iteration-0 state — φ(0) = −29.712495 eV, φ′(0) = −2·(6.268895)²,
α_max = 0.2 / 6.268895 — it exhausts all 20 trials inside `zoom` and returns
**α = 0, ok = false**:

```
SCF-like (phi0=-29.712495 eV < 0), zeroed trials: alpha=0.000000e+00 ok=0 n_eval=20
harmonic-like (phi0=3.500000 eV > 0), zeroed trials: alpha=1.595177e-02 ok=1 n_eval=1
SCF-like, REAL trials (quadratic):                  alpha=1.595177e-02 ok=1 n_eval=1
```

Zeroed trials look like a catastrophic energy *increase* whenever the true
energy is negative (every SCF), so the bracket collapses to α = 0 and **the
geometry never moves**: pre-fix, this case would have spent its whole 30-step
budget at r = 0.90 Å and never converged — failing the r_e pin and the
`converged()` pin outright. The middle line is the same probe with a *positive*
φ(0) (the U5 harmonic stub), which reproduces (a): the first trial passes both
Wolfe tests vacuously and α_init is accepted blindly. The last line confirms
that real trial data accepts α = 1.595177e−02, i.e. exactly the 0.1 Å first step
in the measured trace above — the accepted step is the strong-Wolfe first trial,
not a fixed cap.

**(c) Degradation guard, measured.** The same case with
`LineSearchMode::Fixed` (same `alpha0 = 0.01`) needs **> 200** iterations and is
still at r = 0.7194 Å, unconverged, at step 200. So the frozen bound of 7 sits
between a genuine line search (5) and any step-scaled walk, and it is not a
number a degraded search can reach.

## Runtime / perf note (`perf_exempt`)

There is **no external wall-clock twin** for this case: the iteration count is
an *algorithmic-cost* pin (number of geometry steps), not a timing pin, and
nothing here is asserted against a clock. Measured wall time is 0.03 s
single-threaded (budget: ≤ 60 s).

**Release note:** the StrongWolfe line search now costs **one SCF per trial**
(it used to be free because it was wrong). Users who cannot afford the trials
select `LineSearchMode::Fixed` — `lbfgs.line_search(LineSearchMode::Fixed)` —
which takes the step without evaluating any trial geometry.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_REGRESSIONS=ON \
      -DATV_BUILD_HF=ON -DATV_BUILD_MINIMIZER=ON -DATV_BUILD_CUDA=OFF
cmake --build build --target reg_rhf_h2_linesearch
OMP_NUM_THREADS=1 ./build/regressions/reg_rhf_h2_linesearch
```

Provenance of the measured numbers: this tree at commit `fe50f5c4` (branch
`dev`), GCC 14.3 (`buildenv-gcc/2026.03-mpich`), Release, x86_64 login node,
2026-07-27.
