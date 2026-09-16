# Reference — Monte Carlo on an electronic structure: umbrella-biased H₂ bond-length variance

Metropolis MC at 300 K on H₂, where the trial energy is a **nested
`cpu::scf::RHF`/STO-3G driver node** plus a **harmonic umbrella on the H–H
distance CV**. The chain samples the bond-length distribution; the case pins
its **variance** against the Boltzmann stiff-spring limit

$$\operatorname{Var}(r) \;=\; \frac{k_BT}{k_{\text{eff}}},\qquad
k_{\text{eff}} \;=\; k_{\text{bias}} + k_{\text{RHF}} .$$

The oracle is statistical mechanics — not a snapshot of our own output. The
only quantity taken from the shipped machinery is the *input parameter*
`k_RHF` (the RHF/STO-3G bond curvature at the umbrella centre), measured
offline by finite differences, pinned in the source, **and re-measured by the
case itself** as a drift guard.

---

## 1. Derivation

### 1.1 The sampled distribution

The classical canonical distribution of two atoms interacting through a
central potential `U(r)` factorises into a free centre of mass and a relative
coordinate **r**. Integrating out the two angles leaves the marginal in the
bond length (McQuarrie, *Statistical Mechanics*, ch. 12 — the same measure that gives the pair
configuration integral there; the r² is the
spherical volume element, not an approximation):

$$P(r)\,dr \;\propto\; r^{2}\,e^{-\beta U(r)}\,dr ,\qquad \beta = 1/k_BT .$$

Here

$$U(r) \;=\; E_{\text{RHF}}(r) \;+\; \tfrac12 k_{\text{bias}}\,(r-r_0)^2 .$$

The umbrella is exactly quadratic; the QM part is not. Expand it about the
umbrella centre `r₀` with `u = r − r₀`:

$$E_{\text{RHF}}(r_0+u) = E_0 + g\,u + \tfrac12 k_{\text{RHF}}u^2
  + \tfrac16 g_3\,u^3 + \mathcal O(u^4),$$

$$U(r_0+u) = \text{const} + g\,u + \tfrac12 k_{\text{eff}}u^2
  + \tfrac16 g_3 u^3 + \dots,\qquad
  k_{\text{eff}} = k_{\text{bias}} + k_{\text{RHF}} .$$

**The two curvatures add** — that is the whole physics content of the case,
and it is why the umbrella centre is placed at the RHF equilibrium bond length
(§3): a bias centred anywhere else would still add curvatures, but the linear
term `g` would then dominate the *mean* and make the case harder to read.

### 1.2 The harmonic (stiff-spring) limit

Dropping the Jacobian and the cubic term, `P(u) ∝ exp(−β[gu + ½k_eff u²])` is a
Gaussian of variance

$$\boxed{\operatorname{Var}(r) = \frac{k_BT}{k_{\text{eff}}}}\qquad
  \langle u\rangle = -\,g/k_{\text{eff}} .$$

This is equipartition for one quadratic degree of freedom,
`⟨½k_eff u²⟩ = ½k_BT` — the same relation Frenkel & Smit use throughout
(D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed.,
Academic Press 2002; **ch. 3** — the canonical distribution, the Metropolis
rule that samples it, and trial moves with their acceptance-ratio rule of
thumb; **ch. 4** — the equipartition / ensemble-fluctuation relations).

Note what the variance does **not** depend on: `g`. A linear term shifts the
mean and leaves the second central moment untouched at harmonic order. The
pin is therefore insensitive to the umbrella centre being a few 10⁻⁵ Å off the
true RHF stationary point.

### 1.3 The r² Jacobian — assessed, and bounded

Keep the Jacobian. With `2 ln(r₀+u) = 2 ln r₀ + 2u/r₀ − u²/r₀² + O(u³)`,

$$\ln P \simeq \text{const} + \Big(\tfrac{2}{r_0}-\beta g\Big)u
  \;-\;\Big(\tfrac{\beta k_{\text{eff}}}{2}+\tfrac{1}{r_0^{2}}\Big)u^{2},$$

still Gaussian, with

$$\operatorname{Var}(r) = \frac{k_BT}{k_{\text{eff}}}\cdot
   \frac{1}{1+\dfrac{2k_BT}{k_{\text{eff}}r_0^{2}}},\qquad
   \langle u\rangle \supset +\frac{2}{r_0}\operatorname{Var}(r).$$

For this case (`k_eff = 175.684 eV/Å²`, `k_BT = 0.0258520 eV`, `r₀ = 0.7122 Å`)

$$\frac{2k_BT}{k_{\text{eff}}r_0^{2}} = 5.80\times10^{-4}
  \;\;\Rightarrow\;\; \textbf{the Jacobian lowers Var(r) by 0.058\%},$$

i.e. **86× inside the frozen 5% tolerance**. It is therefore *not* folded into
the asserted value: the case pins the plain `k_BT/k_eff`, and this term is one
of the systematics bounded in §1.5. The Jacobian's visible signature is in the
**mean**, where it contributes `+2Var/r₀ = +4.13 × 10⁻⁴ Å` (§1.4).

Transverse motion is irrelevant to the CV by construction: `TranslateMove`
displaces one atom in 3-D, but the umbrella acts on `|r|` only, so rotations of
the molecule change no energy and simply diffuse the orientation. Their effect
on the *radial* marginal is exactly the r² weight treated above.

### 1.4 The mean, as corroboration (not asserted tightly)

Three shifts of `⟨r⟩` away from `r₀`, all small and all of known sign:

| Contribution | Formula | Value [Å] |
|---|---|---|
| Jacobian | `+2·Var/r₀` | +4.133 × 10⁻⁴ |
| Cubic anharmonicity | `−g₃·k_BT/(2k_eff²)` | +1.410 × 10⁻⁴ |
| Residual RHF slope at r₀ | `−g/k_eff` | +0.094 × 10⁻⁴ |
| **Sum** | | **+5.64 × 10⁻⁴** |

with `g = −1.655 × 10⁻³ eV/Å` and `g₃ = −336.7 eV/Å³` (§3). The exact
quadrature (§1.5) gives `⟨r⟩ = 0.7127634 Å`, i.e. `+5.634 × 10⁻⁴` — the three
terms account for it to 6 × 10⁻⁷ Å. The chain measures `+5.80 × 10⁻⁴ Å`.
The case asserts only `|⟨r⟩ − r₀| ≤ 2 × 10⁻³ Å`, a *basin* check (is the chain
in the right well at all?), because the mean — unlike the variance — needs the
cubic term to be predicted at all, which is exactly why the variance is the
observable worth pinning.

### 1.5 Systematics: the exact marginal, by quadrature

The harmonic formula was checked against the **numerically exact** marginal for
the real (anharmonic) RHF curve: `E_RHF(r)` on a 4001-point Simpson grid
spanning `r₀ ± 8σ`, `U(r) = E_RHF(r) + ½k_bias(r−r₀)²`, moments of
`r² e^{−βU(r)}`. Offline, same machinery, tol 1e-12 Ha:

| k_bias [eV/Å²] | k_eff | Var_exact [Å²] | k_BT/k_eff [Å²] | exact / harmonic |
|---|---|---|---|---|
| 60 | 115.684 | 2.2366949 × 10⁻⁴ | 2.2347083 × 10⁻⁴ | 1.00089 |
| **120** | **175.684** | **1.4713715 × 10⁻⁴** | **1.4715057 × 10⁻⁴** | **0.99991** |
| 240 | 295.684 | 0.8740937 × 10⁻⁴ | 0.8743118 × 10⁻⁴ | 0.99975 |

At the chosen stiffness the Jacobian (−0.058%) and the cubic anharmonicity
(+0.049%) nearly cancel, and **every systematic together is 0.009%** — 550×
inside the tolerance. The tolerance is therefore purely a statistical budget
(§5).

---

## 2. Why the case would fail if the QM were not there

| Chain | Var(r) [Å²] | vs the pin |
|---|---|---|
| Umbrella **+ nested RHF** (this case) | 1.475792 × 10⁻⁴ measured | +0.29% |
| Umbrella **only** (control, same seed/δ/N) | 2.197422 × 10⁻⁴ measured | **+49.3%** |
| Umbrella only, analytic `k_BT/k_bias` | 2.152505 × 10⁻⁴ | +46.4% |

`k_RHF` is a **32% share of `k_eff`**, so an energy source that silently stops
contributing (a nested node that is never dispatched, an evaluator that
overwrites instead of accumulating, a frozen density) moves the measured
variance ten tolerances. The control also validates the framework end to end:
with the QM removed the chain reproduces its own analytic value `k_BT/k_bias`
to +2.1% at that (shorter-correlation) state point.

---

## 3. The offline pin: `k_RHF = 55.684 eV/Å²`

**Method.** Five-point central stencil of `E_RHF(r)` at `r₀ = 0.7122 Å`,
RHF/STO-3G, SCF tol `1e-12` Hartree, H₂ in a 20 Å cubic vacuum box, energies
read from `frame.meta.get(kEnergy)` [eV] after `cpu::Evaluate` +
`cpu::scf::RHF`. Measured 2026-07-27 on the branch tip (Atomiverse `dev`,
commit `fc204c63`), GCC 14.3 `-O3 -march=native`, OpenBLAS 0.3.32, single
thread. No third-party package is involved at any point.

$$k = \frac{-E(r_0{+}2h)+16E(r_0{+}h)-30E(r_0)+16E(r_0{-}h)-E(r_0{-}2h)}{12h^2}$$

| h [Å] | 3-point k [eV/Å²] | **5-point k [eV/Å²]** | 5-point g₃ [eV/Å³] |
|---|---|---|---|
| 0.005 | 55.688430 | **55.684153** | −336.672 |
| 0.010 | 55.701262 | **55.684143** | −336.907 |
| 0.020 | 55.752618 | **55.683985** | −337.848 |
| 0.040 | 55.958517 | 55.681410 | −341.672 |
| 0.080 | 56.789839 | 55.636470 | −357.939 |

The 5-point value is flat to 4 × 10⁻⁶ over `h ∈ {0.005, 0.01, 0.02}`; the
3-point column is not converged there because it carries its own O(h²) error
`+(h²/12)·d⁴E/dr⁴`, which the table itself measures as
`d⁴E/dr⁴ ≈ 2.05 × 10³ eV/Å⁴` (consistent at h = 0.005 and h = 0.01). Pinned
value:

> **k_RHF(r₀ = 0.7122 Å) = 55.684 eV/Å²**, `g₃ = −336.7 eV/Å³`,
> `g = dE/dr|_{r₀} = −1.655 × 10⁻³ eV/Å`, stationary point at
> `r_min = 0.712230 Å`.

`E_RHF(r₀) = −30.4088842472 eV = −1.117506 Hartree`.

**The case re-measures this.** `reg_mc_bias_es` runs the same 5-point stencil
(h = 0.01 Å) before the chain and fails if it differs from the pin by more than
0.5% — measured in-case: **55.6841 eV/Å²**, 2 × 10⁻⁶ from the pin. If the
Gaussian SCF surface ever moves, the analytic prediction is stale and must be
re-derived offline; the case says so in its failure message rather than being
re-toleranced.

### 3.1 Literature cross-checks on the pin

1. **Geometry.** NIST CCCBDB (NIST Standard Reference Database 101,
   <https://cccbdb.nist.gov>), *calculated bond lengths*, H₂ (¹Σg⁺), HF/STO-3G:
   `r_e = 0.712 Å` (0.7122 Å at table precision); originally
   W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657
   (1969), DOI 10.1063/1.1672392. The measured stationary point,
   **0.712230 Å**, sits 3 × 10⁻⁵ Å from the umbrella centre — so the two
   springs are concentric to well inside the sampled width (σ_r = 0.0121 Å).
2. **A second, independent in-tree measurement of the same curvature.**
   `regressions/minimizer/rhf_h2_linesearch` reports `k = |f|/|Δr| ≈ 54 eV/Å²`
   from its L-BFGS trace (a two-point secant over a 3 × 10⁻⁴ Å step, so ~3%
   resolution) — consistent with 55.684 and produced by a *different* code path
   (analytic RHF gradient vs. energy finite differences).
3. **Order of magnitude, against experiment.** `k = 55.684 eV/Å² = 892.2 N/m`
   implies, for μ = m_H/2, a harmonic wavenumber
   `ν̃ = (2πc)⁻¹√(k/μ) = 5481 cm⁻¹`. The experimental H₂ harmonic constant is
   `ω_e = 4401.21 cm⁻¹` (K. P. Huber and G. Herzberg, *Constants of Diatomic
   Molecules*, Van Nostrand 1979). The computed value is 25% high — the
   expected sign and rough size for a minimal-basis Hartree–Fock stretch
   (STO-3G puts the bond 0.029 Å too short, and HF stretching frequencies are
   systematically ~10–15% too high before scaling: W. J. Hehre, L. Radom,
   P. v. R. Schleyer, J. A. Pople, *Ab Initio Molecular Orbital Theory*, Wiley
   1986, §6.3). This is a **sanity band**, not a pin.

---

## 4. Chain parameters, acceptance, and seed stability

| Knob | Value | Why |
|---|---|---|
| T | 300 K | `k_BT = 0.0258520 eV` (`units::kB_ev_per_k = 8.617333262e-5`) |
| Umbrella | `HarmonicBias(DistanceCV(tag 1, tag 2), k = 120 eV/Å², centre 0.7122 Å)` | §1.1, §5 |
| Mover | `TranslateMove{δ = 0.04 Å}` | tuned to ≈ 49% acceptance (below) |
| Method | `cpu::scf::RHF`, STO-3G, tol 1e-8 Ha, budget 100 | 3 × 10⁻⁷ eV ≈ 10⁻⁵ k_BT of energy noise |
| Box | cubic 20 Å, molecule centred | COM random walk is ~1 Å over the run |
| Burn-in | 5 000 sweeps | ≈ 2400 autocorrelation times |
| Sampled | 120 000 sweeps, one r per sweep | §5 |
| Seed | 20260727 | one reproducible RNG stream (movers + acceptance) |
| Threads | `mc.set_threads(1)` | §6 |

**δ tuning** (k_bias = 120, 20 000 sampled sweeps per row, same seed). The
Frenkel & Smit rule of thumb (§3.4) is "≈ 50%"; the displacement is 3-D while
the CV is radial, so the useful projection is `δ/√3 ≈ σ_r`:

| δ [Å] | acceptance | τ_int [sweeps] | Var/pred |
|---|---|---|---|
| 0.02 | 0.706 | 4.44 | 1.025 |
| 0.03 | 0.583 | 2.72 | 0.999 |
| **0.04** | **0.489** | **1.97** | **0.964** |
| 0.05 | 0.415 | 1.89 | 0.985 |
| 0.06 | 0.356 | 2.15 | 0.973 |
| 0.08 | 0.279 | 2.71 | 0.994 |
| 0.12 | 0.192 | 4.22 | 1.039 |

δ = 0.04 Å minimises the autocorrelation time; the ±4% scatter of this column
is the 20 000-sweep statistical error (1σ ≈ 2%), which is why the shipped case
samples 6× longer.

**Seed stability at the frozen configuration** (k_bias = 120, δ = 0.04,
burn 5 000, 120 000 sampled; the seed-20260727 row is bit-identical to the
shipped binary's output):

| seed | acceptance | ⟨r⟩ [Å] | Var(r) [Å²] | deviation from `k_BT/k_eff` |
|---|---|---|---|---|
| **20260727** (shipped) | 0.486 | 0.712780 | 1.475792 × 10⁻⁴ | **+0.29%** |
| 12345 | 0.491 | 0.712867 | 1.487903 × 10⁻⁴ | +1.11% |
| 999331 | 0.484 | 0.712846 | 1.467656 × 10⁻⁴ | −0.26% |
| 7 | 0.490 | 0.712837 | 1.467749 × 10⁻⁴ | −0.25% |
| 424242 | 0.488 | 0.712777 | 1.472150 × 10⁻⁴ | +0.04% |

Mean deviation +0.19%, sample σ 0.61%, worst |deviation| 1.11%.

**Thread independence.** The same binary at `OMP_NUM_THREADS` = 1, 8 and unset
produces *bit-identical* statistics (Var = 1.475792 × 10⁻⁴, acceptance 0.486):
the chain is a single RNG stream and every energy is a serial reduction over
two basis functions.

---

## 5. The frozen tolerance: 5% relative on Var(r)

| Budget item | Size | Source |
|---|---|---|
| Statistical, blocking 1σ (blocks of 500 sweeps, τ_int ≈ 2) | **0.77%** | printed by the case |
| Statistical, observed seed-to-seed σ over 5 chains | 0.61% | §4 |
| Worst deviation actually observed | 1.11% | §4 |
| Systematic: r² Jacobian | −0.058% | §1.3 |
| Systematic: cubic + quartic anharmonicity | +0.049% | §1.5 |
| Systematic: all of it, vs exact quadrature | **0.009%** | §1.5 |
| SCF convergence noise (tol 1e-8 Ha vs k_BT) | 10⁻⁵ relative | §4 |

**5% = 6.5 blocking σ = 4.5 × the worst observed deviation**, while the failure
mode the case exists to catch (the QM energy not reaching the acceptance test)
is **+46%**, i.e. 9 tolerances away. Statistics — not physics — sets this
number, and it is a *frozen* number: if a future chain drifts outside it, the
answer is to find what changed, not to widen the band.

Secondary assertions, all with physical (not statistical) bands:

| Assertion | Band | Measured |
|---|---|---|
| in-case FD curvature vs the pin | 0.5% | 2 × 10⁻⁶ |
| largest force component over the whole chain | ≤ 1 × 10⁻⁸ eV/Å | **0** (exactly) |
| SCF iterations per sweep | ≥ 1 and mean ≤ 4 | 3.000 (min 3, max 3) |
| acceptance | [0.35, 0.65] | 0.486 |
| distance of ⟨r⟩ from the umbrella centre | ≤ 2 × 10⁻³ Å | 5.8 × 10⁻⁴ |
| last-accepted energy | finite, in [−31, −29.5] eV | −30.3796 |

---

## 6. Cost: ≤ 60 s, and what pays for it

Measured for the shipped case (125 000 sweeps = 5 000 burn-in + 120 000
sampled, one nested SCF each, plus 5 single points for the FD guard):

| Configuration | CPU time | Wall time |
|---|---|---|
| `set_threads(1)`, node load ≈ 100–350 | **7.1 s** | 9.9 s … 22.2 s |
| (no thread pin) `OMP_NUM_THREADS=8` | 231 s | 35 s |
| (no thread pin) `OMP_NUM_THREADS` unset → 48 threads | — | **> 660 s, killed** |

Two atoms and two basis functions: the OpenMP fork/join per sweep costs orders
of magnitude more than the physics, and the `regressions` target runs binaries
with no `OMP_NUM_THREADS`. The case therefore pins its own thread count
(`mc.set_threads(1)`, and `cpu::Evaluate::set_threads(1)` for the FD guard) —
the same remedy, and for the same reason, as
`regressions/path/neb_muller_brown`. **7.1 s of CPU is 8× inside the 60 s
ceiling**; the wall spread above is contention from other users on a shared
login node, not the case's cost.

**The energy-only veto is part of that budget.** Measured on the same node,
one RHF/STO-3G H₂ single point costs 150 µs without the gradient and 430 µs
with it (2.9×). The trial `energy` phase runs energy-only
(esmm-contract-04), so the chain never pays the gradient: with it the same
125 000 sweeps would cost roughly 6× more (~40 s of CPU) and the case would sit
against the 60 s ceiling instead of 8× inside it.

**`perf_exempt = true`.** There is no external-engine speed twin for this case:
no MC engine in the oracle set (PySCF / CP2K / MetalWalls) samples a Metropolis
chain against a nested Gaussian-SCF trial energy, so a ratio would compare
different algorithms. The cost that *is* asserted is the algorithmic one — SCF
iterations per sweep (≤ 4, measured 3.000) — which is machine-independent.

---

## 7. What this case pins about the engine

1. **Energy-only trial phases** (esmm-contract-04). `rhf.compute_force` is left
   at its default `true`. The *only* thing that suppresses the gradient is MC
   stamping `forces_requested = false` on the `sampling::Phase::energy` slot and
   that veto ANDing down the nesting channel (`ExecServices::forces_requested`)
   into the method's own stages, where `run_nested_erased` computes
   `opts.forces = compute_force && ctx.exec.forces_requested`. The frame's
   forces therefore stay exactly zero for 120 000 sampled sweeps; if the veto
   broke, the RHF + umbrella gradient (~1 eV/Å at the sampled geometries) would
   land there and the case fails on `|f|_max`.
2. **Node-aware MC validation** (esmm-contract-02): the only energy source that
   scores a trial here is a nested **`DriverNode`**, so `McBase::validate_components`
   must cross-cast it to the `HF` kind. Before that fix `mc.build()` threw.
3. **Per-sweep zeroing** (esmm-contract-01): 125 000 accumulating evaluations
   onto a clean slate. Without the `ZeroForces`/`ZeroEnergy` pair at the head of
   the energy phase, the umbrella's `+=` would integrate the whole chain.
4. **The ASPC-under-MC guard stays off-path**: no `predictor()` is set, so warm
   starting is plain last-density reuse — legal under sampling (a converged SCF
   does not remember its guess) and worth a flat 3 SCF iterations per sweep.
   The refusal itself is unit-covered by
   `tests/cpu/scf/test_scf_mc_predictor_guard.cpp`.

## 8. Sources

- D. Frenkel, B. Smit, *Understanding Molecular Simulation: From Algorithms to
  Applications*, 2nd ed., Academic Press, 2002 — ch. 3 (canonical distribution,
  NVT Metropolis, trial moves and the acceptance-ratio rule of thumb), ch. 4
  (fluctuation / equipartition relations).
- D. A. McQuarrie, *Statistical Mechanics*, University Science Books, 2000 —
  ch. 12 (classical configuration integrals; the r² radial measure).
- N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, E. Teller,
  *J. Chem. Phys.* **21**, 1087 (1953), DOI 10.1063/1.1699114 — the sampling
  rule the driver implements.
- G. M. Torrie, J. P. Valleau, *J. Comput. Phys.* **23**, 187 (1977),
  DOI 10.1016/0021-9991(77)90121-8 — umbrella sampling with a harmonic bias on
  a reaction coordinate.
- W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657 (1969),
  DOI 10.1063/1.1672392 — STO-3G; H₂ `r_e = 0.712 Å`.
- NIST CCCBDB, NIST Standard Reference Database 101,
  <https://cccbdb.nist.gov> — calculated bond lengths, H₂, HF/STO-3G.
- K. P. Huber, G. Herzberg, *Molecular Spectra and Molecular Structure IV:
  Constants of Diatomic Molecules*, Van Nostrand Reinhold, 1979 — H₂
  `ω_e = 4401.21 cm⁻¹`.
- W. J. Hehre, L. Radom, P. v. R. Schleyer, J. A. Pople, *Ab Initio Molecular
  Orbital Theory*, Wiley, 1986 — §6.3, systematic overestimation of HF
  stretching frequencies.
