# Reference — electronic structure × molecular mechanics in one driver pass

One `cpu::Evaluate` over one four-atom `Frame` carrying **two energy sources**:

| atoms | role | source | oracle |
|---|---|---|---|
| 0–1 | MM | `cpu::PairLJCut`, He–He at 2.3 Å | the 12-6 closed form at cited ε/σ |
| 2–3 | QM | nested `cpu::scf::RHF`, H₂/STO-3G at 0.7122 Å, 30 Å away | NIST CCCBDB HF/STO-3G H₂ |

**The pin is the SUM.** Each addend is known independently — one analytically,
one from the literature — so

$$\texttt{kEnergy} \;=\; E_{\text{LJ}}(r) \;+\; E_{\text{RHF}}(\mathrm{H_2})$$

is a statement about the *engine's composition contract*, not about either
physics. Before `esmm-contract-05` the five electronic-structure evaluators
**overwrote** `kEnergy` / `kAtom_f*` instead of accumulating, so whichever
source ran first was silently erased and this case could not be written at all.
Run against that code the case reports the QM term alone (§5).

---

## 1. The MM addend — analytic, no fitting

`cpu::PairLJCut` implements the LAMMPS-shaped **shifted** 12-6 potential

$$V(r) = 4\varepsilon\Big[\big(\tfrac{\sigma}{r}\big)^{12} -
\big(\tfrac{\sigma}{r}\big)^{6}\Big] - V(r_c), \qquad
F(r) = \frac{24\varepsilon}{r}\Big[2\big(\tfrac{\sigma}{r}\big)^{12} -
\big(\tfrac{\sigma}{r}\big)^{6}\Big].$$

### 1.1 Parameters

Lennard-Jones (12,6) **helium**: `ε/k_B = 10.22 K`, `σ = 2.556 Å` — the classic
LJ helium pair of J. de Boer and A. Michels, *Physica* **5**, 945 (1938),
tabulated in Hirschfelder, Curtiss & Bird, *Molecular Theory of Gases and
Liquids* (Wiley, 1954), Table I-A, and still the standard LJ helium in the
quantum-fluid literature. With `k_B = 8.617333262 × 10⁻⁵ eV/K` (CODATA 2018;
exact under the 2019 SI redefinition),

$$\varepsilon = 10.22\ \mathrm{K} \times 8.617333262\times10^{-5}\ \mathrm{eV/K}
= 8.806914593764\times10^{-4}\ \mathrm{eV}.$$

**Why helium and not argon.** The spec sketch used two argon atoms with Rahman's
`ε = 0.0104 eV, σ = 3.4 Å`. That is not realizable here: the QM source sees the
*whole* Frame (there is no QM-region mask yet — §3), and the in-tree STO-3G
table covers **Z = 1…8 only** (`include/atomiverse/cpu/scf/basis_data/sto3g.h`:
"Supported elements: H, He, Li, Be, B, C, N, O"). An argon site — real *or*
ghosted, since a ghost still needs its true-Z basis — throws
`cpu::scf::ShellList: 'STO-3G' has no data for Z=18`. Helium is the noble gas
that both the basis table and the LJ literature cover, and its single s shell is
also the smallest possible perturbation of the QM basis (§3).

### 1.2 Geometry and the closed form

`r = 2.3 Å`, `r_c = 2.5 σ = 6.39 Å` (hence `σ/r_c = 0.4` exactly).

| quantity | value |
|---|---|
| `σ/r` | 1.111304347826087 |
| `(σ/r)⁶` | 1.883640765489089 |
| `(σ/r)¹²` | 3.548102533412322 |
| `4ε` | 3.522765837505600 × 10⁻³ eV |
| `V(r)` | +5.863509053874139 × 10⁻³ eV |
| `V(r_c) = 4ε(0.4¹² − 0.4⁶)` | −1.437014666705 × 10⁻⁵ eV |
| **`E_LJ = V(r) − V(r_c)`** | **+5.877879200541189 × 10⁻³ eV** |
| **`F_LJ`** | **+4.790254812142471 × 10⁻² eV/Å** (repulsive) |

Measured by the case: `kEnergyPair = 5.877879200541 × 10⁻³ eV`
(−8.7 × 10⁻¹⁸ eV from the closed form — the last bit of the double) and
`|f_x| = 4.790254812142 × 10⁻² eV/Å` (−4.2 × 10⁻¹⁷).

**Why 2.3 Å and not the minimum** `r* = 2^{1/6}σ = 2.869 Å`. At `r*` the force
vanishes, and the force channel is half of what this case pins. 2.3 Å is on the
repulsive wall at `+5.88 meV = 68 K` — a thermally accessible close contact, not
an exotic geometry — and gives *both* a nontrivial energy (23.5× the literature
tolerance, 5.9 × 10⁷ × the contract tolerance) and a nontrivial force
(0.048 eV/Å, 4.8 × 10⁶ × the force tolerance). Its sign is also deliberate:
positive, so the co-bound total is *less* negative than the QM term alone and a
clobbered MM source moves the answer in a printed, fixed direction.

### 1.3 Only helium carries an LJ site

The per-type constructor `PairLJCut{{{2, 2, ε, σ}}, r_c}` declares one diagonal,
for `Z = 2`. Hydrogen has no diagonal to mix from, so by the documented rule
("a type with no diagonal has no LJ site" — the TIP3P convention) every pair
involving an H gets `ε = 0`. The H–H pair sits 0.7122 Å apart, well inside the
6.39 Å cutoff, and contributes exactly zero — asserted (`mm_f_h == 0`), because
a 12-6 wall at a chemical bond length would otherwise be ~10⁴ eV and drown
everything else in this file.

---

## 2. The QM addend — literature

NIST CCCBDB (NIST Standard Reference Database 101,
<https://cccbdb.nist.gov>), *calculated* HF/STO-3G data for H₂ (¹Σg⁺):
`r_e = 0.712 Å` (0.7122 Å at table precision) and
**`E_total = −1.11751 Hartree`**. The STO-3G basis and that geometry originate
in W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657
(1969), DOI 10.1063/1.1672392. The same pin backs
`regressions/minimizer/rhf_h2_linesearch`, `regressions/mc/mc_bias_es`, and the
unit test `tests/cpu/evaluate/test_source_cobinding.cpp`.

Converted with `units::convert`, `−1.11751 Ha = −30.408996243754 eV`. The case
measures `−1.117505884204 Ha = −30.408884247248 eV`, i.e. **4.1 × 10⁻⁶ Ha from
the quotation** — inside the ±5 × 10⁻⁶ Ha implied by five quoted decimals (our
value rounds to `−1.11751`). That quantization, not our SCF (converged to
10⁻¹⁰ Ha), is what sets the absolute tolerance in §4.

**On Szabo & Ostlund.** The spec sketch asked for "Szabo & Ostlund, Table 3.11".
*Modern Quantum Chemistry* (revised first edition, Dover 1996) §3.5 works the
minimal-basis H₂ problem in full and is the textbook treatment of the method
used here, but its worked STO-3G number is quoted at the **experimental**
separation `R = 1.4 a₀ = 0.7408 Å`, which is a different geometry from the
STO-3G optimum used in this case, and the exact table number could not be
verified offline. It is therefore cited as background, **not** as the source of
the asserted digits; the digits are CCCBDB's, at CCCBDB's stated precision.
(Sibling in-tree cases quote "−1.117506 Ha"; that is a six-decimal rendering of
*our own* converged value, which is a golden snapshot and not admissible as the
oracle here.)

---

## 3. Keeping the MM sites out of the QM Hamiltonian

`cpu::scf::RHF` runs on the whole Frame: there is no QM-region selector on the
public surface (`qm_z`-style masking is the campaign's Wave 2). The only public
way to say "these atoms are not QM nuclei" today is `ghost_atoms` — documented
BSSE/counterpoise semantics: **basis functions only, zero nuclear charge, zero
electrons**. The case sets `rhf.ghost_atoms = {0, 1}`, so the SCF solves H₂ and
the He pair contributes two (empty) s functions 30 Å away.

**They are inert, by construction and by measurement.** The most diffuse STO-3G
primitives are α = 0.31364979 a₀⁻² (He) and β = 0.1688554 a₀⁻² (H); their
Gaussian product decays as `exp(−μR²)` with
`μ = αβ/(α+β) = 0.1098 a₀⁻²` and `R = 30 Å = 56.69 a₀`, i.e.
`exp(−352.8) ≈ 10⁻¹⁵³`. The case does not rely on that estimate: a control leg
runs the same RHF on a **bare two-atom H₂ frame** and asserts the ghosted energy
equals it to 10⁻¹⁰ eV (measured: identical, difference 0.00 × 10⁰ eV) and the H
force to 10⁻⁸ eV/Å. A third assertion pins that a ghost site carries **no** QM
force (measured |f| < 10⁻²⁶⁰ eV/Å) — which is exactly why an *overwriting*
gradient write is a total loss of the MM force on those atoms, not a small
perturbation (§5).

This is deliberately **not** QM/MM electrostatic embedding — there is no
coupling between the regions at all. That is the point: a coupled case cannot
have two independently known addends, and this case exists to pin the summation,
not the coupling. Electrostatic embedding is `esmm-qmmm-03`.

---

## 4. Tolerances — two tiers, two different claims

| tier | asserted quantities | tolerance | measured |
|---|---|---|---|
| **literature** | `kEnergy` vs `E_LJ + E_CCCBDB`; `kEnergyElectronic` vs `E_CCCBDB` | **2.5 × 10⁻⁴ eV** (8.2 × 10⁻⁶ relative) | 1.12 × 10⁻⁴ eV |
| **contract** | `kEnergy` vs `kEnergyPair + kEnergyElectronic`; co-bound vs the two sole-source legs; `kEnergyPair` vs the closed form; either binding order | **1 × 10⁻¹⁰ eV** (3 × 10⁻¹² relative) | **0.00 eV** (bit-exact) |
| forces | MM force vs closed form; QM force vs its sole-source leg; antisymmetry; Σf = 0 | **1 × 10⁻⁸ eV/Å** (project FP table, force/exact) | ≤ 4.2 × 10⁻¹⁷ eV/Å |

The literature tier can be no tighter than the ±5 × 10⁻⁶ Ha = 1.4 × 10⁻⁴ eV
quantization of a five-decimal quotation; 2.5 × 10⁻⁴ eV is 2.2× that, and the MM
term it must not hide is 23.5× larger still.

The contract tier carries the weight, and it needs **no** literature precision:
those comparisons are between channels of one run, or between legs of one
binary — the same doubles added in a different order. `+=` makes them agree
bit-for-bit; overwriting makes them disagree by a whole MM energy
(5.9 × 10⁷ tolerances). The chain that closes the absolute claim is

```
kEnergy  ==  E_LJ(closed form) + E_QM(sole-source leg)      [exact,  1e-10 eV]
E_QM(sole-source leg)  ==  E_CCCBDB ± 1.4e-4 eV             [literature]
⇒ kEnergy == E_LJ + E_CCCBDB ± 1.4e-4 eV
```

so every step is pinned by something that is not our own output.

---

## 5. What the case reports under the old contract

`MeanFieldMethod`'s epilogue used to write `frame.meta.get(kEnergy) = …` and
`fx[A] = …`. Under that code, with the MM source bound first:

| channel | shipped (`+=`) | pre-`esmm-contract-05` (`=`) |
|---|---|---|
| `kEnergy` | −30.403006368048 eV | **−30.408884247248 eV** (the QM term alone) |
| `kEnergyPair` | +5.877879 × 10⁻³ eV | +5.877879 × 10⁻³ eV (still written) |
| `kEnergy − kEnergyPair − kEnergyElectronic` | 0 | **+5.88 × 10⁻³ eV** |
| `f_x` on each He | ∓4.790255 × 10⁻² eV/Å | **~10⁻²⁹¹ eV/Å** (the ghosts' own gradient) |

Note the second row: the per-term key survives the clobber, so
"`kEnergyPair` is right" proves nothing — it is the **decomposition invariant**
(row 3) and the total that fail. The force row is the starkest: the MM force is
not perturbed, it is *deleted*.

The case also runs both binding orders. Under the old contract the QM-first
order accidentally passed (the accumulating source happened to run last), which
is precisely why "it worked when I bound them the other way round" was a real
user experience and not a contract.

**Per-term key semantics.** `ZeroEnergy` clears `kEnergy` and every per-term key
in lockstep at `evaluate::Phase::prepare` (orders −1000/−999/−998), and each
source accumulates into both the total and its own term key —
`kEnergyPair` for pair styles, `kEnergyElectronic` for GPW / MFM / PW / AM1 /
RM1 / DFTB2. The honest invariant is

> `kEnergy == Σ per-term + unclassified`

not `kEnergy == Σ per-term`: a `Bias` component accumulates into the total and
owns **no** term key, so its contribution is unclassified. That gap is
pre-existing, deliberately named debt (`esmm-contract-05`, "Out of scope"; the
comment lives at `src/cpu/component/zero.cpp`). This case has no `Bias`
component, so here the remainder is exactly zero and the invariant is asserted
in its strict form.

---

## 6. Cost — and `perf_exempt = true`

Measured on the login node, `OMP_NUM_THREADS` unset, five evaluation passes
(4 legs + the ghost control), each a 2-electron SCF over 2–4 basis functions:

| | value |
|---|---|
| wall time | **0.02 s** |
| ceiling | 60 s (3000× inside) |

Every driver pins `set_threads(1)`, the remedy `regressions/path/neb_muller_brown`
and `regressions/mc/mc_bias_es` needed for the same reason: the `regressions`
target runs binaries with no `OMP_NUM_THREADS`, and on a 48-core login node the
OpenMP fork/join per pass costs orders of magnitude more than two basis
functions of physics. Single-threading is also what makes the bit-exact contract
tier (§4) reproducible — reduction order must not vary.

**`perf_exempt = true`.** There is no external-engine speed twin for this case.
It is an *architecture* case: no engine in the oracle set (PySCF / CP2K /
MetalWalls) exposes "one classical pair style and one SCF method accumulating
into the same energy channel in one pass" as a comparable operation, so a ratio
would compare different things. The physics in it is two textbook numbers whose
cost is microseconds.

---

## 7. What this case pins about the engine

1. **The accumulation contract** (`esmm-contract-05`): every in-plan source
   `+=`s into `kEnergy` / `kEnergy*` / `kAtom_f*`, and the hosting driver owns
   the zeroing. The first in-tree end-to-end assertion that two *heterogeneous*
   sources sum.
2. **Driver-owned zeroing** (`esmm-contract-01`): four `Evaluate` passes run
   over the same Frame in sequence and none of them integrates the previous
   pass's energy or force.
3. **`kEnergyElectronic`**: the electronic share is observable as its own term,
   and the strict decomposition invariant holds when no `Bias` is bound.
4. **Order independence**: `on(phase).add(…, order)` is scheduling, not physics
   — both orders agree bit-for-bit.
5. **A nested `Driver` node as an energy source**: `cpu::scf::RHF` is bound into
   another driver's phase and its `run_nested_erased` entry is the accumulating
   one, while its standalone `run(Frame&)` keeps overwrite semantics (a sole
   author cannot have a co-author) — that split is what the two sole-source legs
   and the co-bound legs exercise together.
6. **`ghost_atoms` as today's only public QM/MM partition**, and its inertness.

## 8. Sources

- J. de Boer, A. Michels, *Physica* **5**, 945 (1938) — LJ(12,6) helium,
  `ε/k_B = 10.22 K`, `σ = 2.556 Å`.
- J. O. Hirschfelder, C. F. Curtiss, R. B. Bird, *Molecular Theory of Gases and
  Liquids*, Wiley, 1954 — Table I-A, LJ parameters of the noble gases.
- J. E. Lennard-Jones, *Proc. R. Soc. Lond. A* **106**, 463 (1924) — the 12-6
  functional form.
- NIST CCCBDB, NIST Standard Reference Database 101,
  <https://cccbdb.nist.gov> — calculated HF/STO-3G total energy and bond length,
  H₂ (¹Σg⁺).
- W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657 (1969),
  DOI 10.1063/1.1672392 — STO-3G; H₂ `r_e = 0.712 Å`.
- S. F. Boys, F. Bernardi, *Mol. Phys.* **19**, 553 (1970) — counterpoise /
  ghost (basis-only) sites, the semantics `ghost_atoms` implements.
- A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, revised 1st ed., Dover,
  1996 — §3.5, minimal-basis H₂ (background; see §2 for why it is not the pin).
- CODATA 2018 / SI 2019 — `k_B = 8.617333262 × 10⁻⁵ eV/K` (exact).
