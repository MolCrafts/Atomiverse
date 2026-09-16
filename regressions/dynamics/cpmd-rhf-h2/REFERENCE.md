# Reference — AO Car–Parrinello MD (H₂ NVE, RHF/STO-3G)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic degrees
of freedom on a **Gaussian-basis (AO) mean-field method** — the AO twin of the
GPW case `dynamics/cpmd-water`:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::scf::RHF` with **`cpmd = true`**, **`cpmd_mu = 1000` a.u.**;
  the electronic dt is **not** set on the method — `cpu::md::MD` pushes the
  nuclear dt in from `VelocityVerlet` through the `TimestepProvider` /
  `TimestepConsumer` protocol at build, so nuclei and electrons share one clock
  by construction (there is deliberately **no** `cpmd_dt` member)
- Propagated variables: the occupied MO coefficient columns \(C\) of the
  determinant, at occupancy 2 (restricted, one channel), advanced by
  `cpu::electronic::Cpmd` — velocity Verlet plus the holonomic constraint
  \(C^\mathsf{T} S C = I\)
- Step 1 is a **cold seed**: one full converged SCF plus Born–Oppenheimer
  forces. Every later step is a **fixed-density evaluation** — the Fock matrix
  is rebuilt at the propagated \(C\), forces use the multiplier form of the
  Pulay weight \(W = \mathrm{occ}\,C(C^\mathsf{T} F C)C^\mathsf{T}\), and
  `iterations()` is **0 forever after**
- \(E_{\mathrm{cons}}\) is **not** assembled by this case: `cpu::md::CpmdConserved`
  (bound at its default `end_of_step @ -90`, after `KineticEnergy @ -100`) sums
  \(E_{\mathrm{SCF}} + T_{\mathrm{n}} + T_e + E_{\mathrm{therm}}^{e}\) onto
  `frame.meta[kConserved]` [eV] each step, and the case reads that key — the
  same channel Thermo's `Conserved` column and a user drift `Criterion` see.
  The case re-assembles the four terms once per step purely to prove the
  published channel **is** that documented sum

**Not BOMD** (production steps re-run zero SCF iterations).
**Not XL-BOMD** (no auxiliary density; \(T_e\) is physical to the CP Lagrangian).

## Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985) — the extended
  Lagrangian \(\mathcal{L} = \tfrac{1}{2}\mu\sum_i\langle\dot\psi_i|\dot\psi_i\rangle
  + \tfrac12\sum_I M_I \dot R_I^2 - E[\{\psi\},\{R\}]\) with orthonormality
  constraints
- Marx & Hutter, *Ab Initio Molecular Dynamics: Basic Theory and Advanced
  Methods* (CUP, 2009) **§2.4–2.5** — CP equations of motion, the conserved
  energy \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{e} + E_{\mathrm{el}}\), and
  the adiabaticity / fictitious-mass criterion. The **form** of the gate below
  is theirs; the numeric envelope is an empirical pin (see Tuning)
- Herzberg, *Molecular Spectra and Molecular Structure I* — H₂ \(r_e\) = 0.7414 Å
- GPW twin of this case: `regressions/dynamics/cpmd-water`

## Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂, \(r\) = 0.74 Å, centred in the cell |
| Cell | 20 Å cubic (SCF is non-periodic and ignores it; MD wraps into it, hence the centring) |
| Method / basis | `cpu::scf::RHF` / STO-3G, `tol = 1e-10` Ha |
| Initial velocities | COM-preserving stretch, ±0.002 Å/fs along x |
| dt | 0.05 fs (nuclear = electronic) |
| \(\mu\) | 1000 a.u. |
| Production | 1 SCF seed (3 iterations) + 1 CP baseline + 40 CP drift steps |
| Wall time | ≈ 0.03 s single-threaded (budget 60 s) |

Seed energy \(E_{\mathrm{SCF}}\) = −30.38885 eV (= −1.11675 Ha, the RHF/STO-3G
H₂ value); baseline \(E_{\mathrm{cons}}\) = −30.38828 eV.

## Tuning — μ × dt scan (worst relative \(E_{\mathrm{cons}}\) drift, 40 steps)

Envelopes here are **empirical pins, not literature numbers**: scanned before
freezing, exactly as the spec requires.

| μ [a.u.] \ dt [fs] | 0.05 | 0.10 |
| --- | --- | --- |
| 400  | 4.195e−07 | 1.613e−06 |
| 1000 | **4.195e−07** | 1.613e−06 |
| 2000 | 4.195e−07 | 1.613e−06 |

The drift is **identical to five digits across every μ** — see Honest scope:
on H₂ the fictitious electronic subsystem never moves, so what is measured
here is the *nuclear* Verlet integration error, and only dt changes it. Long
run at the frozen setting: 120 production steps give the same 4.195e−07 (no
secular growth). μ = 1000 is carried only to keep the AO CPMD family on one
setting — on the siblings the plane-wave textbook μ = 400 a.u. is genuinely
unstable (`cpmd-rks-water`, `cpmd-uhf-o2` REFERENCEs).

## Pin

| Observable | Gate | Observed |
| ---------- | ---- | -------- |
| Seed SCF iterations | > 0 and `converged()` | 3, converged |
| Production SCF iterations, **every** post-seed step | **0** (fixed-density) | 0 |
| \(E_{\mathrm{cons}}\) relative drift | ≤ **5e−6** (11.9× observed) | 4.195e−07 |
| \(|E_{\mathrm{cons}} - (E + T_{\mathrm{n}} + T_e + E_{\mathrm{therm}})|\) | ≤ 1e−9 eV | 0.000e+00 (bit-exact) |
| \(T_e\) maximum over production | < 1e−20 eV — the persymmetry identity, gated | 2.3e−32 eV |
| \(T_e\) on the seed step | exactly 0 | 0 |
| \(E_{\mathrm{therm}}^{e}\) (no electronic thermostat wired) | exactly 0 | 0 |
| `last_electronic_ke_ev()` vs `frame.meta[kEnergyElectronKinetic]` | exactly equal | equal |
| Seed \(E_{\mathrm{SCF}}\) | in (−40, −25) eV — a bound H₂ | −30.38885 eV |
| \(r\)(H–H) after the trajectory | in (0.50, 1.20) Å | 0.6955 Å |

**perf_exempt** — no third-party AO-CPMD twin is run at ctest time; the gate is
the CP conservation criterion plus structural sanity.

## Honest scope

- **\(T_e \equiv 0\) here, by symmetry — not by accident and not by bug.** H₂ is
  homonuclear, so its AO Fock matrix is persymmetric at every bond length and
  the 2-function STO-3G space splits into one-dimensional σ_g ⊕ σ_u blocks. The
  occupied σ_g vector is therefore *geometry-independent* and stays an exact
  S-metric eigenvector along the whole trajectory, so the CP band force
  \(-(I - SCC^\mathsf{T})(\partial E/\partial C)\) vanishes identically and the
  fictitious kinetic energy stays at 0 (observed: ≤ 2.3e−32 eV = double-precision
  zero). The case **gates** that identity (\(T_e\) < 1e−20 eV) rather than merely
  reporting it — a nonzero \(T_e\) here would mean the band-force projector
  \((I - SCC^\mathsf{T})\) or the multiplier weight is wrong. The same symmetry
  is documented, with the HeH⁺ counterexample that *does* move, in
  `tests/cpu/scf/test_scf_cpmd.cpp`.
- **What this case therefore does and does not prove.** It *does* pin the CPMD
  machinery end-to-end: the cold SCF seed, zero-iteration production steps, the
  dt-from-integrator protocol, the `kConserved` assembly channel, and the fact
  that the fixed-density energy/force pair is conservative. It does *not*
  exercise a moving electronic subsystem — for that see `cpmd-rks-water`
  (\(T_e\) up to 7.4e−02 eV) and `cpmd-uhf-o2` (dual channel), whose molecules
  have multi-dimensional symmetry blocks and no such freezing.
- Restricted closed shell only; the dual-spin path is `cpmd-uhf-o2`.
- `determinant()` is **not** refreshed on production steps — propagated
  orbitals have no canonical eigenvalues, so the published wavefunction stays
  the seed SCF's. Documented limitation, not an oversight.
- NVE electrons (no electronic Nosé–Hoover thermostat wired).
- Short trajectory (40 production steps ≈ 2 fs): this is a conservation
  envelope, not a sampled ensemble.
