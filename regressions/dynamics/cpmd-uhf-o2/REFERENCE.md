# Reference — AO Car–Parrinello MD (triplet O₂ NVE, UHF/STO-3G)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic degrees
of freedom on an **unrestricted Gaussian-basis mean-field method** — the
dual-channel leg of the AO CPMD family:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::scf::UHF{9, 7}` with **`cpmd = true`**,
  **`cpmd_mu = 1000` a.u.**; the electronic dt is **not** set on the method —
  `cpu::md::MD` pushes the nuclear dt in from `VelocityVerlet` through the
  `TimestepProvider` / `TimestepConsumer` protocol at build (there is
  deliberately **no** `cpmd_dt` member)
- Propagated variables: **two independent** fictitious electronic subsystems —
  the 9 occupied α and 7 occupied β MO coefficient columns, each at **occupancy
  1**, each with its own `cpu::electronic::Cpmd` propagator and its own
  \(C_\sigma^\mathsf{T} S C_\sigma = I\) constraint. This is the case that fails
  if the β channel is dropped, mis-occupied, or aliased onto α
- Step 1 is a **cold seed** (one converged UHF SCF + Born–Oppenheimer forces);
  every later step is a **fixed-density evaluation** with the per-spin
  multiplier weight \(W_\sigma = C_\sigma(C_\sigma^\mathsf{T} F_\sigma
  C_\sigma)C_\sigma^\mathsf{T}\), and `iterations()` is **0** forever after
- \(E_{\mathrm{cons}}\) comes off `frame.meta[kConserved]` [eV], published each
  step by `cpu::md::CpmdConserved` at `end_of_step @ -90` (after
  `KineticEnergy @ -100`). `kEnergyElectronKinetic` carries the **sum** of the
  two channels' fictitious kinetic energies. The case re-assembles
  \(E_{\mathrm{UHF}} + T_{\mathrm{n}} + T_e + E_{\mathrm{therm}}^{e}\) once per
  step only to prove the published channel is that documented sum

**Not BOMD** (production steps re-run zero SCF iterations).
**Not XL-BOMD** (no auxiliary density; \(T_e\) is physical to the CP Lagrangian).

## Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985) — the extended
  Lagrangian and the fictitious electronic mass μ
- Marx & Hutter, *Ab Initio Molecular Dynamics: Basic Theory and Advanced
  Methods* (CUP, 2009) **§2.4–2.5** — CP equations of motion, the conserved
  energy \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{e} + E_{\mathrm{el}}\), and
  the adiabaticity criterion. The **form** of the gates below is theirs; the
  numeric envelope is an empirical pin (see Tuning)
- Huber & Herzberg, *Constants of Diatomic Molecules* — O₂ X³Σ_g⁻,
  \(r_e\) = 1.2075 Å
- GPW twin of the unrestricted shape: `regressions/dynamics/cpmd-uks`

## Setup

| Item | Value |
| ---- | ----- |
| System | Isolated triplet O₂, \(r\) = 1.21 Å, centred in the cell |
| Cell | 20 Å cubic (SCF is non-periodic and ignores it; MD wraps into it, hence the centring) |
| Method / basis | `cpu::scf::UHF`, \(N_\alpha\) = 9 / \(N_\beta\) = 7 / STO-3G, `tol = 1e-9` Ha |
| Configuration | 1σ_g² 1σ_u² 2σ_g² 2σ_u² 3σ_g² 1π_u⁴ 1π_g²(↑↑) — high-spin triplet |
| Initial velocities | COM-preserving stretch, ±0.001 Å/fs along x |
| dt | 0.05 fs (nuclear = electronic) |
| \(\mu\) | 1000 a.u. |
| Production | 1 SCF seed (21 iterations) + 1 CP baseline + 40 CP drift steps |
| Wall time | ≈ 1.3 s single-threaded (budget 60 s) |

Seed energy \(E_{\mathrm{UHF}}\) = −4010.41922 eV; baseline
\(E_{\mathrm{cons}}\) = −4010.41835 eV.

## Tuning — μ × dt scan (worst relative \(E_{\mathrm{cons}}\) drift, 40 steps)

Envelopes here are **empirical pins, not literature numbers**: scanned before
freezing, exactly as the spec requires.

| μ [a.u.] \ dt [fs] | 0.05 | 0.10 |
| --- | --- | --- |
| 400  | 1.146e−07 | 5.459e−02 (\(T_e\) → 161 eV — adiabaticity lost) |
| 1000 | **2.810e−07** | 5.352e−07 |
| 2000 | 2.620e−07 | 1.199e−06 |

Stability margin at dt = 0.05 fs, 120 production steps (3× the pinned length):
μ = 400 → 6.900e−03 with \(T_e\) → 24.6 eV (**heating**); μ = 1000 →
4.184e−07 with \(T_e\) ≤ 1.6e−02 eV (**flat**).

**Why not the plane-wave textbook μ = 400 a.u.:** it looks fine over 40 steps
but is already outside the adiabatic window — the electrons heat by five orders
of magnitude by step 120, and at dt = 0.1 fs the run is destroyed inside 40.
μ = 1000 at dt = 0.05 fs is the scanned stable window and is the setting shared
by all three AO CPMD cases.

## Pin

| Observable | Gate | Observed |
| ---------- | ---- | -------- |
| Seed SCF iterations | > 0 and `converged()` | 21, converged |
| Seed determinant \(N_\alpha\) / \(N_\beta\) | exactly 9 / 7 | 9 / 7 |
| Seed spin contamination \(\langle S^2\rangle - S(S+1)\) | \|·\| < 0.5 (clean high spin) | 0.0128 |
| Production SCF iterations, **every** post-seed step | **0** (fixed-density) | 0 |
| \(E_{\mathrm{cons}}\) relative drift | ≤ **5e−6** (17.8× observed) | 2.810e−07 |
| \(|E_{\mathrm{cons}} - (E + T_{\mathrm{n}} + T_e + E_{\mathrm{therm}})|\) | ≤ 1e−9 eV | 0.000e+00 (bit-exact) |
| \(T_e\) minimum over production | **> 0** (both channels really move) | 2.10e−10 eV |
| \(T_e\) maximum over production | < 1 eV (adiabaticity: electrons stay cold) | 7.67e−03 eV |
| \(T_e\) on the seed step | exactly 0 | 0 |
| \(E_{\mathrm{therm}}^{e}\) (no electronic thermostat wired) | exactly 0 | 0 |
| Seed \(E_{\mathrm{UHF}}\) | in (−4200, −3800) eV — a bound O₂ | −4010.41922 eV |
| \(r\)(O–O) after the trajectory | in (1.00, 1.45) Å | 1.2506 Å |

**perf_exempt** — no third-party AO-CPMD twin is run at ctest time; the gate is
the CP conservation + adiabaticity criteria plus spin and structural sanity.

## Honest scope

- **O₂ is homonuclear, but \(T_e\) is genuinely nonzero here — and that is a
  real distinction from the H₂ sibling.** In `cpmd-rhf-h2` the persymmetric Fock
  matrix and the *one-dimensional* σ_g / σ_u blocks of a 2-function basis freeze
  the occupied vector, so \(T_e \equiv 0\) by symmetry. In O₂/STO-3G the σ_g and
  σ_u blocks are **three-dimensional** (1s, 2s, 2p_z combinations), so the
  occupied subspace mixes as the bond length changes, the carried-forward
  coefficients are no longer exact eigenvectors of the new Fock matrix, and both
  fictitious channels acquire real kinetic energy. The case *verifies* this
  (\(T_e^{\min} > 0\) is a hard gate) rather than assuming it.
- `determinant()` is **not** refreshed on production steps — propagated
  orbitals have no canonical eigenvalues, so the published wavefunction (and
  therefore the pinned \(\langle S^2\rangle\) and occupation numbers) is the
  **seed SCF's**. That is the documented limitation, and it is why the spin gate
  is stated as a property of the launch state.
- Spin contamination is a *sanity* bound (0.0128 ≪ 0.5), not an accuracy claim;
  minimal-basis high-spin UHF O₂ is only mildly contaminated.
- Minimal STO-3G basis, chosen for the ≤ 60 s budget. STO-3G overstretches this
  bond (the trajectory relaxes 1.21 → 1.25 Å), so absolute energies and the
  equilibrium geometry are **not** the physical answer — this case gates
  *conservation and channel wiring*, not accuracy.
- NVE electrons (no electronic Nosé–Hoover thermostat wired).
- Short trajectory (40 production steps ≈ 2 fs): a conservation envelope, not a
  sampled ensemble.
