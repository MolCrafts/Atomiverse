# dynamics/car-parrinello

One case; every section below is run by `atv_regressions --case dynamics/car-parrinello` and keeps its own oracle record.


## Reference — Car–Parrinello MD short trajectory (H₂O NVE, GPW)

### Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on closed-shell Γ-point GPW:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with **`cpmd = true`**,
  **`cpmd_mu = 2000` a.u.**, `compute_forces = true`; the electronic dt is
  **not** set on the evaluator — `cpu::md::MD` pushes the nuclear dt in from
  the `VelocityVerlet` integrator through the `TimestepProvider` /
  `TimestepConsumer` protocol at build, so nuclei and electrons share one clock
  by construction
- Propagator: `cpu::electronic::Cpmd` — position Verlet on occupied MO coefficients
  \(C\) with Löwdin re-orthonormalization
- \(E_{\mathrm{cons}}\) is **not** assembled by this case: `cpu::md::CpmdConserved`
  (bound at its default `end_of_step @ -90`, after `KineticEnergy @ -100`) sums
  \(E_{\mathrm{KS}} + T_{\mathrm{n}} + T_e + E_{\mathrm{therm}}^{e}\) onto
  `frame.meta[kConserved]` [eV] each step, and the case reads that key — the
  same channel Thermo's `Conserved` column and a user drift `Criterion` see.
  \(T_e\) likewise comes off `frame.meta[kEnergyElectronKinetic]`, published by
  every CPMD launch, rather than off an evaluator readback

**Not BOMD** (production steps use fixed-density KS evaluation, `iters==0`).
**Not XL-BOMD** (no auxiliary density + SCF; electronic kinetic energy is
physical to the CP Lagrangian).

### Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
  \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}}\)
- Twin geometry / basis family: `gpw-bomd-water`, `xl-bomd-water`

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. (AO residual stiffer than PW textbook ~400) |
| Production | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

### Pin

| Observable | Gate |
| ---------- | ---- |
| \(E_{\mathrm{cons}}\) relative drift | ≤ 0.10 |
| Production SCF iters | **0** (fixed-density) |
| O–H distances | < 1.8 Å |
| PE after warm-up | < −100 eV |

**perf_exempt** — no CP2K CPMD twin wall on the worker host; literature
envelope + structural gates.

### Honest scope

- Closed-shell RKS Γ only; UKS / OT / k-points refused at `build()`
- Löwdin reorth foothold (not full RATTLE)
- No electronic thermostat
- Soft drift tol vs BO (0.10 vs 0.05) for short Löwdin foothold traj


## Reference — AO Car–Parrinello MD (H₂ NVE, RHF/STO-3G)

### Method

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

### Literature

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
- GPW twin of this case: `regressions/dynamics/car-parrinello`

### Setup

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

### Tuning — μ × dt scan (worst relative \(E_{\mathrm{cons}}\) drift, 40 steps)

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

### Pin

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

### Honest scope

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


## Reference — AO Car–Parrinello MD (H₂O NVE, RKS/LSDA/STO-3G)

### Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic degrees
of freedom on a **Gaussian-basis Kohn–Sham method** — the AO/KS leg of the AO
CPMD family, and the AO twin of the GPW case `dynamics/cpmd-water` (same
nuclei, same initial kick, Gaussian SCF in the force slot instead of GPW):

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::scf::RKS{XCType::LSDA}` with **`cpmd = true`**,
  **`cpmd_mu = 1000` a.u.**; the electronic dt is **not** set on the method —
  `cpu::md::MD` pushes the nuclear dt in from `VelocityVerlet` through the
  `TimestepProvider` / `TimestepConsumer` protocol at build (there is
  deliberately **no** `cpmd_dt` member)
- Propagated variables: the occupied MO coefficient columns \(C\) at occupancy
  2 (restricted, one channel), advanced by `cpu::electronic::Cpmd` — velocity
  Verlet plus the holonomic constraint \(C^\mathsf{T} S C = I\)
- Step 1 is a **cold seed** (one converged KS SCF + Born–Oppenheimer forces);
  every later step is a **fixed-density KS evaluation** — the Kohn–Sham matrix,
  the XC energy and the **XC gradient** are rebuilt at the propagated,
  *non-stationary* density, and `iterations()` is **0** forever after. This is
  the leg that fails if the KS energy branch or the XC force term is only
  correct at a converged density
- \(E_{\mathrm{cons}}\) comes off `frame.meta[kConserved]` [eV], published each
  step by `cpu::md::CpmdConserved` at `end_of_step @ -90` (after
  `KineticEnergy @ -100`, so the nuclear term is this step's). The case
  re-assembles \(E_{\mathrm{KS}} + T_{\mathrm{n}} + T_e + E_{\mathrm{therm}}^{e}\)
  once per step only to prove the published channel is that documented sum

**Not BOMD** (production steps re-run zero SCF iterations).
**Not XL-BOMD** (no auxiliary density; \(T_e\) is physical to the CP Lagrangian).

### Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985) — the extended
  Lagrangian and the fictitious electronic mass μ
- Marx & Hutter, *Ab Initio Molecular Dynamics: Basic Theory and Advanced
  Methods* (CUP, 2009) **§2.4–2.5** — CP equations of motion, the conserved
  energy \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{e} + E_{\mathrm{el}}\), and
  the adiabaticity criterion (electronic subsystem must stay cold relative to
  the nuclear one). The **form** of the gates below is theirs; the numeric
  envelope is an empirical pin (see Tuning)
- Geometry: CP2K Quickstep getting-started H₂O — shared with the twin family
  `gpw-bomd-water`, `xl-bomd-water`, `cpmd-water`

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O, CP2K getting-started geometry (\(r_{\mathrm{OH}}\) ≈ 0.911 Å) |
| Cell | 10 Å cubic (SCF is non-periodic and ignores it; MD wraps into it, and the molecule is centred at 5 Å) |
| Method / XC / basis | `cpu::scf::RKS` / LSDA / STO-3G, `tol = 1e-9` Ha |
| Initial velocities | COM-preserving H stretch, ±0.0015 Å/fs along x |
| dt | 0.05 fs (nuclear = electronic) |
| \(\mu\) | 1000 a.u. |
| Production | 1 SCF seed (9 iterations) + 1 CP baseline + 40 CP drift steps |
| Wall time | ≈ 2.0 s single-threaded (budget 60 s) |

Seed energy \(E_{\mathrm{KS}}\) = −2032.92949 eV; baseline
\(E_{\mathrm{cons}}\) = −2032.92856 eV.

### Tuning — μ × dt scan (worst relative \(E_{\mathrm{cons}}\) drift, 40 steps)

Envelopes here are **empirical pins, not literature numbers**: scanned before
freezing, exactly as the spec requires.

| μ [a.u.] \ dt [fs] | 0.05 | 0.10 |
| --- | --- | --- |
| 400  | 3.434e−06 | **diverges** (eigensolver failure ≈ step 25) |
| 1000 | **5.851e−06** | 1.553e−04 |
| 2000 | 5.204e−06 | 1.415e−05 |

Stability margin at dt = 0.05 fs, 120 production steps (3× the pinned length):
μ = 400 **diverges**; μ = 1000 → 1.789e−05 with \(T_e\) ≤ 0.109 eV;
μ = 2000 → 1.059e−05.

**Why not the plane-wave textbook μ = 400 a.u.:** the AO residual is stiffer
than a plane-wave one, so the CP frequency spectrum at μ = 400 overlaps the
Verlet stability limit even at dt = 0.05 fs — the run survives 40 steps but
loses adiabaticity and dies inside 120. μ = 1000 at dt = 0.05 fs is the scanned
stable window and is the setting shared by all three AO CPMD cases. (The GPW
twin `cpmd-water` needed μ = 2000 at dt = 0.1 fs for the same reason.)

### Pin

| Observable | Gate | Observed |
| ---------- | ---- | -------- |
| Seed SCF iterations | > 0 and `converged()` | 9, converged |
| Production SCF iterations, **every** post-seed step | **0** (fixed-density) | 0 |
| \(E_{\mathrm{cons}}\) relative drift | ≤ **5e−5** (8.5× observed) | 5.851e−06 |
| \(|E_{\mathrm{cons}} - (E + T_{\mathrm{n}} + T_e + E_{\mathrm{therm}})|\) | ≤ 1e−9 eV | 0.000e+00 (bit-exact) |
| \(T_e\) minimum over production | **> 0** (the electrons really move) | 1.13e−07 eV |
| \(T_e\) maximum over production | < 1 eV (adiabaticity: electrons stay cold) | 7.36e−02 eV |
| \(T_e\) on the seed step | exactly 0 | 0 |
| \(E_{\mathrm{therm}}^{e}\) (no electronic thermostat wired) | exactly 0 | 0 |
| Seed \(E_{\mathrm{KS}}\) | in (−2200, −1900) eV — a bound H₂O | −2032.92949 eV |
| O–H distances after the trajectory | in (0.70, 1.30) Å | 1.0239 / 1.0239 Å |

**perf_exempt** — no third-party AO-CPMD twin is run at ctest time; the gate is
the CP conservation + adiabaticity criteria plus structural sanity.

### Honest scope

- Restricted **closed-shell** KS only; the dual-spin (UKS-shaped) fictitious
  subsystem is pinned by the UHF sibling `cpmd-uhf-o2`.
- LSDA only. GGA/hybrid XC gradients at a non-stationary density are the same
  code path but are not pinned here.
- Minimal STO-3G basis, chosen for the ≤ 60 s budget. Absolute energies are
  minimal-basis numbers and are **not** the physical answer — this case gates
  *conservation*, not accuracy.
- `determinant()` is **not** refreshed on production steps — propagated
  orbitals have no canonical eigenvalues, so the published wavefunction stays
  the seed SCF's. Documented limitation, not an oversight.
- NVE electrons (no electronic Nosé–Hoover thermostat wired).
- Short trajectory (40 production steps ≈ 2 fs): this is a conservation
  envelope, not a sampled ensemble. Beyond ≈ 120 steps at this μ/dt the
  Löwdin/RATTLE foothold starts to heat the electrons — deliberately outside
  the pinned window.


## Reference — AO Car–Parrinello MD (triplet O₂ NVE, UHF/STO-3G)

### Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic degrees
of freedom on an **unrestricted Gaussian-basis mean-field method** — the
dual-channel leg of the AO CPMD family:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::scf::UHF{cpu::scf::BasisSet::STO3G, 9, 7}` with **`cpmd = true`**,
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

### Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985) — the extended
  Lagrangian and the fictitious electronic mass μ
- Marx & Hutter, *Ab Initio Molecular Dynamics: Basic Theory and Advanced
  Methods* (CUP, 2009) **§2.4–2.5** — CP equations of motion, the conserved
  energy \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{e} + E_{\mathrm{el}}\), and
  the adiabaticity criterion. The **form** of the gates below is theirs; the
  numeric envelope is an empirical pin (see Tuning)
- Huber & Herzberg, *Constants of Diatomic Molecules* — O₂ X³Σ_g⁻,
  \(r_e\) = 1.2075 Å
- GPW twin of the unrestricted shape: `regressions/dynamics/car-parrinello`

### Setup

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

### Tuning — μ × dt scan (worst relative \(E_{\mathrm{cons}}\) drift, 40 steps)

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

### Pin

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

### Honest scope

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


## Reference — Car–Parrinello MD on Γ-UKS (O₂ triplet NVE, GPW)

### Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on **collinear Γ-point UKS** GPW:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with **`cpmd = true`**,
  **`n_alpha = 7`**, **`n_beta = 5`** (O₂ GTH-q6 triplet),
  **`cpmd_mu = 2000` a.u.**, `compute_forces = true`; the electronic dt is
  **not** set on the evaluator — `cpu::md::MD` pushes the nuclear dt in from
  the `VelocityVerlet` integrator through the `TimestepProvider` /
  `TimestepConsumer` protocol at build
- Propagator: dual `cpu::electronic::Cpmd` instances (α and β) with electronic
  RATTLE; occupancy-1 S-tangent force
  \(\mu\,\ddot C_\sigma = -(I-S C_\sigma C_\sigma^\top)(2 H_\sigma C_\sigma)\)

**Not BOMD** (production steps use fixed-density dual-KS evaluation,
`iters==0`). **Not XL-BOMD** (`xl-bomd-uks`). **Not** second-gen CP.

### Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- H. C. Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
  \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}}\)
  with \(T_{\mathrm{e}} = T_{\mathrm{e}}^\alpha + T_{\mathrm{e}}^\beta\)
- Twin geometry / UKS family: `xl-bomd-uks`, `gpw-uks-forces`,
  `gpw-uks-o2-gamma`
- Closed-shell CPMD twin: `cpmd-rattle-water`

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated O₂ triplet (\(R=1.2075\) Å) |
| Cell | 12 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PADE LSD / DZVP-MOLOPT-GTH / GTH-PBE-q6 |
| CUTOFF | 200 Ry |
| Spin | \(n_\alpha=7\), \(n_\beta=5\) (multiplicity 3) |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. (AO residual stiffer than PW textbook ~400) |
| Production | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

### Pin

| Observable | Gate |
| ---------- | ---- |
| \(E_{\mathrm{cons}}\) relative drift | ≤ 0.10 |
| Production SCF iters | **0** (fixed-density) |
| ⟨S²⟩ | [1.5, 2.5] (triplet window) |
| O–O distance | [0.9, 1.6] Å |
| PE after warm-up | < −100 eV |

**Offline CP2K twin input:** `tools/cp2k_oracle/inputs/cpmd_uks_o2_nve.inp`
(QS PADE/LSD, DZVP-MOLOPT-GTH, GTH-PADE, MT 12 Å, MULTIPLICITY 3, short NVE).
Pin is a **literature conservation envelope**, not a live CP2K energy twin
in ctest.

**perf_exempt** — no CP2K CPMD-UKS wall twin on the worker host; literature
envelope + structural / spin gates (same inventory class as `cpmd-rattle-water`,
`xl-bomd-uks`).

### Honest scope

- Collinear Γ-UKS only (RKS CPMD remains product on the same flag)
- OT / XL / second-gen-CP / k-point CPMD refused or out of scope
- Electronic RATTLE (pos + vel); geometry transport Löwdin + vel project
- No electronic thermostat


## Reference — multi-k Car–Parrinello MD short trajectory (H₂ NVE, GPW)

### Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on **multi-k closed-shell RKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces (Bloch \(C_k\)):

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  **`cpmd = true`**, **`cpmd_mu`**; the electronic dt is **not** set on the
  evaluator — `cpu::md::MD` pushes the nuclear dt in from the `VelocityVerlet`
  integrator through the `TimestepProvider` / `TimestepConsumer` protocol at
  build
- One `cpu::electronic::CpmdCplx` per sampled k propagates occupied Bloch
  coefficients with electronic RATTLE; production steps use fixed-density
  multi-k KS eval (`iters == 0`)
- Conserved diagnostic:
  \(E_{\mathrm{cons}} = T_n + T_e + E_{\mathrm{KS}}\) with
  \(T_e = \sum_k w_k\,\tfrac12\mu\lVert\dot C_k\rVert_F^2\)
- Periodic cubic cell; PADE XC
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:**

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — electronic RATTLE
- Γ product paths: `regressions/dynamics/car-parrinello`, `cpmd-rattle-water`
- Multi-k analytic-force cases: `regressions/gpw/mbd`, `gpw-vdw-kpoint-forces`
- Multi-k AIMD family twin: `regressions/dynamics/xl-bomd`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative \(E_{\mathrm{cons}}\) drift with multi-k Bloch
CP engaged — a wrong \(C_k\) seed, real-only force path, or SCF on production
steps destroys the envelope immediately.

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | closed-shell RKS, \(n_\mathrm{occ}=1\) |
| XC | PADE (LDA) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF seed | `tol = 1e-5`, `max_iter = 60`, DIIS |
| CPMD | \(\mu = 400\) a.u.; production fixed-density |
| Ensemble | NVE |
| `dt` | 0.1 fs (nuclear = electronic) |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 cold SCF seed + 1 CP baseline + 4 production |

### Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`cpmd-water` / `xl-bomd-kpoints`. A CP2K twin sketch (short NVE Quickstep
with fictitious electronic DOF + `&KPOINTS`) lives in
`tools/cp2k_oracle/inputs/cpmd-kpoints/`. When a binary is available,
re-run offline and replace this section with version + measured drift +
fair wall.

Until then the regression asserts the **literature CP/NVE conservation
envelope** plus multi-k payload diagnostics (`n_kpts >= 2`), production
`iters == 0`, and soft H–H window.

### Observable and tolerance

Let \(E(t) = E_{\mathrm{KS}}(t) + T_n(t) + T_e(t)\) [eV], with \(E_0\) after
the first CP production step (Te engaged).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Production SCF **iters == 0** (fixed-density multi-k eval)
- `n_kpts >= 2` after seed
- PE finite and bound (\(< -5\) eV after warm-up)
- KE finite and \(\ge 0\), non-trivial after warm-up
- All coordinates finite
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step multi-k CP AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |


## Reference — multi-k UKS Car–Parrinello MD short trajectory (periodic H₂ NVE, GPW)

### Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on **multi-k collinear UKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces — the dual-spin Bloch orbital set \(C_k^\sigma\)
propagated per spin **and** per sampled k (the `cpmd-kpoints` ×
`cpmd-uks` crossing):

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 1`**, **`n_beta = 1`** (closed-shell fillings on the UKS
  path — both spin channels propagate and H₂ stays bound; triplet H₂ is
  repulsive and cannot pin NVE),
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  wrapped in `cpu::electronic::CarParrinello{gpw, μ}`; the electronic dt is
  pushed by `cpu::md::MD` from the `VelocityVerlet` integrator
  (`TimestepProvider` / `TimestepConsumer`) at build
- One `cpu::electronic::CpmdCplx` per **(k, spin)** propagates occupied
  Bloch coefficients with electronic RATTLE
  (\(C_k^{\sigma\dagger} S_k C_k^\sigma = I\)); production steps use the
  dual-spin fixed-density multi-k KS eval (`iters == 0`,
  `H_ks_k` + `H_ks_k_beta`), spin-diagonal EOM occupancy 1:
  \(\mu \ddot C_k^\sigma = -(I - S_k C_k^\sigma C_k^{\sigma\dagger})
  (2 H_k^\sigma C_k^\sigma) + \text{RATTLE}\)
- Conserved diagnostic:
  \(E_{\mathrm{cons}} = T_n + T_e + E_{\mathrm{KS}}\) with
  \(T_e = \sum_k w_k \sum_\sigma \tfrac12\mu\lVert\dot C_k^\sigma\rVert_F^2\)
- Periodic cubic cell; PADE (LSD) XC
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:**

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — electronic RATTLE
- Parent products: `regressions/dynamics/car-parrinello` (multi-k RKS Bloch CP),
  `regressions/dynamics/car-parrinello` (Γ dual-spin CP)
- Multi-k UKS BO surface: `regressions/gpw/hirshfeld`
  (multi-k UKS energy path)

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative \(E_{\mathrm{cons}}\) drift with dual-spin
multi-k Bloch CP engaged, plus a **UKS-vs-RKS closed-shell consistency
identity** — a wrong \(C_k^\sigma\) seed, a missing β channel, a real-only
force path, or SCF on production steps destroys one of them immediately.

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | collinear UKS, \(n_\alpha = n_\beta = 1\) (closed-shell fillings) |
| XC | PADE (LSD) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF seed | `tol = 1e-5`, `max_iter = 60`, DIIS |
| CPMD | \(\mu = 400\) a.u.; production fixed-density dual-spin |
| Ensemble | NVE |
| `dt` | 0.1 fs (nuclear = electronic) |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 cold SCF seed + 1 CP baseline + 4 production |

### Offline oracle

**Family policy (`cpmd-kpoints` / `cpmd-uks`, queue-assigned): literature
NVE conservation envelope.** The gate is a *method-family lock*, not a
missing binary: CP2K 2026.1 builds and runs on this host, but CP2K has **no fictitious-mass
Car–Parrinello propagator** — its `RUN_TYPE MD` is Born–Oppenheimer and its
CP-like method is second-generation CPMD (ASPC + corrector). The CP
conserved quantity \(T_n + T_e + E_{\mathrm{KS}}\) with the fictitious
\(T_e[\dot C_k^\sigma]\) is not a number CP2K can print for any input.

The multi-k BO surface is CP2K-anchored by `gpw-kpoints-si` (RKS), and this
regression additionally asserts the closed-shell identity below between the
multi-k UKS and **RKS** paths.

### Observable and tolerance

Let \(E(t) = E_{\mathrm{KS}}(t) + T_n(t) + T_e(t)\) [eV], with \(E_0\) after
the first CP production step (\(T_e\) engaged).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- **UKS-vs-RKS closed-shell consistency:** dedicated multi-k UKS
  (`n_alpha = n_beta = 1`) and multi-k RKS (`n_occ = 1`) single points
  (`cpu::single_point`, plain `GPWEvaluator`, same geometry/mesh/cutoff,
  `tol = 1e-7`) agree within **3×10⁻⁴ eV** ≈ 1.1×10⁻⁵ Ha (the UKS≡RKS
  closed-shell identity holds to 5×10⁻⁶ Ha at this tol; at the MD
  envelope `tol = 1e-5` the residual is convergence noise)
- Production SCF **iters == 0** (dual-spin fixed-density multi-k eval)
- `n_kpts >= 2` after seed
- \(T_e > 0\) once the electrons are propagated (dual-spin sum)
- PE finite and bound (\(< -5\) eV after warm-up); KE finite, \(\ge 0\),
  non-trivial after warm-up
- All coordinates finite
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | Oracle family lock: no engine on host implements fictitious-mass multi-k UKS CPMD (CP2K 2026.1 has no CP propagator — BOMD/SGCP only), so no fair same-method wall twin exists |
| `ratio_atv_over_oracle` | n/a |


## Reference — multi-k Car–Parrinello MD with electronic Nosé–Hoover thermostat (H₂ NVE, GPW)

### Method

**Car–Parrinello molecular dynamics (CPMD)** on **multi-k closed-shell RKS**
Gaussian-and-plane-waves (GPW / Quickstep) forces (Bloch \(C_k\)), with the
**Nosé–Hoover electronic thermostat engaged on every k channel** — the
capability this case pins is the lift of the old build-time refusal
"NVE electrons only on the k path" (`cpmd-thermostat-kpoints`):

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (nuclei NVE; only the
  fictitious electronic DOF are thermostatted)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  wrapped by `cpu::electronic::CarParrinello{gpw, μ,
  ElectronicThermostat{K₀, τ_e}}`; the electronic dt is pushed by the MD
  integrator (`TimestepProvider` / `TimestepConsumer`), never set here
- One `cpu::electronic::CpmdCplx` per sampled k, **each carrying its own NH
  variable targeting the full \(K_0\)** (the Γ-UKS convention); production
  steps are fixed-density multi-k KS evals (`iters == 0`)
- Published totals are k-weighted:
  \(T_e = \sum_k w_k T_e^{(k)}\),
  \(E_{\mathrm{therm}} = \sum_k w_k E_{\mathrm{therm}}^{(k)}\)
- Conserved diagnostic (Marx & Hutter extended energy):
  \(E_{\mathrm{ext}} = T_n + T_e + E_{\mathrm{KS}} + E_{\mathrm{therm}}\)

**Not BOMD** (production `iters == 0`). **Not nuclear NVT.**
**Not XL-BOMD / SGCP.**

### Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Blöchl & Parrinello, *Phys. Rev. B* **45**, 9413 (1992) — electronic
  Nosé–Hoover for adiabaticity control
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — extended conserved
  energy with electronic NH
- Twin family: Γ thermostat product `cpmd-electronic-thermostat`; multi-k
  NVE products `cpmd-kpoints` / `cpmd-uks-kpoints`
- Γ-limit **solver identity** (thermostatted `CpmdCplx` ≡ real `Cpmd` on
  real inputs, 1e-12) is unit-pinned in
  `tests/cpu/electronic/test_cpmd.cpp` (`ThermostatGammaLimitMatchesRealCpmd`)

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | closed-shell RKS, \(n_\mathrm{occ}=1\) |
| XC / basis / PP | PADE / DZV-GTH-PADE / GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF seed | `tol = 1e-5`, `max_iter = 60`, DIIS |
| CPMD | \(\mu = 400\) a.u.; production fixed-density |
| Thermostat | \(K_0 = 10^{-5}\) Ha, \(\tau_e = 50\) fs, per k channel |
| `dt` | 0.1 fs (nuclear = electronic) |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 cold SCF seed + 1 CP baseline + 4 production |

### Offline oracle (CP2K)

**Family lock (2026-07-31, same policy as `cpmd-electronic-thermostat` /
`cpmd-kpoints` / `cpmd-uks-kpoints`):** CP2K 2026.1 is present on the worker
host but has **no fictitious-mass Car–Parrinello propagator** — the CP
extended conserved quantity (and its electronic NH bookkeeping term) is not a
number CP2K can print, so no live or offline CP2K twin of this observable
exists. The BO surface underneath is CP2K-anchored by `gpw-kpoints-si`. An **unexecutable sketch** of the
intended twin lives in
`tools/cp2k_oracle/inputs/cpmd-thermostat-kpoints/h2_cpmd_thermostat_kpoints_nve.inp`.
The pin is therefore the **literature extended-energy / \(T_e\) control
envelope** below, never a golden snapshot of Atomiverse output.

### Observable and tolerance

Let \(E_{\mathrm{ext}}(t) = E_{\mathrm{KS}} + T_n + T_e + E_{\mathrm{therm}}\)
[eV], with \(E_0\) after the first CP production step (thermostat engaged).

\[
\delta = \max_t \frac{|E_{\mathrm{ext}}(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Production SCF **iters == 0** (fixed-density multi-k eval)
- `n_kpts >= 2` after seed
- \(T_e\) ceiling over production ≤ **0.05 eV** (thermostatted control —
  the fictitious electrons must not heat)
- \(E_{\mathrm{therm}}\) finite on every production step
- PE finite and bound (\(< -5\) eV after warm-up); KE finite, \(\ge 0\)
- All coordinates finite; soft H–H window \(0.4 < r_{\mathrm{HH}} < 1.5\) Å

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | Oracle family lock: CP2K 2026.1 has no fictitious-mass CP propagator — no fair wall twin of thermostatted multi-k CP AIMD exists (2026-07-31 inventory) |
| `ratio_atv_over_cp2k` | n/a |


## Reference — Car–Parrinello MD with electronic Nosé–Hoover thermostat (H₂O, GPW)

### Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on closed-shell Γ-point GPW, **electronic RATTLE**, and a
**Nosé–Hoover electronic thermostat** on \(\dot C\):

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with **`cpmd = true`**,
  **`cpmd_thermostat = true`**, **`cpmd_ekin_target_ha`**, **`cpmd_tau_e`**,
  **`cpmd_mu = 2000` a.u.**, `compute_forces = true`; the electronic dt is
  **not** set on the evaluator — `cpu::md::MD` pushes the nuclear dt in from
  the `VelocityVerlet` integrator through the `TimestepProvider` /
  `TimestepConsumer` protocol at build
- Propagator: `cpu::electronic::Cpmd` — VV + Andersen RATTLE + NH half-steps

**Not BOMD** (production `iters==0`). **Not nuclear NVT** as the pin (nuclei
NVE; only electronic DOF thermostatted). **Not XL-BOMD / SGCP**.

### Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Blöchl & Parrinello, *Phys. Rev. B* **45**, 9413 (1992) — separate electronic
  thermostats for adiabaticity
- Sprik, *J. Phys. Chem.* **95**, 2283 (1991)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — extended conserved
  energy with electronic NH:
  \(E_{\mathrm{ext}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}} + E_{\mathrm{therm}}\)
- Twin family: `cpmd-rattle-water`, `cpmd-water`

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. |
| \(K_0\) | \(1\times 10^{-5}\) Ha (cool electronic target) |
| \(\tau_e\) | 50 fs |
| Production | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

### Pin

| Observable | Gate |
| ---------- | ---- |
| \(E_{\mathrm{ext}}\) relative drift | ≤ **0.10** |
| Production SCF iters | **0** (fixed-density) |
| \(T_e\) ceiling over production | ≤ **0.05** eV |
| O–H distances | < 1.8 Å |
| PE after warm-up | < −100 eV |

**Offline CP2K twin input:** `tools/cp2k_oracle/inputs/cpmd_electronic_thermostat_h2o.inp`
(QS PBE / DZVP-MOLOPT-GTH / GTH-PBE, MT, 10 Å, short MD). Classical fictitious-mass
CPMD with electronic NH is not a stock CP2K QS ensemble twin — pin is a
**literature extended-energy / \(T_e\) envelope**, not a live CP2K energy twin in ctest.

**perf_exempt** — no CP2K classical-CPMD+e-T wall twin on the worker host;
literature envelope + structural gates (same inventory class as `cpmd-rattle-water`).


## Reference — Car–Parrinello MD with electronic RATTLE (H₂O NVE, GPW)

### Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on closed-shell Γ-point GPW, using **electronic RATTLE**
to enforce \(C^\top S C = I\) (position) and \(\dot\sigma = 0\) (velocity):

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with **`cpmd = true`**,
  **`cpmd_mu = 2000` a.u.**, `compute_forces = true`; the electronic dt is
  **not** set on the evaluator — `cpu::md::MD` pushes the nuclear dt in from
  the `VelocityVerlet` integrator through the `TimestepProvider` /
  `TimestepConsumer` protocol at build
- Propagator: `cpu::electronic::Cpmd` — velocity-Verlet + Andersen RATTLE on
  occupied MO coefficients \(C\) (supersedes Löwdin-only `cpmd-water`)

**Not BOMD** (production steps use fixed-density KS evaluation, `iters==0`).
**Not XL-BOMD**. **Not classical nuclear RATTLE** (`FixRattle`).

### Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- H. C. Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
  \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}}\)
- Twin geometry family: `cpmd-water`, `gpw-bomd-water`, `xl-bomd-water`

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. |
| Production | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

### Pin

| Observable | Gate |
| ---------- | ---- |
| \(E_{\mathrm{cons}}\) relative drift | ≤ **0.05** (tighter than Löwdin foothold 0.10) |
| Production SCF iters | **0** (fixed-density) |
| O–H distances | < 1.8 Å |
| PE after warm-up | < −100 eV |

**Offline CP2K twin input:** `tools/cp2k_oracle/inputs/cpmd_rattle_water_nve.inp`
(QS PBE / DZVP-MOLOPT-GTH / GTH-PBE, MT, 10 Å, short NVE). Pin is a
**literature conservation envelope**, not a live CP2K energy twin in ctest.

**perf_exempt** — no CP2K CPMD+RATTLE wall twin on the worker host; literature
envelope + structural gates (same inventory class as `cpmd-water`).

### Honest scope

- Closed-shell RKS Γ only; UKS / OT / k-points refused at `build()`
- Electronic RATTLE (pos + vel); geometry transport still uses Löwdin + vel project
- No electronic thermostat
