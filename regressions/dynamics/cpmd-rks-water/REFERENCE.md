# Reference — AO Car–Parrinello MD (H₂O NVE, RKS/LSDA/STO-3G)

## Method

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

## Literature

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

## Setup

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

## Tuning — μ × dt scan (worst relative \(E_{\mathrm{cons}}\) drift, 40 steps)

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

## Pin

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

## Honest scope

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
