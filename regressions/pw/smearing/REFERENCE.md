# pw/smearing

One case; every section below is run by `atv_regressions --case pw/smearing` and keeps its own oracle record.


## Reference — bulk Fe collinear UKS + metallic smearing (plane-wave)

### Method

Collinear spin-polarized plane-wave KS (`cpu::pw::PW` with
`spin_polarized = true`) **together with** metallic Fermi–Dirac smearing
(`smearing = fermi_dirac`, `degauss` [Ry], `extra_bands`). One shared chemical
potential \(\mu\) over spin-up and spin-down spin-orbitals; occupations
\(f_{\sigma kb}\in[0,1]\); magnetization \(M = N_\uparrow - N_\downarrow\) is
**self-consistent** (multiplicity only seeds the initial \(\rho_\uparrow/
\rho_\downarrow\) split). Total energy is the Mermin free energy
\(F = E_{\mathrm{KS}} - TS\), reported on `PWResult::energy`; the \(-TS\)
contribution is reported separately on `PWResult::e_entropy`.

Closes the product gap left by:

| Foothold | What it lacked |
| -------- | -------------- |
| restricted metallic smearing (FCC Al) | spin |
| `pw-spin-polarized-o2` / fixed-occupation k-mesh UKS Fe | metallic smearing |

### System

| Quantity | Value |
| -------- | ----- |
| Geometry | BCC Fe primitive cell, \(a_{\mathrm{cubic}} = 2.87\) Å, 1 atom at origin |
| PP | SG15 ONCV PBE Fe `tests/data/pw/Fe_ONCV_PBE-1.2.upf` (\(Z_{\mathrm{val}}=16\)) |
| XC | PBE (true collinear spin-GGA) |
| \(E_{\mathrm{cut}}\) | 12 Ry (algorithm pin, **not** a production cutoff) |
| k-mesh | Monkhorst–Pack \(2\times2\times2\) |
| Smearing | Fermi–Dirac at \(\sigma = 0.02\) Ry **and** \(\sigma = 0.005\) Ry |
| Extra bands | 4 per spin channel |
| Multiplicity seed | 5 (initial \(M\approx 4\); free under smear) |
| SCF | `tol=2×10⁻³`, `max_iter=30`, `mix_beta=0.3` |

### Gates — all exact, none frozen

The case runs the same system twice, at \(\sigma\) and \(\sigma/4\).

#### 1. Electron count (exact)

\[
N_{\mathrm{occ}} = \sum_{\sigma k b} w_k f_{\sigma kb} = Z_{\mathrm{val}} = 16
\quad (|\Delta N| \le 10^{-4}),\ \text{at both } \sigma .
\]

Gates k-weight normalisation \(\sum_k w_k = 1\) **and** the requirement that
ONE \(\mu\) is shared by both spin channels — two independent \(\mu_\sigma\)
would still hit 16 but only by construction, so this is paired with gate 4.

#### 2. Entropy sign (exact)

\(S \ge 0\), hence the reported \(-TS\) term is never positive.

#### 3. \(\partial F/\partial T = -S\) (exact)

Widening the smearing must **lower** the Mermin free energy:

\[
F(\sigma{=}0.02) - F(\sigma{=}0.005) < -10^{-2}\ \mathrm{Ry}.
\]

Measured: \(-5.13\times10^{-2}\) Ry, against an SCF wobble of \(\approx
4\times10^{-3}\) Ry at this tolerance. A build that reported the internal
energy \(E\) instead of \(F\), or that got the sign of the entropy term
wrong, fails here. So does the pre-2026-07-29 solver (see below):
it produced \(-1.2\times10^{-3}\) Ry.

#### 4. \(TS \to 0\) as \(T \to 0\) (exact)

\(|{-}TS|\) must shrink with \(\sigma\): measured \(3.71\times10^{-2}\ \to\
5.19\times10^{-3}\) Ry.

#### 5. Ferromagnetic ground state

\(M > 0.1\) at both \(\sigma\). Measured \(M = 0.799\) (\(\sigma=0.02\)) and
\(M = 0.999\) (\(\sigma=0.005\)); tightly converged (`tol=1e-6`) the small-σ
limit is \(M \to 1.00\). This 12 Ry / 8-k setup undershoots experimental BCC
Fe (\(\approx 2.2\,\mu_B\), Kittel) by design — it is an algorithm pin, so the
gate is the ferro sign and its stability under smearing, not the magnitude.

### Root-cause fix 2026-07-29 — spin ↓ was solved with the wrong \(V_{\rm eff}\)

`PW::run`'s collinear+smearing branch rebound `ws.gvec` / `ws.n_g` to each
k-point's **wavefunction** G-sphere inside the k loop and never restored them
before the next spin channel. Spin ↑ (first iteration of the spin loop) got a
correct \(V_{xc,\uparrow}\) copy and `VeffBuilder::step` on the **density**
sphere; spin ↓ then copied only the first \(n_g(k_{\rm last})\) coefficients
of \(V_{xc,\downarrow}\) and FFT-ed \(V_{\rm eff}\) using the last k-point's
\((h,k,l)\) indices — i.e. **the two spin channels were solved with different
Hamiltonians**.

The identical defect had already been found and repaired in the sibling helper
`internal::run_scf_smear_collinear` (`restricted_scf.cpp`, commit `ea3ae031`,
2026-07-26, comment *"restore after each k so the next spin channel does not
inherit a k-sphere n_g"*), which is why `bands-spin-metallic-si` and
`dos-spin-metallic-si` were unaffected. It was never back-ported to
`PW::run`.

**Exact invariant that catches it** (`tests/cpu/planewave/test_pw_spin.cpp`,
`PWSpin.CollinearSmearNonMagneticSeedKeepsSpinSymmetry` and
`…MultiKSharedMuConservesElectrons`): with `multiplicity = 1` the two channels
are seeded identically and every operator in the loop is spin-symmetric, so
\(M \equiv 0\) with **no** tolerance. Before the fix a non-magnetic cell
produced \(M = 1.90\) (Γ) and \(M = -0.25\) (2×2×2) — spontaneous, unphysical
symmetry breaking. After the fix, exactly `0.0`.

Effect on this case: \(F(\sigma{=}0.02)\) moved from \(-205.89959\) Ry
(broken) to \(-205.25820\) Ry (repaired), i.e. **0.641 Ry = 8.7 eV**.

### Void pin — what the old number actually was

This case previously gated `F = −205.8365363098 Ry ± 0.05`, described as an
"offline ATV freeze" recorded 2026-07-25. **That number is void**: it was
recorded on a solver carrying *both* defects above.

| Quantity | \(F\) [Ry] | \(M\) | \(E_F\) [Ry] |
| -------- | ---------- | ----- | ------------ |
| Fossil pin (2026-07-25) | −205.8365363098 | 0.991 (recorded) | ≈2.16 (recorded) |
| broken solver, `xc = LSDA` | −205.84536156 | **0.991644** | **2.157846** |
| broken solver, `xc = PBE` | −205.89958574 | 1.083737 | 2.194362 |
| repaired solver, `xc = PBE` | −205.25820008 | 0.799445 | 2.095300 |

The fossil pin's own recorded \(M = 0.991\) and \(E_F \approx 2.16\) match the
**LSDA-on-broken-solver** row to 3–4 digits and match nothing else. So the pin
was taken before *both* `c6a6273c` (true collinear spin-GGA; before it, a PBE
request silently ran the σ = 0 LDA limit) and the \(V_{\rm eff}\) repair
above.

**Why the pin was deleted rather than moved.** BCC Fe at \(E_{\rm cut}\) = 12
Ry with an 8-point mesh has no citable external absolute energy — SG15 ONCV Fe
needs \(\approx\)80 Ry, and no literature or analytic value exists for this
setting. Nor is a σ→0 extrapolation available as an anchor: with 8 k-points
the spectrum is discrete, so \(TS \propto \sigma^{1.3}\) rather than the
Sommerfeld \(\sigma^2\) of a smooth DOS, and the σ scan crosses a magnetic
transition (tightly converged: \(M = 0.026,\ 0.807,\ 0.995,\ 0.999\) at
\(\sigma = 0.04,\ 0.02,\ 0.01,\ 0.005\) Ry). Re-freezing ATV's current output
would validate nothing. The gates above are exact thermodynamic and
counting statements instead.

### Offline CP2K twin (algorithm / documentation)

**Not an absolute-energy twin** (GTH-GPW ≠ NC-UPF PW). Documents that CP2K
UKS + `&SMEAR METHOD FERMI_DIRAC` is the product counterpart.

| Item | Value |
| ---- | ----- |
| Note | `tools/cp2k_oracle/inputs/pw-uks-fe-smearing.note` |
| Host CP2K | not loadable as a fair NC-UPF PW twin |
| Method class | Quickstep UKS + FERMI_DIRAC + Fe GTH (if re-run offline) |

No live CP2K at ctest.

### Perf

| Field | Value |
| ----- | ----- |
| perf_exempt | true |
| reason | no fair single-rank NC-UPF PW twin wall vs CP2K GTH-GPW |
| T_atv (this host, OMP=1) | **~34 s** (two solves) |
| ratio T_atv/T_cp2k | n/a |

### Dual gate

1. Correctness: converge at both σ + \(N_{\rm occ}=16\) + \(S\ge0\) +
   \(\partial F/\partial T=-S\) + \(TS\to0\) + ferro \(M>0.1\) + finite \(E_F\).
2. Perf: exempt (no fair CP2K NC-UPF twin).


## cuda-pw-metallic-smear

### What

Dual-gate restricted Fermi–Dirac metallic smear (FCC Al primitive cell) on
`cuda::pw::PW`, against the same-settings CPU `cpu::pw::PW` smear path.

### Oracle

1. **Electron count** — \(N_{\mathrm{occ}} = 3\) (Al \(Z_{\mathrm{val}}\)).
2. **Same-settings CPU twin** — `cpu::pw::PW` at identical members is the
   correctness oracle for CUDA free energy (ES CUDA open-gate law).

Not a freeze of CUDA’s own energy. Not forces/stress/spin+smear.

### Settings

| Member | Value |
| --- | --- |
| PP | `tests/data/pw/Al_pz_nc.upf` |
| ecut | 20 Ry |
| k_mesh | 2 (MP \(2\times2\times2\)) |
| cell | FCC primitive \(a = 4.05\) Å |
| smearing | Fermi–Dirac |
| degauss | 0.02 Ry |
| extra_bands | 4 |
| mix_beta | 0.4 |
| tol | 1e-5 |

### Tolerances

| Quantity | Tol |
| --- | --- |
| \(N_{\mathrm{occ}}\) | 1e-5 |
| \|E_cuda − E_cpu\| | 5e-3 Ry |

Non-bit-exact: cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm,
Davidson trajectory.


## cuda-pw-smear-forces

### What

Dual-gate restricted Fermi–Dirac metallic-smear **analytic ionic forces** on
`cuda::pw::PW` vs same-settings `cpu::pw::PW`. Geometry is a stretched Al
dimer in a cubic box (relative bond stretch so |F| is non-vacuous; a monatomic
1-atom cell has F≡0 by lattice gauge).

### Oracle

1. **Same-settings CPU twin** — `cpu::pw::pw_forces_analytic` under identical
   members (restricted FD smear, fractional occupations). Not a self-golden of
   CUDA.
2. **Non-vacuous** — max|F_cpu| > 0.05 eV/Å.

Not spin∧smear, not stress, not smear×PBE force depth.

### Settings

| Member | Value |
| --- | --- |
| PP | `tests/data/pw/Al_pz_nc.upf` |
| ecut | 16 Ry |
| k_mesh | 1 (Γ) |
| cell | cubic \(a = 6.5\) Å |
| geometry | Al₂ bond \(= 2.55\) Å along z |
| smearing | Fermi–Dirac |
| degauss | 0.02 Ry |
| extra_bands | 4 |
| mix_beta | 0.4 |
| tol | 1e-5 |

### Tolerances

| Quantity | Tol |
| --- | --- |
| max\|F_cuda − F_cpu\| | 5e-3 eV/Å |

Non-bit-exact: cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm,
Davidson trajectory.

### Perf

**perf_exempt** — small-cell correctness dual-gate only (no CP2K wall claim).

### Measured (GH200)

| Job | max\|ΔF\| |
| --- | ---: |
| **1130427** (rebased wave-9) | **1.36e-11** eV/Å |
| 1130401 (pre-rebase) | 1.36e-11 eV/Å |


## cuda-pw-smear-stress

### What

Dual-gate restricted Fermi–Dirac metallic-smear **Nielsen–Martin cell stress**
on `cuda::pw::PW` vs same-settings `cpu::pw::PW`. Geometry is compressed FCC
Al (1 atom primitive) so hydrostatic |σ| is non-vacuous and valence is odd
(requires fractional occupations).

The free energy surface is the **Mermin free energy**
\(F = E_{\mathrm{KS}} - TS\); stress is
\(\sigma = -(1/\Omega)\,\partial F/\partial\varepsilon\).

### Oracle

1. **Same-settings CPU twin** — `cpu::pw::pw_stress` under identical members
   (restricted FD smear, fractional occupations, Mermin F). Not a self-golden
   of CUDA.
2. **Non-vacuous** — mean|diag_cpu| > 1e-3 eV/Å³.
3. **Cubic symmetry** — |σ_xx−σ_yy|, |σ_yy−σ_zz|, |σ_off| within soft bound.

Not spin∧smear, not multi-k smear stress, not smear×PBE stress.

### Settings

| Member | Value |
| --- | --- |
| PP | `tests/data/pw/Al_pz_nc.upf` |
| ecut | 16 Ry |
| k_mesh | 1 (Γ) |
| cell | FCC primitive \(a = 3.90\) Å (compressed) |
| XC | LSDA |
| smearing | Fermi–Dirac |
| degauss | 0.02 Ry |
| extra_bands | 4 |
| mix_beta | 0.4 |
| tol | 1e-5 |

### Tolerances

| Quantity | Tol |
| --- | --- |
| max\|σ_cuda − σ_cpu\| | 5e-2 eV/Å³ |

Non-bit-exact: cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm,
Davidson trajectory.

### Perf

**perf_exempt** — small-cell correctness dual-gate only (no CP2K wall claim).

### Measured (GH200)

| Job | max\|Δσ\| | notes |
| --- | ---: | --- |
| **1133111** | **7.252e-11** eV/Å³ | re-verify after rebase onto GapCensus wave 13 @ 85c6c5eb; dual-gate both; mean\|diag_cpu\|=3.493; \|ΔE\|=1.63e-13 Ry; units green |
| 1132266 | 7.252e-11 eV/Å³ | prior tip-green pre-rebase |
| 1131116 | — | exit 127 (job-script wipe mid-run); not a product fail |
| 1130720 | — | unit suite timeout (pre unit-gate trim); not a product fail |
