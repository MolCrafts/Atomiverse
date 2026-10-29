# pw/rks

One case; every section below is run by `atv_regressions --case pw/rks` and keeps its own oracle record.

**2026-10-28 model audit:** the archived 2026-08-09 fair-wall records for
`pw-pbe-gga-si` and `pw-analytic-stress-si` use QE `input_dft='SLA PW'`
(Slater/PW92), whereas current `XCType::LSDA` uses Slater/VWN5. Those timings
and decks are retained as historical records; their exact-model twin claims
do not qualify current VWN5 science or matched-model performance. The original
Si QE printed-XC outputs were not located, so they cannot be relabeled VWN.
The current H3 LSDA FD/Gaussian/MP references independently use explicit
`SLA+VWN+NOGX+NOGC` in both their raw inputs and printed QE output and are
unaffected. Native activation, symmetry and full-SCF finite-difference gates
remain distinct from an absolute external LSDA comparison.
[Audit and raw-reference hashes](../../../tools/qe_oracle/inputs/pw-analytic-stress-si/model-audit-2026-10-28.json).


## Reference — plane-wave LDA bulk silicon (k-mesh energy)

### Method

Spin-unpolarized plane-wave Kohn–Sham DFT with LDA exchange–correlation
(Slater exchange + VWN5 correlation on the PW real-space grid). Norm-conserving
ONCVPSP pseudopotential (UPF). Brillouin-zone sampling is a uniform
Monkhorst–Pack mesh (`PW::k_mesh`³). Total energy is reported in Rydberg on
`PWResult::energy` and in eV on `frame.meta.get(kEnergy)`.

Public path: `cpu::single_point(frame, method)` with a concrete `cpu::pw::PW`;
outer workflows mount `PW` directly on the parent Driver.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013); M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Upstream | <https://github.com/PseudoDojo/ONCVPSP-LDA-SR-noCCv0.4> (`Si/Si.upf`) |

### ABINIT-class same-pseudo pin (converged limit)

PseudoDojo ships an ABINIT deltafactor EOS for **this** ONCV LDA Si
pseudopotential in `Si/Si.djrepo` (retrieved 2026-07-23):

| Quantity | Value |
| -------- | ----- |
| Source | `Si.djrepo` deltafactor key `"8.0"`, `num_sites = 2` |
| \(E_\mathrm{min}\) | **−230.158202 eV/cell** = **−16.916316 Ry/cell** |
| \(v_0\) | 20.43445 Å³/atom → \(a_0 \approx 5.468\) Å (conventional cubic) |
| \(B_0\) | 89.96 GPa |

This is the **converged** ABINIT-class absolute energy for the same PP at the
EOS minimum. Absolute PW energies shift with \(E_\mathrm{cut}\) and k-mesh;
the regression twin below uses tutorial-class settings for a ≤ 60 s e2e case.

### Twin geometry and settings (regression)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| k-mesh | \(2\times2\times2\) Monkhorst–Pack (`k_mesh = 2`) |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 50 |

### Acceptance band (incomplete basis)

Because the twin is deliberately under-converged in basis and BZ sampling,
\(E_\mathrm{ATV}\) is **higher** (less bound) than the PseudoDojo ABINIT
\(E_\mathrm{min}\). The regression asserts the variational incomplete-basis
window:

\[
E_\mathrm{min}^\mathrm{ABINIT}
  \le E_\mathrm{ATV}
  \le E_\mathrm{min}^\mathrm{ABINIT} + 1.50~\mathrm{Ry}
\]

with \(E_\mathrm{min}^\mathrm{ABINIT} = -16.916316\) Ry. This is **not** a
freeze of Atomiverse’s own output: the lower edge is the external ABINIT-class
pin; the 1.50 Ry headroom covers incomplete ecut/k at \(a=5.43\) Å while
still rejecting unit disasters (0, positive, free-atom-scale errors of many Ry).

Measured on this host (2026-07-23, Release, 16 OpenMP threads): twin converges
in 5 iterations to \(E \approx -15.670898\) Ry (~1.25 Ry above \(E_\mathrm{min}\)),
wall ≈ 7–8 s — inside the band and well under 60 s.

Frame energy must match \(E_\mathrm{Ry} \times 13.605693\) eV within \(10^{-4}\) eV
(engine conversion constant in `qe_units.h`).

### CP2K note (absolute energy)

CP2K Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH** (different PP and
discretization). Absolute total energies are **not** comparable to ONCV
plane-wave; EOS (\(a_0\), \(B_0\)) is the fair CP2K science comparison (see
historical `benchmarks/pw` docs). An optional twin input skeleton lives under
`tools/cp2k_oracle/inputs/si_lda_pw_note.inp` for offline wall timing only.

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K (or ABINIT) binary / module on the build host; fair wall twin not run offline this shift |
| ATV twin wall (informative) | ≈ 8 s (Release, 16 threads, ecut 15 / k=2) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time the same geometry with the oracle input,
append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) here, require \(r \le 10\).


## Reference — plane-wave true PBE GGA on bulk Si

### Method

Spin-unpolarized plane-wave Kohn–Sham total energy and ionic forces with the
**true GGA PBE** exchange–correlation functional (Perdew, Burke, Ernzerhof,
*Phys. Rev. Lett.* **77**, 3865 (1996)). Density gradients enter the XC energy
and potential spectrally on the FFT grid (not the historical `σ = 0`
LDA-limit call of `evaluate_xc(PBE, …)`).

Public API: concrete
`cpu::pw::PW{UpfPseudopotential{path}, ecut_ry, k_mesh, XCType::PBE}` (knobs
as members); Frame I/O; `pw_forces_analytic` for forces
[eV/Å]. No public `*Config` bag.

### System

Silicon diamond (FCC primitive cell, 2 atoms):

| | |
| --- | --- |
| Lattice constant | \(a = 5.43\) Å (standard experimental Si) |
| Basis | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Pseudopotential | in-tree ONCVPSP NC UPF `tests/data/pw/Si_lda_nc.upf` (scalar-relativistic, LDA-generated; used here only as a fixed NC valence PP for the GGA path smoke) |
| Cutoff | \(E_{\mathrm{cut}} = 15\) Ry |
| k-mesh | \(2\times 2\times 2\) Monkhorst–Pack |
| Mixing | Pulay, `mix_beta = 0.5`, `tol = 1e-5`, `max_iter = 40` |

### Offline oracle status

**No CP2K or Quantum ESPRESSO binary** was available on the worker host at pin
time (`module avail cp2k` / `qe` empty; no `pw.x` / `cp2k.psmp` on `PATH`).
Absolute total-energy digits are therefore **not** invented from memory and
**not** taken from a live third-party process at ctest time.

Pin strategy (binding for this regression):

1. **Crystal-symmetry force pin (literature physics).** On ideal diamond
   sites the Hellmann–Feynman force on every atom vanishes by lattice
   symmetry. Residual after a modest SCF is bounded by
   \[
   |F_i| < 0.5~\mathrm{eV/Å}
   \]
   for both Si atoms (same bound style as `regressions/pw/rks`).

2. **GGA activation pin.** The same geometry is also run with
   `xc = XCType::LSDA`. True PBE GGA must shift the total energy by at least
   \[
   |E_{\mathrm{PBE}} - E_{\mathrm{LSDA}}| > 10^{-4}~\mathrm{Ry}
   \]
   (documents that the production path is not stuck in a pure density-only
   LDA limit of a single functional call). Unit tests separately prove
   PBE@\(σ=0\) vs true GGA on a nonuniform density.

3. **Energy finiteness / convergence.** \(E_{\mathrm{PBE}}\) is finite;
   preferred: SCF converges within `max_iter`.

A QE twin is now installed offline (see the fair wall below); its total
energies at these exact settings are recorded there as provenance output
lines. The ctest gates above are unchanged — promoting the QE energies to
correctness pins is separate queue work, not part of the fair-wall repin.

#### Observed Atomiverse numbers (informational, not a self-golden pin)

Worker run 2026-07-23 (`wf/pw-pbe-gga-si`, Release, single rank):

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{PBE}}\) | −15.70481672 Ry (converged, 6 iters) |
| \(E_{\mathrm{LSDA}}\) | −15.67175235 Ry (converged, 5 iters) |
| \(\|E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}\|\) | 0.0330644 Ry (> 1e−4 floor) |
| \(\|F_0\|\), \(\|F_1\|\) | ~0.0015 eV/Å (< 0.5 eV/Å) |
| Wall (PBE+forces+LSDA) | ~27 s |

### Performance gate — measured fair wall (oracle: qe)

**Measured 2026-08-09** against the offline Quantum ESPRESSO twin
(`Program PWSCF v.7.6`, source build of 2026-08-09,
`/nobackup/proj/disk/teoroo/personal/jicli594/work/qe/install/bin/pw.x`;
install record `tools/qe_oracle/pin_notes.md`). Twin job = the exact
regression workload: PBE SCF + forces, then LSDA SCF, same NC UPF
(`tests/data/pw/Si_lda_nc.upf`), cell, \(E_\mathrm{cut}=15\) Ry, MP
\(2\times2\times2\). Inputs: `tools/qe_oracle/inputs/pw-pbe-gga-si/`
(`si_pbe.in` with `tprnfor`, `si_lsda.in` with `input_dft='SLA PW'`).

| Item | Value |
| ---- | ----- |
| \(T_{\mathrm{atv}}\) | **10.94 s** (`reg_pw-pbe-gga-si`, median of 3) |
| \(T_{\mathrm{qe}}\) | **0.69 s** (0.35 s PBE+forces + 0.34 s LSDA, QE-printed WALL, median of 3) |
| \(T_{\mathrm{atv}}/T_{\mathrm{qe}}\) | **15.9** |
| within_10x | **false — open PW perf debt** (was the stated target when exempt) |

Protocol: Arrhenius x86_64 epyc login node (48-core, shared; load avg 7–24
during the window), one shell, 3 interleaved repetitions, medians; both sides
single process, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` (the regression's
declared thread shape), QE as `mpirun -np 1 pw.x`. ATV binary: Release,
`-march=native`, this commit.

QE key output lines (provenance, not ctest pins):

```text
si_pbe.out : !    total energy              =     -15.70442105 Ry   (6 iters)
si_pbe.out :      Total force =     0.000000
si_lsda.out: !    total energy              =     -15.67051079 Ry   (7 iters)
```

Known cost asymmetry, recorded not excused: QE reduces the unshifted
\(2\times2\times2\) mesh to 3 IBZ k-points by symmetry; ATV diagonalizes all
8. That accounts for ≲ 2.7× of the 15.9× — the rest is ATV solver cost.

### What the binary checks

`reg_pw-pbe-gga-si`:

1. `PW{xc=PBE}` SCF on Si diamond → finite energy [Ry].
2. `pw_forces_analytic` → \(|F_i| < 0.5\) eV/Å for both atoms.
3. Second run `xc=LSDA` → finite; \(|E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}| > 1e-4\) Ry.
4. No external process spawn.

### Provenance

- Spec: `.claude/specs/pw-pbe-gga-si/spec.md`
- Functional: PBE (1996); shared `cpu::scf::evaluate_xc` (libxc-parity tree)
- Geometry: textbook Si diamond \(a=5.43\) Å
- PP: Hamann ONCVPSP NC UPF shipped under `tests/data/pw/`


## Reference — plane-wave analytic ionic forces (GTH-PBE-q1)

### Method

Hellmann–Feynman local + Ewald ionic forces after one plane-wave SCF
(`pw_forces_analytic`). Plane waves are cell-fixed. See Martin, *Electronic
Structure* (CUP), and Quantum ESPRESSO `force_lc.f90` / `force_ew.f90` for the
production form.

**Coverage limit — read before trusting this case.** The nonlocal KB term is
zero here, because `GTH-PBE-q1` for hydrogen is purely local: it carries a
local channel only (`kPot_H_GTH_PBE_q1` declares zero projector channels in
`include/atomiverse/cpu/gpw/basis_data/gpw_tables.h`). So this case exercises
the local + Ewald force terms and **cannot** detect a dropped or wrong KB
projector force. That coverage lives in the Si/GTH gate in
`tests/cpu/planewave/test_pw_forces_fd.cpp`, which must be kept green
alongside this one.

FD of the total energy (`pw_forces_fd`) remains the numerical oracle; this
regression pins the **symmetry** result that both paths must satisfy.

### System

One hydrogen atom (Z=1) at the origin of a cubic cell \(L=6\) Å. Settings:
\(E_\mathrm{cut}=5\) Ry, Γ-only k-mesh, `max_iter=8`, `GthPseudopotential{"GTH-PBE-q1"}`
(analytic Goedecker–Teter–Hutter pseudopotential). The bare-Coulomb "toy" local
potential this case originally used has been deleted from the tree.

**Open shell, not closed shell.** One valence electron is a doublet
(\(S=1/2\)), so the run is collinear spin-polarized with `spin_polarized = true`
and `multiplicity = 2` (\(N_\uparrow=1\), \(N_\downarrow=0\)). The restricted,
fixed-occupation path rejects an odd electron count outright —
`internal::require_closed_shell` (`src/cpu/planewave/internal/pw_setup.cpp`),
the one gate `PW::run`, `pw_forces_analytic` and `pw_stress` all call. The
default `multiplicity = 1` fails a *different* guard, the collinear path's
parity check: \(N_e + (M-1)\) must be even, which 1 + 0 is not.

### Pin

By lattice symmetry the ionic force on a single atom at a high-symmetry site is
**exactly zero**. This is a symmetry identity and does *not* depend on the SCF
reaching `cfg.tol` — the 8-iteration run deliberately stops short — so the
bound is a machine-precision one:

\[
|F| < 10^{-9}~\mathrm{eV/Å}.
\]

Measured 2026-08-15 with GTH-PBE-q1, spin-polarized doublet:
\(|F| = 6.58\times10^{-16}\) on both the analytic and the evaluator path
(\(E = -12.0277\) eV; SCF converged in 5 iterations, \(m = 1\)).

The public nested `PW` force request must write a finite energy and a force
vector satisfying the same bound (analytic path).

### Shared-state request coverage

The regression also checks combined force/stress and a changed-cell stress-only
request against the owning method's `last_result()`. Restricted Fermi–Dirac
smearing and collinear spin plus smearing reuse the single-H fixture, with two
extra bands, width 0.02 Ry, SCF tolerance 1e-8 and a 100-iteration cap. Their
forces are compared with the existing `pw_forces_fd` oracle at 5e-3 Å using the
unchanged h_force_fd bound (1e-4 eV/Å), alongside the analytic symmetry bound
(1e-9 eV/Å). This does not test nonzero bond forces or nonlocal projectors.

Restricted smearing additionally compares stress with `pw_stress_fd` at strain
2e-3. The 0.05 eV/Å³ dual-gate bound is inherited from pw-analytic-stress-si;
the separate off-diagonal symmetry bound is 1e-9 eV/Å³. This broad diagonal
bound is a smoke gate, not a precision stress benchmark. Spin plus smearing
stress remains unsupported and is not counted as coverage. New matrix cases
must be executed after compilation; no new measured reference values are claimed.


## Reference — plane-wave FD ionic forces (GTH-PBE-q1)

### Method

Central finite differences of the plane-wave total energy:

\[
F_{a\alpha} = -\frac{E(R_{a\alpha}+h)-E(R_{a\alpha}-h)}{2h}
\]

with \(E\) converted to eV and \(h\) in Å (`pw_forces_fd`). Same oracle form as
GPW's `gpw_forces_fd` before analytic forces. See Martin, *Electronic Structure*
(CUP), Hellmann–Feynman chapter for the physical force; FD is the standard
numerical check.

### System

One hydrogen atom (Z=1) at the origin of a cubic cell \(L=6\) Å. Settings:
\(E_\mathrm{cut}=5\) Ry, Γ-only k-mesh, `max_iter=8`, `GthPseudopotential{"GTH-PBE-q1"}`
(analytic Goedecker–Teter–Hutter pseudopotential). The bare-Coulomb "toy"
local potential this case originally used has been deleted from the tree.

**Open shell, not closed shell.** One valence electron is a doublet
(\(S=1/2\)), so the run is collinear spin-polarized with `spin_polarized = true`
and `multiplicity = 2` (\(N_\uparrow=1\), \(N_\downarrow=0\)). The restricted,
fixed-occupation path rejects an odd electron count outright —
`internal::require_closed_shell` (`src/cpu/planewave/internal/pw_setup.cpp`),
called by `PW::run` (which is what the \(6N\) FD solves go through) and by
`pw_forces_analytic` (the evaluator half of this case) alike. The default
`multiplicity = 1` fails a *different* guard, the collinear path's parity
check: \(N_e + (M-1)\) must be even, which 1 + 0 is not.

### Pin

By lattice symmetry the ionic force on a single atom at a high-symmetry site is
**exactly zero**. This is a symmetry identity, so it does *not* depend on the
SCF reaching `cfg.tol` — the 8-iteration run deliberately stops short. Only
numerical noise survives, and the two paths have different floors:

\[
|F|_\mathrm{FD} < 10^{-4}~\mathrm{eV/Å},
\qquad
|F|_\mathrm{evaluator} < 10^{-9}~\mathrm{eV/Å}.
\]

The FD bound is set by the \(h=5\times10^{-3}\) Å step and the 1e-4 SCF
tolerance, not by machine epsilon; the evaluator returns the analytic gradient.
Measured 2026-08-15 with GTH-PBE-q1, spin-polarized doublet:
\(|F|_\mathrm{FD}=1.4\times10^{-11}\),
\(|F|_\mathrm{evaluator}=6.58\times10^{-16}\) (\(E=-12.0277\) eV).

The public nested `PW` force request must write a finite energy and a force
vector satisfying the evaluator bound.


## Reference — plane-wave cell stress (cubic single H)

### Method

Cell stress is obtained from central finite differences of the plane-wave SCF
energy under symmetric strain at fixed fractional coordinates (`pw_stress`).
This is the same oracle class used for early ionic FD forces. Nielsen–Martin
analytic decomposition is a follow-on.

### System

One H at the origin of a cubic cell \(L=6\) Å. `GthPseudopotential{"GTH-PBE-q1"}` (analytic Goedecker–Teter–Hutter
pseudopotential; the bare-Coulomb "toy" potential this case originally used has
been deleted from the tree), \(E_\mathrm{cut}=5\) Ry,
Γ-only, short SCF.

**Open shell, not closed shell.** One valence electron is a doublet
(\(S=1/2\)), so the run is collinear spin-polarized with `spin_polarized = true`
and `multiplicity = 2` (\(N_\uparrow=1\), \(N_\downarrow=0\)). The restricted,
fixed-occupation path rejects an odd electron count outright —
`internal::require_closed_shell` (`src/cpu/planewave/internal/pw_setup.cpp`),
which `pw_stress` calls exactly as `PW::run` and `pw_forces_analytic` do. The
default `multiplicity = 1` fails a *different* guard, the collinear path's
parity check: \(N_e + (M-1)\) must be even, which 1 + 0 is not.

### Pin

By cubic symmetry, off-diagonal Voigt components (xy, xz, yz) vanish within
noise:

\[
|\sigma_{xy}|,|\sigma_{xz}|,|\sigma_{yz}| < 10^{-9}~\mathrm{eV/Å^3}.
\]

Diagonal components are finite and mutually equal within

\[
\max_{i\neq j}|\sigma_{ii}-\sigma_{jj}| < 10^{-6}~\mathrm{eV/Å^3}.
\]

Both are symmetry identities and do *not* depend on the SCF reaching
`cfg.tol`, but they have different numerical floors and so carry different
bounds. The off-diagonals are structurally zero (machine precision), whereas
each *diagonal* component comes from an independent central-difference
evaluation inside `pw_stress`, so its floor is set by the FD step and the
1e-4 / 8-iteration SCF.

Measured 2026-08-15 with GTH-PBE-q1, spin-polarized doublet: off-diagonals
\(\le 1.049\times10^{-14}\); \(\sigma_{xx}=-0.0050847238\),
\(\sigma_{yy}=-0.005084722912\), \(\sigma_{zz}=-0.005084722795\), i.e. spread
\(1.006\times10^{-9}\) against a signal of \(0.005085\)
(\(2.0\times10^{-7}\) relative). Identical to the printed 10 significant
digits at 1, 4 and 8 OpenMP threads.


## Reference — plane-wave Nielsen–Martin analytic stress (diamond Si)

### Method

Cell stress from **Nielsen–Martin** Hellmann–Feynman cell derivatives after
**one SCF**: density is affinely remapped under symmetric strain (fixed
fractional coordinates), \(V_H\) / \(V_{xc}\) / \(V_\mathrm{loc}\) / Ewald
are rebuilt, one Davidson diagonalization is performed per strained cell,
and \(\sigma = (1/\Omega)\,\partial E_\mathrm{NSCF}/\partial\varepsilon\)
is formed by central differences. This is the numerical HF form of
Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) — **one SCF class**
cost, not 12 full SCFs.

Public path: `cpu::pw::pw_stress` → Voigt [eV/Å³].
Full multi-SCF strain-FD oracle: `cpu::pw::pw_stress_fd` (unit tests /
offline pin only).

### System (same twin as `pw-lda-si`)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| Lattice \(a\) | 5.43 Å |
| Basis | (0,0,0), \((a/4,a/4,a/4)\) |
| PP | `tests/data/pw/Si_lda_nc.upf` (PseudoDojo ONCVPSP-LDA, \(Z_\mathrm{val}=4\)) |
| XC | LSDA (Slater + VWN5) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; incomplete vs production ~60 Ry) |
| k-mesh | \(2\times2\times2\) |
| SCF | `tol=10^{-6}`, `max_iter=50` |

### Offline dual-gate pin (multi-SCF strain-FD)

Measured **2026-07-24** on Arrhenius epyc (Release, `OMP_NUM_THREADS=16`)
via `pw_stress_fd` (\(h=2\times 10^{-3}\)) on this exact twin:

| Component | Value [eV/Å³] |
| --------- | ------------- |
| \(\sigma_{xx},\sigma_{yy},\sigma_{zz}\) | ≈ **+0.05755** mechanical (mean; components equal within \(3\times 10^{-5}\)) |
| \(\sigma_{xy},\sigma_{xz},\sigma_{yz}\) | ≈ **0** (crystal symmetry; measured \(\lesssim 0.02\) residual at incomplete ecut/k) |
| Live dual-gate \(\max_v\lvert\sigma^\mathrm{an}-\sigma^\mathrm{FD}\rvert\) | **\(7.3\times 10^{-5}\)** eV/Å³ (magnitude; sign unified 2026-07-25) |

**Sign (2026-07-25 productization):** mechanical / CP2K
\(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\) (positive when
compressed) — matches GPW and `IsotropicCellRelax`. The offline multi-SCF
FD pin magnitude is unchanged; the hard-coded diagonal is **+0.05755**.

### CP2K note

CP2K Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH**, not ONCV
plane-wave. Absolute `STRESS_TENSOR` values are **not** comparable to this
NC-UPF twin; the fair PW twin is Quantum ESPRESSO (below). Optional
skeleton: `tools/cp2k_oracle/inputs/si_lda_pw_note.inp`.

### Literature / PseudoDojo context

| Quantity | Value | Source |
| -------- | ----- | ------ |
| \(a_0\) (EOS) | ≈ 5.468 Å | PseudoDojo `Si.djrepo` deltafactor \(v_0\) |
| \(B_0\) | 89.96 GPa | same |
| Twin \(a\) | 5.43 Å (compressed vs \(a_0\)) | regression |

Compressed cell → compressive hydrostatic \(\sigma_{xx}>0\) (mechanical /
CP2K \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\)), consistent
with the FD pin sign.

### Performance — measured fair wall (oracle: qe)

**Measured 2026-08-09** against the offline Quantum ESPRESSO twin
(`Program PWSCF v.7.6`, source build of 2026-08-09,
`/nobackup/proj/disk/teoroo/personal/jicli594/work/qe/install/bin/pw.x`;
install record `tools/qe_oracle/pin_notes.md`). Twin job = one SCF + cell
stress on the same NC UPF / cell / \(E_\mathrm{cut}=15\) Ry /
\(2\times2\times2\) mesh. Input:
`tools/qe_oracle/inputs/pw-analytic-stress-si/si_stress.in`
(`input_dft='SLA PW'`, `tstress=.true.`).

| Item | Value |
| ---- | ----- |
| \(T_{\mathrm{atv}}\) | **13.69 s** (`reg_pw-analytic-stress-si`: 1 SCF + 12 NSCF strained diagonalizations; median of 3) |
| \(T_{\mathrm{qe}}\) | **0.35 s** (1 SCF + analytic NM stress, QE-printed WALL, median of 3) |
| \(T_{\mathrm{atv}}/T_{\mathrm{qe}}\) | **39.1** |
| within_10x | **false — open PW perf debt** (was the stated target when exempt) |
| Live dual-gate (analytic + full FD) | ≈ 116 s offline only — not in ctest |

Protocol: Arrhenius x86_64 epyc login node (48-core, shared; load avg 7–24
during the window), one shell, 3 interleaved repetitions, medians; both sides
single process, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, QE as
`mpirun -np 1 pw.x`. ATV binary: Release, `-march=native`, this commit.

QE key output lines (provenance, not ctest pins):

```text
!    total energy              =     -15.67051079 Ry   (7 iters)
total   stress  (Ry/bohr**3)  diag 0.00019885 = 29.25 kbar, off-diag 0
```

Two recorded asymmetries, not excuses: (a) QE evaluates the **analytic**
Nielsen–Martin stress after 1 SCF while ATV's production path does 12 NSCF
strained diagonalizations — that algorithmic gap is most of the 39×; QE also
reduces the mesh to 3 IBZ k-points vs ATV's 8. (b) QE's analytic 29.25 kbar
vs ATV's FD-consistent ≈ 92 kbar (+0.0575 eV/Å³) differ because ATV's
strain-FD derivative of the *discretized* energy includes the finite-basis
(Pulay) contribution, which QE's infinite-basis analytic expression omits —
at this deliberately incomplete \(E_\mathrm{cut}=15\) Ry that term is tens of
kbar. The regression's correctness gate stays the internal analytic-vs-FD
dual gate, which is basis-consistent by construction.


## Reference — plane-wave LSDA bulk silicon on CUDA (`cuda::pw::PW`)

### Method

Spin-unpolarized plane-wave Kohn–Sham DFT with LSDA exchange–correlation
(Slater exchange + VWN5 correlation), norm-conserving ONCVPSP pseudopotential
(UPF), uniform Monkhorst–Pack Brillouin-zone sampling (`PW::k_mesh`³). Total
energy is reported in Rydberg on `PWResult::energy` and in Hartree on
`PWResult::energy_ha`.

Public path (both legs): set members on the method object → `build(frame)` →
`run(frame)`.

| Leg | Type | Where the work runs |
| --- | ---- | ------------------- |
| `cpu` | `cpu::pw::PW` | everything on the host |
| `cuda` | `cuda::pw::PW` | Davidson band solve, density accumulation **and** the whole density side (ρ(G) → V_H → V_xc → V_eff(r) → Pulay mixing → energy/residual reductions) on the GPU (cuFFT/cuBLAS). Host-side per run: problem assembly and the per-k table binding. Host-side per iteration: the ≤ 8×8 Pulay coefficient solve (shared verbatim with the CPU via `solve_pulay_coefficients`) and the energy report — 1 KB + 40 B of traffic, no O(N_G)/O(N_r) array. |

Device envelope of `cuda::pw::PW` (anything else throws with a how-to):
restricted (`spin_polarized = false`, `multiplicity = 1`), fixed closed-shell
occupations (`smearing = none`), `xc = XCType::LSDA`. k-points are **not**
distributed across MPI ranks on this backend.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013); M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Upstream | <https://github.com/PseudoDojo/ONCVPSP-LDA-SR-noCCv0.4> (`Si/Si.upf`) |

### Gate 1 — ABINIT-class literature window (external pin)

PseudoDojo ships an ABINIT deltafactor EOS for **this** ONCV LDA Si
pseudopotential in `Si/Si.djrepo` (retrieved 2026-07-23):

| Quantity | Value |
| -------- | ----- |
| Source | `Si.djrepo` deltafactor key `"8.0"`, `num_sites = 2` |
| \(E_\mathrm{min}\) | **−230.158202 eV/cell** = **−16.916316 Ry/cell** |
| \(v_0\) | 20.43445 Å³/atom → \(a_0 \approx 5.468\) Å (conventional cubic) |
| \(B_0\) | 89.96 GPa |

`oracle: literature` (ABINIT-class converged value for the same PP). The pin
and its provenance chain are shared verbatim with
`regressions/pw/rks/REFERENCE.md`; this case reuses it so the CUDA leg is
anchored to the same external number as the CPU leg, not to ATV output.

Because the twin is deliberately under-converged in basis and BZ sampling,
\(E_\mathrm{ATV}\) is **higher** (less bound) than \(E_\mathrm{min}\). Both
legs must satisfy the variational incomplete-basis window

\[
E_\mathrm{min}^\mathrm{ABINIT}
  \le E
  \le E_\mathrm{min}^\mathrm{ABINIT} + 1.50~\mathrm{Ry}.
\]

### Gate 2 — same-settings CPU parity (the CUDA-slice oracle)

Per the ES optimization order, a **same-settings CPU result is the correctness
oracle for a CUDA method slice**. With `--device=both` the binary runs both
legs on one frame and asserts

\[
|E_\mathrm{cuda} - E_\mathrm{cpu}| < 1.0\times10^{-3}~\mathrm{Ry}.
\]

The CPU leg is a **parity** reference, not a literature number — it carries no
independent physical authority and must never be cited as one.

The two backends share the host problem assembly, the Pulay coefficient solve
and the energy expression; the density side is now an independent device
implementation of the same algorithm. They therefore differ in:

* cuFFT vs FFTW butterfly order (ρ(r), V_xc(r), V_eff(r) and everything
  downstream, ~1e-15 relative);
* fixed-geometry tree reductions vs serial host loops for E_xc and the
  Parseval sums (the device form is the more accurate of the two);
* libdevice vs libm transcendentals in the VWN5/Slater kernel (≲ 1 ulp);
* Davidson arithmetic order (cuBLAS ZGEMM vs the serial CPU contraction);
* the 1e-4 `startingwfc` perturbation.

Per-stage agreement against the CPU stage objects is asserted as a **unit**
test in `tests/cuda/planewave/test_density.cu`; this slug asserts only that the
two converged total energies agree. \(10^{-3}\) Ry is a **design-time** bound
chosen before any GH200 measurement existed. Re-pin it from the first GH200
run; do not tighten it at design time.

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| Cell rows | \((0,h,h)/(h,0,h)/(h,h,0)\), \(h = a/2\) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| k-mesh | \(2\times2\times2\) Monkhorst–Pack (`k_mesh = 2`) |
| `xc` | `XCType::LSDA` on **both** legs |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 50 |

Identical to `regressions/pw/rks` except that the CPU leg pins
`xc = LSDA` explicitly (the CPU default is PBE) so the two legs are the same
Hamiltonian.

### Measurement log

| Item | Value |
| ---- | ----- |
| GH200 job id | **948152** (node type: NVIDIA GH200 120GB) — the committed tree |
| Toolchain | `GPU/buildenv-gcccuda/2026.03-cu13.0`, nvcc **13.0.88** / GNU 14.3.0, `-DCMAKE_CUDA_ARCHITECTURES=90` |
| GH200 log path | `.claude/cuda-pw-slice-948152.out` |
| First green | job **948055** (node `n438`) on the same slice before the review-fix round; **bit-identical** numbers, which is the evidence that the fix round was behaviour-preserving |
| Measured \(E_\mathrm{cpu}\) | **−15.67177311 Ry** (5 iterations) |
| Measured \(E_\mathrm{cuda}\) | **−15.67177311 Ry** (5 iterations) |
| Measured \(\lvert\Delta\rvert\) | **2.132 × 10⁻¹⁴ Ry** |
| Gate 1 (literature window) | PASS — both legs inside [−16.9163, −15.4163] Ry |
| Gate 2 (CPU parity) | PASS — 2.132e−14 Ry < 1.0e−03 Ry |

Both legs converge in the same iteration count, which is the stronger of the
two signals: the device mixing reproduces the host Pulay trajectory, not just
its fixed point.

The 1.0e−03 Ry parity bound is still the **design-time** value. The measured
gap is eleven orders of magnitude below it, so the bound is loose — but it is
deliberately not tightened from a single measurement on a single cell. Tighten
it only against a spread of geometries and cutoffs.

Independent band-level evidence from the same job (`diag_band_parity`, not a
ctest target): **MAX_EIG_DIFF = 6.051 × 10⁻¹⁵ Ha** across the Monkhorst–Pack
set, i.e. the Davidson eigenvalues agree to machine precision before any
density-side arithmetic enters.

### Performance

No performance claim. This slug is a correctness/parity gate on a 2-atom cell —
small molecules and 2-atom cells are correctness-only and are never the
"CUDA wins" story (ES optimization order, step 4). No wall-clock ratio is
recorded and none may be inferred from it.


## Reference — CUDA plane-wave analytic ionic forces

### Method

Device twin of `cpu::pw::pw_forces_analytic`:

- local Hellmann–Feynman from ρ(G) × v(G) (device kernel)
- Kleinman–Bylander nonlocal from β, ψ (device kernels)
- Ewald ion–ion (shared host classical sum)

Public API: `cuda::pw::pw_forces_analytic(frame, method)` [eV/Å].

### System

2-atom Si FCC primitive cell, lattice \(a = 5.43\) Å, second atom stretched
by \(+0.08\) Å along \(x\) so forces are nonzero and exercise KB projectors.

Settings (both backends):

| Member | Value |
| --- | --- |
| `gth_name` | `GTH-PBE-q4` |
| `ecut_ry` | 8.0 |
| `k_mesh` | 1 (Γ) |
| `tol` | 1e-6 |
| `max_iter` | 40 |
| `xc` | `XCType::LSDA` |

### Oracle

Same-settings CPU `cpu::pw::pw_forces_analytic` (not CP2K; force twin is the
landed CPU analytic path). Optional class: CPU FD gate in
`tests/cpu/planewave/test_pw_forces_fd.cpp` already covers analytic↔FD.

### Pin / gate

\[
\max_{i,\alpha}\lvert F^{\mathrm{cuda}}_{i\alpha} - F^{\mathrm{cpu}}_{i\alpha}\rvert
  \le 5\times 10^{-3}\ \mathrm{eV/Å}
\]

and \(\max |F^{\mathrm{cpu}}| > 0.05\) eV/Å (non-vacuous).

**GH200 job 1050105:** measured \(\max|\Delta F| = 2.86\times 10^{-13}\) eV/Å
(CPU = CUDA \(\max|F| = 2.517979\) eV/Å; force-path SCF energy both
\(-13.85556302\) Ry). Bound kept at design-time \(5\times 10^{-3}\).

**Root cause of earlier \(\sim 2\times 10^{-2}\) residual:** CPU force path
called `XCBuilder::step(ws)` without `method.xc`, so the NL band re-solve
rebuilt \(V_{\mathrm{eff}}\) with the default **PBE** while SCF (and CUDA)
used **LSDA**. Fixed to `xc.step(ws, method.xc)`.

Not bit-exact by design (independent SCF + FFT/eigensolve paths) but
observed residual is at floating-point noise for this cell.

### Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU force parity as above |
| Wall | **perf_exempt** — correctness foothold on small cell |

### Non-goals

Stress product, collinear spin, metallic smear, energy-path changes.


## Reference — CUDA plane-wave Nielsen–Martin cell stress

### Method

Device twin of `cpu::pw::pw_stress`:

1. One device SCF at the reference geometry.
2. Hellmann–Feynman cell stress by central finite differences of the
   *non-self-consistent* total energy under symmetric strain (density
   remapped \(\rho\to\rho\cdot\Omega_0/\Omega\), potentials rebuilt, one
   Davidson diagonalization per strained cell).

Sign: mechanical / CP2K \(\sigma=-(1/\Omega)\,\partial E/\partial\varepsilon\).

Public API: `cuda::pw::pw_stress(frame, method[, energy_ry_out])` → Voigt
σ [eV/Å³].

### System

2-atom Si FCC primitive cell, lattice \(a = 5.43\) Å (compressed vs
\(a_0\approx 5.47\) → positive hydrostatic stress).

Settings (both backends):

| Member | Value |
| --- | --- |
| `gth_name` | `GTH-PBE-q4` |
| `ecut_ry` | 8.0 |
| `k_mesh` | 1 (Γ) |
| `tol` | 1e-6 |
| `max_iter` | 40 |
| `xc` | `XCType::LSDA` |

### Oracle

Same-settings CPU `cpu::pw::pw_stress` (not CP2K; stress twin is the landed
CPU Nielsen–Martin analytic path). Offline multi-SCF FD class covered by
`reg_pw-analytic-stress-si` / `pw_stress_fd` on CPU.

### Pin / gate

\[
\max_{v}\lvert \sigma^{\mathrm{cuda}}_{v} - \sigma^{\mathrm{cpu}}_{v}\rvert
  \le 5\times 10^{-2}\ \mathrm{eV/Å^3}
\]

plus cubic symmetry (off-diag and diag equality within \(2.5\times 10^{-2}\)).

### Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU stress parity as above |
| Wall | **perf_exempt** — correctness foothold (1 SCF + 12 NSCF; small cell) |

### Non-goals

Collinear spin, metallic smear, multi-SCF FD production path, variable-cell
driver product, energy-path changes.


## Reference — CUDA plane-wave Nielsen–Martin cell stress under PBE

### Method

Device twin of `cpu::pw::pw_stress` with **`XCType::PBE`**:

1. One device SCF at the reference geometry (device PBE-GGA on the FFT grid:
   spectral ∇ρ / −∇·flux in `DeviceDensity`, kernels in `xc_pbe.cu`).
2. Hellmann–Feynman cell stress by central finite differences of the
   *non-self-consistent* total energy under symmetric strain (density
   remapped \(\rho\to\rho\cdot\Omega_0/\Omega\), potentials rebuilt with the
   same PBE path, one Davidson diagonalization per strained cell).

Sign: mechanical / CP2K \(\sigma=-(1/\Omega)\,\partial E/\partial\varepsilon\).

Public API: `cuda::pw::pw_stress(frame, method[, energy_ry_out])` → Voigt
σ [eV/Å³] for a `method` constructed with `XCType::PBE`.

Sibling of `cuda-pw-stress` (LSDA). Closes the closed-shell PBE stress
dual-gate left open after the device PBE energy path + `cuda-pw-stress` (LSDA).

### System

2-atom Si FCC primitive cell, lattice \(a = 5.43\) Å (compressed vs
\(a_0\approx 5.47\) → positive hydrostatic stress).

Settings (both backends):

| Member | Value |
| --- | --- |
| `gth_name` | `GTH-PBE-q4` |
| `ecut_ry` | 8.0 |
| `k_mesh` | 1 (Γ) |
| `tol` | 1e-6 |
| `max_iter` | 40 |
| `xc` | **`XCType::PBE`** |

### Oracle

Same-settings CPU `cpu::pw::pw_stress` under PBE (not CP2K). Offline multi-SCF
FD class covered by `reg_pw-analytic-stress-si` / `pw_stress_fd` on CPU (LSDA
class); PBE stress twin is this dual-gate.

### Pin / gate

\[
\max_{v}\lvert \sigma^{\mathrm{cuda}}_{v} - \sigma^{\mathrm{cpu}}_{v}\rvert
  \le 5\times 10^{-2}\ \mathrm{eV/Å^3}
\]

plus cubic symmetry (off-diag and diag equality within \(2.5\times 10^{-2}\)).

Design-time bound matches LSDA `cuda-pw-stress` / CPU analytic↔FD window;
re-pin from the first GH200 log without tightening at design time.

**GH200 job 1122143** (`wf/cuda-pw-pbe-stress`): unit
`test_cuda_pw_cell_stress` **1/1 PASS** (H₂ LSDA + H₂ PBE twins);
dual-gate \(\max|\Delta\sigma| = 5.585\times 10^{-12}\) eV/Å³ (tol
\(5\times 10^{-2}\)); \(\mathrm{mean}|diag_{cpu}| = 0.273236\) eV/Å³;
CPU = CUDA \(E = -13.90403280\) Ry, stress-SCF **iter=7**,
\(|\Delta E| = 4.441\times 10^{-14}\) Ry. Bound kept at design-time
\(5\times 10^{-2}\).

**Re-verify GH200 job 1122705** after rebase onto `dev@d701a1cd`: unit
1/1 PASS; dual-gate max|Δσ| = \(5.585\times 10^{-12}\) eV/Å³; same E and
|ΔE|; `BUILDS_WIPED_OK`.

**Re-verify GH200 job 1123959** after rebase onto `dev@4913b33b` (Assign
wave 3): unit 1/1 PASS; dual-gate max|Δσ| = \(5.585\times 10^{-12}\) eV/Å³;
same E and |ΔE|; `BUILDS_WIPED_OK`.

### Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU PBE stress parity as above |
| Wall | **perf_exempt** — correctness foothold (1 SCF + 12 NSCF; small cell) |

### Non-goals

Collinear spin, metallic smear, multi-SCF FD production path, variable-cell
driver product, energy-path changes beyond stress-path SCF.


## Reference — MPI PW k-mesh round-robin scaling gate (4 ranks)

### Method

Spin-unpolarized plane-wave Kohn–Sham LDA on bulk silicon — the **identical
twin** to `regressions/pw/rks` (same geometry, pseudopotential, cutoff,
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
invoked by ctest**; the ctest entry `pw/mpi-pw-scaling-4rank` runs the case
binary once and the case starts its own ranks.

### Pseudopotential (same in-tree file as pw-lda-si)

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013); M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |

### Twin geometry and settings

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

### Pinned acceptance

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

### Measured on this host (2026-07-25, Release, MPICH, 4 OMP threads/rank)

| Quantity | np = 1 | np = 4 |
| -------- | ------ | ------ |
| \(E\) [Ry] | −15.704816618 | −15.704816618 |
| SCF iterations | 6 | 6 |
| owned k per rank | 8 | 2 |
| \(T_\mathrm{run}\) [s] | 3.801 | 1.495 |

\(|\Delta E| = 0.0\) Ry (bit-identical); speedup **2.54×** (floor 1.2);
total regression wall ≈ 12 s — inside the 60 s budget.

### CP2K note (absolute energy + wall)

CP2K Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH** (different PP and
discretization): absolute total energies are **not** comparable to ONCV
plane-wave, and the fair CP2K science comparison is EOS-level (see
`regressions/pw/rks/REFERENCE.md`). An offline wall-twin input
skeleton lives at `tools/cp2k_oracle/inputs/si_lda_pw_note.inp`.

### Performance

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
