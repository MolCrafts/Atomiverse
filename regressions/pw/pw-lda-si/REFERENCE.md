# Reference — plane-wave LDA bulk silicon (k-mesh energy)

## Method

Spin-unpolarized plane-wave Kohn–Sham DFT with LDA exchange–correlation
(Slater exchange + PW92 correlation on the PW real-space grid). Norm-conserving
ONCVPSP pseudopotential (UPF). Brillouin-zone sampling is a uniform
Monkhorst–Pack mesh (`PW::k_mesh`³). Total energy is reported in Rydberg on
`PWResult::energy` and in eV on `frame.meta.get(kEnergy)`.

Public path: `cpu::pw::PW` (members) → `cpu::pw::PWEvaluator` →
`cpu::single_point`.

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013); M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Upstream | <https://github.com/PseudoDojo/ONCVPSP-LDA-SR-noCCv0.4> (`Si/Si.upf`) |

## ABINIT-class same-pseudo pin (converged limit)

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

## Twin geometry and settings (regression)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| k-mesh | \(2\times2\times2\) Monkhorst–Pack (`k_mesh = 2`) |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 50 |

## Acceptance band (incomplete basis)

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

## CP2K note (absolute energy)

CP2K Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH** (different PP and
discretization). Absolute total energies are **not** comparable to ONCV
plane-wave; EOS (\(a_0\), \(B_0\)) is the fair CP2K science comparison (see
historical `benchmarks/pw` docs). An optional twin input skeleton lives under
`tools/cp2k_oracle/inputs/si_lda_pw_note.inp` for offline wall timing only.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K (or ABINIT) binary / module on the build host; fair wall twin not run offline this shift |
| ATV twin wall (informative) | ≈ 8 s (Release, 16 threads, ecut 15 / k=2) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time the same geometry with the oracle input,
append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) here, require \(r \le 10\).
