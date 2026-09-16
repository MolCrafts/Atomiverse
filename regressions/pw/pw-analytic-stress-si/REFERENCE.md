# Reference — plane-wave Nielsen–Martin analytic stress (diamond Si)

## Method

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

## System (same twin as `pw-lda-si`)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| Lattice \(a\) | 5.43 Å |
| Basis | (0,0,0), \((a/4,a/4,a/4)\) |
| PP | `tests/data/pw/Si_lda_nc.upf` (PseudoDojo ONCVPSP-LDA, \(Z_\mathrm{val}=4\)) |
| XC | LSDA (Slater + PW92) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; incomplete vs production ~60 Ry) |
| k-mesh | \(2\times2\times2\) |
| SCF | `tol=10^{-6}`, `max_iter=50` |

## Offline dual-gate pin (multi-SCF strain-FD)

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

## CP2K note

CP2K Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH**, not ONCV
plane-wave. Absolute `STRESS_TENSOR` values are **not** comparable to this
NC-UPF twin; the fair PW twin is Quantum ESPRESSO (below). Optional
skeleton: `tools/cp2k_oracle/inputs/si_lda_pw_note.inp`.

## Literature / PseudoDojo context

| Quantity | Value | Source |
| -------- | ----- | ------ |
| \(a_0\) (EOS) | ≈ 5.468 Å | PseudoDojo `Si.djrepo` deltafactor \(v_0\) |
| \(B_0\) | 89.96 GPa | same |
| Twin \(a\) | 5.43 Å (compressed vs \(a_0\)) | regression |

Compressed cell → compressive hydrostatic \(\sigma_{xx}>0\) (mechanical /
CP2K \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\)), consistent
with the FD pin sign.

## Performance — measured fair wall (oracle: qe)

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
