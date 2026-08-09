# Reference — k-point collinear UKS (O₂ multi-k)

## Method

Collinear spin-polarized plane-wave KS (`cpu::pw::PW` with
`spin_polarized = true`, `multiplicity = 3`) on a **Monkhorst–Pack
`k_mesh = 2`** mesh (not Γ-only). Fixed occupations N↑ = 7, N↓ = 5 for 12
valence electrons (oxygen NC-PP Z_val = 6 × 2). XC is `XCType::PBE`, which
since 2026-07-26 means **true collinear spin-GGA** (spectral ∇ρ_σ in
`XCBuilder::step_spin`).

**Geometry choice (historical):** the slug says `fe` because the queue item
allowed Fe *or* an O₂ cell, and at the time bulk Fe needed spin **and**
metallic smearing, which `PW::run` did not yet co-support. It does now —
that product is `pw-uks-fe-smearing`. This case therefore gates **multi-k
collinear UKS on O₂**; the `fe` in the directory name is a fossil. (Renaming
the directory is a separate change; it touches `regressions/CMakeLists.txt`.)

## System

| Quantity | Value |
| -------- | ----- |
| Molecule | O₂ |
| R(O–O) | 1.2075 Å |
| Cell | cubic L = 5.5 Å |
| PP | SG15 ONCV PBE oxygen `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| E_cut | 5 Ry |
| k-mesh | Monkhorst–Pack 2×2×2 (8 points, weights 1/8) |
| Multiplicity | 3 (triplet ³Σg⁻) |
| SCF tol (density L2) | 2×10⁻³ |
| max_iter | 20 |
| mix_beta | 0.5 |

## Gates

### 1. Magnetization (literature + fixed occupations)

O₂ ground state is the triplet ³Σg⁻ (Herzberg, *Molecular Spectra and
Molecular Structure* I). With fixed integer occupations and multiplicity 3:

\[
M = N_\uparrow - N_\downarrow = 2.
\]

**Gate:** `|M − 2| ≤ 1×10⁻⁹` on `PWResult::magnetization`, for **both**
functionals below (the identity is XC-independent).

### 2. The run is genuinely multi-k

**Gate:** `PWResult::n_kpts == k_mesh³ == 8`. A k-point case that silently
degrades to Γ would otherwise be indistinguishable from `pw-spin-polarized-o2`.

### 3. Spin-GGA activation floor

**Gate:** `|E(xc = PBE) − E(xc = LSDA)| > 1×10⁻³ Ha` at identical settings.

This is an **activation threshold, not a frozen value**: it asserts only that
the gradient terms of the requested functional actually reach the collinear
solver. Measured on this host: `8.15×10⁻² Ha`.

Its reason for existing is the defect described below.

## Void pin — what the old number actually was

This case previously gated an absolute total energy
`E_tot = −25.1087178036 Ha ± 5×10⁻³` described as an "offline method pin"
(ATV's own output frozen on 2026-07-25). **That number is void, and it was
never an external reference.**

Before commit `c6a6273c` (2026-07-26, `pw-spin-gga`),
`XCBuilder::step_spin` evaluated *every* functional at σ = 0 — so a request
for `xc = PBE` on the collinear path silently ran the **LDA limit**. The
fossil pin is a snapshot of that degraded Hamiltonian. Measured at these
exact settings on Arrhenius (x86_64, OMP=1):

| Quantity | Value [Ha] |
| -------- | ---------- |
| Fossil pin (recorded 2026-07-25) | −25.1087178036 |
| `xc = LSDA` (local spin XC, σ = 0) | −25.1108284698 |
| `xc = PBE` (true collinear spin-GGA) | −25.1923610263 |
| pin − LSDA | **+0.0021** |
| pin − spin-GGA | **+0.0836** (17 × the old tolerance) |

The pin sits 0.0021 Ha from the *local* answer and 0.0836 Ha from the answer
for the functional the case asks for. The 0.0021 Ha residue is the
"PBE-evaluated-at-σ=0 vs the LSDA parameterisation" difference; the Γ-only
sibling shows the same fossil independently (its own old σ=0 pin
−28.6267426458 Ha vs measured LSDA −28.6294634700 Ha, i.e. 0.0027 Ha).
`c6a6273c` re-pinned that Γ sibling and missed this one.

**Why the pin was deleted rather than moved.** O₂ at E_cut = 5 Ry in a 5.5 Å
box has no citable external value — no literature number, no analytic result
— and re-freezing ATV's current output would validate nothing; it is what
produced two stale pins in five days. The gates above are exact statements (a
literature occupation identity, a mesh-size identity, and an activation
floor) instead. A `pw.x` NC-UPF twin now exists offline (see Perf below);
promoting its total energies to a correctness pin is separate queue work.

**Oracle note:** host CP2K is not an absolute-energy twin here (GTH-GPW ≠
NC-UPF PW); same limitation as `pw-spin-polarized-o2`. Optional documentation
twin (algorithm only, never energy-compared): Quickstep UKS + `KPOINTS`
Monkhorst–Pack 2×2×2 on the same O₂ cell.

## Perf — measured fair wall (oracle: qe)

**Measured 2026-08-09** against the offline Quantum ESPRESSO twin
(`Program PWSCF v.7.6`, source build of 2026-08-09,
`/nobackup/proj/disk/teoroo/personal/jicli594/work/qe/install/bin/pw.x`;
install record `tools/qe_oracle/pin_notes.md`). Twin job = the exact
regression workload: collinear spin-PBE SCF then spin-LSDA SCF, same SG15
ONCV PBE oxygen UPF (`tests/data/pw/O_ONCV_PBE-1.2.upf`), cubic L = 5.5 Å,
\(E_\mathrm{cut}=5\) Ry, MP \(2\times2\times2\), fixed occupations
`tot_magnetization = 2`. Inputs: `tools/qe_oracle/inputs/pw-uks-kmesh-fe/`
(`o2_pbe.in`, `o2_lsda.in` with `input_dft='SLA PW'`).

| Field | Value |
| ----- | ----- |
| \(T_{\mathrm{atv}}\) | **25.22 s** (`reg_pw-uks-kmesh-fe`, two solves; median of 3) |
| \(T_{\mathrm{qe}}\) | **0.30 s** (0.17 s spin-PBE + 0.13 s spin-LSDA, QE-printed WALL, median of 3) |
| \(T_{\mathrm{atv}}/T_{\mathrm{qe}}\) | **84.1 — worst of the wave-3 PW batch; open PW perf debt** |

Protocol: Arrhenius x86_64 epyc login node (48-core, shared; load avg 7–24
during the window), one shell, 3 interleaved repetitions, medians; both sides
single process, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, QE as
`mpirun -np 1 pw.x`. ATV binary: Release, `-march=native`, this commit.

QE key output lines (provenance, not ctest pins — the case deliberately
carries no absolute-energy pin, see "Void pin"):

```text
o2_pbe.out : !    total energy              =     -50.38166627 Ry  (15 iters, M = 2.00)
o2_lsda.out: !    total energy              =     -50.20591384 Ry  (10 iters, M = 2.00)
```

Known cost asymmetry, recorded not excused: QE reduces the unshifted
\(2\times2\times2\) mesh to 6 IBZ k-points; ATV solves all 8 (that is this
case's own gate 2). That is ≲ 1.3× of the 84× — at this tiny basis
(\(E_\mathrm{cut}=5\) Ry) essentially the whole gap is ATV per-iteration and
fixed-cost overhead, making this the sharpest small-system profile target of
the batch.

## Dual gate

1. Correctness: `converged` + M = 2 + `n_kpts = 8` + spin-GGA activation, for
   both functionals.
2. Perf: measured fair wall above (`oracle: qe`); ratio 84.1 recorded as open
   perf debt.
