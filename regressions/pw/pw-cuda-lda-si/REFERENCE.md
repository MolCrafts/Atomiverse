# Reference — plane-wave LSDA bulk silicon on CUDA (`cuda::pw::PW`)

## Method

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

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013); M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Upstream | <https://github.com/PseudoDojo/ONCVPSP-LDA-SR-noCCv0.4> (`Si/Si.upf`) |

## Gate 1 — ABINIT-class literature window (external pin)

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
`regressions/pw/pw-lda-si/REFERENCE.md`; this case reuses it so the CUDA leg is
anchored to the same external number as the CPU leg, not to ATV output.

Because the twin is deliberately under-converged in basis and BZ sampling,
\(E_\mathrm{ATV}\) is **higher** (less bound) than \(E_\mathrm{min}\). Both
legs must satisfy the variational incomplete-basis window

\[
E_\mathrm{min}^\mathrm{ABINIT}
  \le E
  \le E_\mathrm{min}^\mathrm{ABINIT} + 1.50~\mathrm{Ry}.
\]

## Gate 2 — same-settings CPU parity (the CUDA-slice oracle)

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

## Twin geometry and settings

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

Identical to `regressions/pw/pw-lda-si` except that the CPU leg pins
`xc = LSDA` explicitly (the CPU default is PBE) so the two legs are the same
Hamiltonian.

## Measurement log

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

## Performance

No performance claim. This slug is a correctness/parity gate on a 2-atom cell —
small molecules and 2-atom cells are correctness-only and are never the
"CUDA wins" story (ES optimization order, step 4). No wall-clock ratio is
recorded and none may be inferred from it.
