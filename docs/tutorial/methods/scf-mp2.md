# Møller–Plesset perturbation theory (MP2, MP3, MP4)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Møller–Plesset perturbation theory adds electron correlation to a converged
Hartree–Fock reference by Rayleigh–Schrödinger perturbation theory with
\(H_0=\sum_p \varepsilon_p a_p^\dagger a_p\). The public entry points are
post-HF objects that bind a concrete mean-field method and drive a fresh SCF
on it (`cpu/scf/post_hf.h`, mirroring `pyscf.mp.MP2(mf)`):

- `cpu::scf::MP2<Method>{mf}` — members `frozen`, `os_scale`, `ss_scale`;
  `run(frame, guess)` → `MP2Result` (`e_corr_os`, `e_corr_ss`, `e_corr`,
  `e_tot` [Ha], `frozen`, `ri`); `forces(frame, guess)` → `GradientResult`
  [eV/Å] on \(E_{\mathrm{HF}}+E_{\mathrm{corr}}\). `RHF`/`RKS` → RMP2,
  `UHF` → UMP2, `ROHF` → ROMP2.
- `cpu::scf::MP3<RHF>{rhf}` / `cpu::scf::MP4<RHF>{rhf}` — member `frozen`;
  `run(frame)` → `MP3Result` / `MP4Result`.

All sections run on CPU. The frozen-core water, SCS water and eight-water
chain sections also support `--device=cuda|both`. Besides the seven documented below, the case carries
`mp2-grad-h2o-sto3g` (H₂O RMP2/STO-3G analytic gradient) and
`posthf-eri-block-mp2` (H₂O RMP2/STO-3G through the block-assembled dense
ERI tensor), whose oracle records live in the case `REFERENCE.md`. The
double-hybrid use of `MP2<RKS>` is [`b2plyp-h2o`](scf-rks.md#b2plyp-h2o).
Background: [Ch. 7 Correlation](../ch07-correlation.md),
[Ch. 3 Open shell](../ch03-open-shell.md).

Case: `ctest -R '^scf/mp2$'` — sources and oracle record in [`regressions/scf/mp2/`](../../../regressions/scf/mp2/). One section of the case per `##` heading below: `atv_regressions --case scf/mp2 --section <name>`.

## CUDA closed-shell MP2

Bind a CUDA RHF reference that outlives the correlation object:

```cpp
#include "atomiverse/cuda/scf/rhf.cuh"
#include "atomiverse/cuda/scf/mp2.cuh"

cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-10, 200, 0.0, false};
cuda::scf::MP2 correlation{rhf};
correlation.frozen = 0;
correlation.os_scale = 1.2;       // SCS; leave both scales at 1 for plain MP2
correlation.ss_scale = 1.0 / 3.0;
const auto result = correlation.run(frame);
```

`result` contains Hartree energies; mutable `run(frame)` writes the correlated
total to `frame.meta.get(kEnergy)` in eV, as does CPU MP2. The CUDA const-Frame
entry leaves its input unchanged. CUDA MP2 supports canonical RHF correlation,
frozen occupied orbitals and OS/SS scaling. Unsupported force requests throw.
It does not provide UMP2, RI-MP2 or MP2 gradients.

The RHF orbitals, eigenvalues and AO ERIs remain on device through the MP2
calculation. Occupied tiling and cuBLAS/CUB avoid a full MO ERI tensor.
The optional constructor byte budget bounds additional MP2 scratch (default
512 MiB), excluding the reference HF workspace and library contexts. The
current dense HF indices require `nAO^4 <= INT_MAX` (at most 215 AOs).
The method can also borrow an energy-only parent CUDA driver; its physical
result remains in that parent's Frame.

`bench_mp2_cpu_vs_cuda` measures complete first and repeated RHF→MP2 calls,
including setup and final output. Interpret its thread counts, iteration
counts and memory planning alongside the numerical comparisons; a memory
snapshot alone does not establish peak use or device residency.

## Common setup and oracle

- **Oracle:** offline PySCF **2.14.0** pins (`docs/tutorial/appendix-pyscf-oracle.md`);
  nothing in ctest imports PySCF. Every pin uses Cartesian AOs (`cart=True`)
  and SCF `conv_tol=1e-12`. PySCF has no `mp.MP3` / `mp.MP4`; those pins
  rebuild the diagrams on PySCF RHF orbitals and `ao2mo` MO ERIs.
- **Water** (RMP2 sections): O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) Å.
- **Triplet O₂** (UMP2 / ROMP2 sections): O(0, 0, 0); O(0, 0, 1.2075) Å,
  charge 0, \(2S=2\), \(n_\alpha=9\), \(n_\beta=7\), STO-3G, cubic box 20 Å.
- **H₂** (MP3 / MP4 sections): H(0, 0, 0); H(0, 0, 0.74) Å, charge 0,
  singlet (\(2S=0\)), STO-3G; AO / MO 2 / 2 (1 occupied spatial MO); RHF,
  canonical, no frozen core; `tol = 1e-12`, `max_iter = 100`.
- **Build:** `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU build
  environment `buildenv-gcc/2026.03-mpich`, see the repository build notes).
- **Fair wall:** \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) against
  the matched PySCF twin (`python <pyscf fair-bench> --method <m> … -- <section binary>`).
  Absolute times on these minimal systems are tiny and noise-dominated; the
  ratio is not a large-N scaling claim.

**Multi-minima O₂ UHF.** Cold UHF from the core Hamiltonian often lands on a
higher local solution for O₂ (MINAO reaches the PySCF minimum). The UMP2
sections warm-start UHF from ROHF \(\alpha/\beta\) densities:

```cpp
cpu::Frame frame(2);   // O₂ along z
frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 8;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_z)[1] = 1.2075;
frame.box = cpu::Box::cubic(20.0);

cpu::scf::ROHF rohf{cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7};
rohf.tol = 1e-12;
rohf.max_iter = 200;
rohf.run(frame);

using Spin = cpu::scf::SlaterDeterminant::Spin;
cpu::scf::SCFInitialGuess guess;
guess.P_alpha = rohf.determinant().density(Spin::alpha);
guess.P_beta  = rohf.determinant().density(Spin::beta);
guess.has_alpha = true;
guess.has_beta  = true;

cpu::scf::UHF uhf{cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7};
uhf.tol = 1e-12;
uhf.max_iter = 200;
```

## References

- C. Møller, M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
- A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, Dover
- Pople, Krishnan, Schlegel, Binkley, *Int. J. Quantum Chem. Symp.* **13**, 225
  (1979)
- Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* (response /
  gradient chapters)

## frozen-core-mp2-h2o — Frozen-core RMP2, H₂O / 6-31G* {#frozen-core-mp2-h2o}

Closed-shell MP2 with the lowest \(N_{\mathrm{frozen}}\) occupied MOs excluded
from the correlation sum (PySCF `mp.MP2(mf, frozen=N)` convention).

### Theory

Core orbitals contribute little to valence chemistry but cost the same
\(O(N^5)\) terms. Freezing the O 1s on water is the textbook frozen-core demo:
\(E_{\mathrm{corr}}\) rises by ~2.4 mHa relative to full RMP2 on the same
geometry/basis.

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Pople631Gs};  // 6-31G*
rhf.tol = 1e-12;
cpu::scf::MP2<cpu::scf::RHF> mp_corr{rhf};
mp_corr.frozen = 1;   // freeze lowest occupied spatial MO (O 1s)
const cpu::scf::MP2Result mp = mp_corr.run(frame);
```

### Results

#### Accuracy vs PySCF 2.14.0

Water geometry; cart 6-31G\*; frozen=1.

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{RHF}}\) | −76.010504988257 | 1e-8 |
| \(E_{\mathrm{corr}}\) (frozen) | **−0.186119540118** | **1e-6** |
| \(E_{\mathrm{tot}}\) | **−76.196624528375** | **1e-6** |

Unfrozen RMP2 \(E_{\mathrm{corr}}\approx -0.18852\) Ha on the same setup (diagnostic).

#### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `mp2` | 3.10 s | 0.23 s | **0.073** | ✓ |

## scs-mp2-h2o-sto3g — SCS-MP2, H₂O / STO-3G {#scs-mp2-h2o-sto3g}

Spin-component-scaled MP2 (Grimme): scale opposite-spin and same-spin
correlation channels separately.

$$ E_{\mathrm{SCS}} = \tfrac{1}{3} E_{\mathrm{SS}} + 1.2\, E_{\mathrm{OS}}. $$

### Theory

Plain RMP2 overbinds same-spin correlation for many organic systems. SCS-MP2
is a cheap empirical reweighting of channels already available after the MP2
amplitudes — no extra \(O(N^5)\) cost beyond RMP2.

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
cpu::scf::MP2<cpu::scf::RHF> mp_corr{rhf};
mp_corr.os_scale = 1.2;
mp_corr.ss_scale = 1.0 / 3.0;
const cpu::scf::MP2Result mp = mp_corr.run(frame);
```

### Results

#### Accuracy vs PySCF 2.14.0

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{OS}}\) (unscaled) | −0.033515820085 | 1e-6 |
| \(E_{\mathrm{SS}}\) (unscaled) | −0.002029831562 | 1e-6 |
| plain \(E_{\mathrm{corr}}\) | −0.035545651647 | 1e-6 |
| **SCS \(E_{\mathrm{corr}}\)** | **−0.040895594623** | **1e-6** |

#### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| plain `mp2` | 2.05 s | 0.026 s | **0.012** | ✓ |

## ump2-o2-sto3g — UMP2, triplet O₂ / STO-3G {#ump2-o2-sto3g}

Unrestricted Møller–Plesset second-order correlation on a UHF reference.

### Theory

Closed-shell RMP2 assumes a restricted HF reference. Open-shell
singlets/triplets need **UMP2**: spin-orbital second-order perturbation theory
on **UHF** orbitals, with same-spin and opposite-spin channels (unprojected;
residual \(\langle S^2\rangle\) contamination is accepted as in raw PySCF
`mp.UMP2`).

$$ E_{\mathrm{corr}}^{\mathrm{UMP2}} = E_{\mathrm{SS}} + E_{\mathrm{OS}}, \qquad
   E_{\mathrm{tot}} = E_{\mathrm{UHF}} + E_{\mathrm{corr}}. $$

### Practice

After the ROHF warm-start of the common setup:

```cpp
const cpu::scf::MP2Result mp =
    cpu::scf::MP2<cpu::scf::UHF>{uhf}.run(frame, &guess);   // e_corr, e_tot [Ha]
```

### Results

#### Accuracy vs PySCF 2.14.0

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{UHF}}\) | −147.633946785450 | 1e-8 (diag.) |
| \(E_{\mathrm{corr}}\) (UMP2) | **−0.096393144130** | **1e-6** |
| \(E_{\mathrm{tot}}\) | **−147.730339929580** | **1e-6** |

#### Speed (fair wall)

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `ump2` | 2.51 s | 0.57 s | **0.23** | ✓ |

## ump2-grad-o2-sto3g — UMP2 analytic gradient, triplet O₂ / STO-3G {#ump2-grad-o2-sto3g}

### Summary

Analytic nuclear forces on the **total UMP2 energy**
\(E_{\mathrm{UHF}} + E_{\mathrm{corr}}\) for high-spin triplet O₂ with
Cartesian STO-3G, via `MP2<UHF>::forces`. Forces return as `GradientResult`
([eV/Å]); the section converts to Ha/Bohr for the pin. UHF is warm-started
from ROHF densities as in [`ump2-o2-sto3g`](#ump2-o2-sto3g). Oracle: PySCF
`mp.UMP2(...).nuc_grad_method().kernel()`. Fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 1.37\).

### Method

**Reference.** UMP2 is second-order Møller–Plesset perturbation theory in spin
orbitals on the UHF reference (unprojected). No frozen core.

\[
E_{\mathrm{UMP2}} = E_{\mathrm{UHF}} + E_{\mathrm{corr}}^{\mathrm{UMP2}},
\qquad
E_{\mathrm{corr}}^{\mathrm{UMP2}} = E_{\mathrm{SS}} + E_{\mathrm{OS}}.
\]

**Gradient composition.** Analytic UMP2 nuclear gradient (Pople / Helgaker
response form; PySCF `pyscf.grad.ump2`):

- Spin-resolved **Lagrangian 1-RDM** (unrelaxed MP2 density plus
  orbital-response Z-vector on the unrestricted \(A+B\) coupling blocks).
- **Non-separable 2-RDM** from the \(T_2\) amplitudes.
- Contraction of those densities with **skeleton integral derivatives**
  (\(\partial h/\partial R\), \(\partial S/\partial R\), \(\partial(ij|kl)/\partial R\)).

Both Atomiverse and PySCF deliver **analytic** UMP2 gradients; agreement to
\(\sim 10^{-6}\)–\(10^{-8}\) Ha/Bohr is expected on the minimal system.
Missing response or 2-RDM terms typically fail by \(\ge 10^{-3}\) Ha/Bohr.

**Scope.**

- Canonical UMP2 only: `MP2::forces` rejects ROHF / ROMP2, RI/DF,
  `frozen > 0`, non-default SCS scales, and UKS/ROKS references.
- Analytic nuclear gradient only; no UMP2 Hessian / vibrational analysis.

### Implementation

After the ROHF warm-start of the common setup:

```cpp
const cpu::scf::GradientResult res =
    cpu::scf::MP2<cpu::scf::UHF>{uhf}.forces(frame, &guess);
// res.forces(A, d) in eV/Å; res.energy total UMP2 [eV]
// Ha/Bohr: -units::convert(f, Force::eVPerAngstrom, Force::HartreePerBohr)
// (sign: GradientResult stores forces = −∇E)
```

Headers: `post_hf.h` (`MP2<Method>`), `methods.h` (concrete `UHF` / `ROHF`),
`gradient_result.h`, `frame.h`. Private analytic kernel: `mp2_gradient.h`
(`ump2_analytic_forces`).

### Validation (vs PySCF)

Oracle: `mp.UMP2(mf).run(); m2.nuc_grad_method().kernel()`. Energy pins are
those of [`ump2-o2-sto3g`](#ump2-o2-sto3g):

| Quantity | Pin [Ha] | Notes |
| -------- | -------: | ----- |
| \(E_{\mathrm{UHF}}\) | −147.633946785450 | |
| \(E_{\mathrm{corr}}\) (UMP2) | −0.096393144130 | |
| \(E_{\mathrm{UMP2}}\) | −147.730339929580 | \(E_{\mathrm{UHF}}+E_{\mathrm{corr}}\) |

Analytic gradient \(\partial E/\partial R\) [Ha/Bohr] (collinear diatomic on
\(z\); \(x,y\) ~0):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O (0) | 0 | 0 | **+1.665132663664e−01** |
| O (1) | 0 | 0 | **−1.665132663664e−01** |

| Gate | Tol | Notes |
| ---- | --: | ----- |
| Element-wise \(\lvert g_{\mathrm{ATV}} - g_{\mathrm{PySCF}}\rvert\) | **\(1\times 10^{-5}\)** Ha/Bohr | expected match \(\sim 10^{-6}\)–\(10^{-8}\) |
| Translational invariance \(\lvert\sum_A g_A\rvert_{\max}\) | **\(1\times 10^{-8}\)** Ha/Bohr | \(\sum_A g_A \approx 0\) to \(\sim 10^{-15}\) on pin |

Why \(1\times 10^{-5}\) Ha/Bohr: it covers packing / AO-slice assembly noise
without masking a missing Z-vector or 2-RDM term (those errors are typically
\(\ge 10^{-3}\) Ha/Bohr).

Oracle re-pin:

```bash
python <pyscf pin> ump2 --geom o2 --basis sto-3g \
    --spin 2 --conv-tol 1e-12 --grad
```

### Performance

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, `--method ump2_grad --repeat 3`.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `ump2_grad` (UHF + UMP2 + analytic grad) | **0.296** | **0.404** | **1.37** | ✓ | matched method |

**Twin caveats.** The section binary includes the ROHF warm-start, full
element-wise + translational checks, and unit conversion; the PySCF side is
UHF + UMP2 + `nuc_grad_method` only. Walls are short (\(\sim 0.3\)–\(0.4\) s).
The energy section `ump2-o2-sto3g` has a separate (energy-only) fair wall; do
not conflate the two ratios.

### Limitations

- Single textbook case: O₂ triplet, STO-3G cart, canonical UMP2 only.
- No UMP2 Hessian / vibrational analysis.
- Spherical (non-Cartesian) AO packs are not supported on this gradient path.
- CUDA UMP2 / UMP2 forces: not available.
- Multi-minima UHF for O₂ requires an explicit density guess; cold UHF may
  miss the PySCF minimum.

### References

- PySCF `pyscf.grad.ump2`; `mp.UMP2`

## romp2-o2-sto3g — ROMP2, triplet O₂ / STO-3G {#romp2-o2-sto3g}

### Summary

ROMP2 is second-order Møller–Plesset correlation on a **spin-pure ROHF**
reference: shared spatial orbitals, occupations \(n_\alpha / n_\beta\), and
semi-canonical spin-Fock orbital energies in the UMP2 doubles sum (PySCF
`mp.UMP2(ROHF)` convention). Use it when open-shell spin purity of the
reference matters and a full UHF/UMP2 path is not required. Roadmap:
[post-hf.md](../../roadmap/post-hf.md) — Restricted open-shell MP2.

### Method

- **Reference:** ROHF (Guest–Saunders / PySCF canonical Roothaan
  \(F_{\mathrm{eff}}\)). Shared spatial \(C\); \(n_\alpha=9\), \(n_\beta=7\).
- **Spin purity:** \(\langle S^2\rangle = S(S+1)\) on the ROHF determinant
  (here \(S=1 \Rightarrow \langle S^2\rangle=2\)).
- **Denominators:** not Roothaan eigenvalues. Semi-canonical
  \(\varepsilon_i^\sigma = (C^\dagger F_\sigma C)_{ii}\) from the **true** spin
  Focks \(F_\alpha\), \(F_\beta\) (Atomiverse: `F_alpha_uncoupled_` /
  `F_beta_uncoupled_`).
- **Correlation:** unprojected UMP2 doubles on that UHF-shaped view of the ROHF
  MOs:

$$
E_{\mathrm{corr}}^{\mathrm{ROMP2}} = E_{\mathrm{SS}} + E_{\mathrm{OS}}, \qquad
E_{\mathrm{tot}} = E_{\mathrm{ROHF}} + E_{\mathrm{corr}}.
$$

- **Not included:** ROMP2 analytic gradient/Hessian; frozen-core ROMP2;
  RI-ROMP2; spin-projected MP2; orbital-optimized ROMP2.

PySCF 2.14.0 has no separate `mp.ROMP2` class: `mp.UMP2(mf_rohf)` converts
ROHF→UHF (`addons.convert_to_uhf`) then runs raw UMP2. `MP2<ROHF>` mirrors
that path via the private `romp2_energy` kernel.

### Implementation

```cpp
ATV::cpu::scf::ROHF rohf{ATV::cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7};
rohf.tol = 1e-12;
rohf.max_iter = 200;
const ATV::cpu::scf::MP2Result mp2 =
    ATV::cpu::scf::MP2<ATV::cpu::scf::ROHF>{rohf}.run(frame);
// mp2.e_corr, mp2.e_tot [Ha]; rohf.energy() = E_ROHF [Ha]
```

| Path | Role |
| ---- | ---- |
| `rohf.h` | concrete ROHF method |
| `post_hf.h` | `MP2<Method>` |
| `mp2.h` | `MP2Result` |
| `mp2_pack.h` | private `romp2_energy` |

### Validation (vs PySCF)

Post-HF oracle: `pyscf.mp.UMP2(mf_rohf).run()`.

| Quantity | Pin (Ha) | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{ROHF}}\) | −147.632166990682 | 1e-8 Ha | reference consistency |
| \(E_{\mathrm{corr}}\) (ROMP2) | −0.096718716392 | **1e-6 Ha** | gate |
| \(E_{\mathrm{tot}}\) | −147.728885707074 | **1e-6 Ha** | gate; \(= E_{\mathrm{ROHF}}+E_{\mathrm{corr}}\) |

Cross-check: the ROHF energy matches the `o2_rohf_sto3g` section of
[scf/rohf](scf-rohf.md); UMP2 on UHF ([`ump2-o2-sto3g`](#ump2-o2-sto3g))
differs (\(E_{\mathrm{UHF}}\) lower, different \(E_{\mathrm{corr}}\)) because
the reference orbitals/energies differ.

### Performance

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `romp2` (O₂ / STO-3G) | 0.118 | 0.162 | **1.37** | ✓ | Median ≥3; `OMP_NUM_THREADS=1`; STO-3G wall is noise-dominated |

### Limitations

- Energy-only ROMP2 on CPU; **no** ROMP2 nuclear gradients or Hessians.
- No frozen-core ROMP2 (`frozen > 0` rejected); no RI-ROMP2.
- The canonical path materializes the full AO ERI tensor in private
  workspace — fine for STO-3G O₂; scales poorly without RI for larger bases.
- CUDA ROMP2 absent (`docs/roadmap/post-hf.md` CUDA ○).
- Open axes: projected/spin-adapted open-shell MP2 variants; OO-MP2; ROMP2
  response properties.

### References

- PySCF ROHF→UMP2 path: `pyscf.mp.UMP2` on `scf.ROHF` (v2.14.0)
- Guest–Saunders ROHF projectors / constrained-UHF view: see ROHF stage docs
  (`rohf_couple_stage.h`)

## mp3-h2-sto3g — RMP3, H₂ / STO-3G {#mp3-h2-sto3g}

### Summary

Canonical closed-shell third-order Møller–Plesset (RMP3) correlation energy on
a converged RHF reference: \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\). A cheap
post-HF step beyond MP2 when a full CCSD is not required; the gate is total
\(E_{\mathrm{corr}}\). The offline pin rebuilds Szabo–Ostlund / Psi4NumPy
diagrams on PySCF RHF orbitals.

### Method

- **Second order** (same as canonical RMP2 doubles):

$$
t_{ij}^{ab}=\frac{\langle ij\|ab\rangle}{D_{ijab}},\qquad
E^{(2)}=\frac14\sum_{ijab} t_{ij}^{ab}\,\langle ij\|ab\rangle,
$$

with \(D_{ijab}=\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b\) and
antisymmetrized spin-orbital integrals \(\langle pq\|rs\rangle\).

- **Third order:** residual \(u_{ijab}\) from ladder + ring (and partner)
  diagrams; \(E^{(3)}=\frac14\sum_{ijab} t_{ij}^{ab}\,u_{ijab}\). The spatial
  reduction of the same diagrams is the Psi4NumPy / Szabo–Ostlund 12-diagram
  sum used by the offline pin.
- **Total:**

$$
E_{\mathrm{corr}}^{\mathrm{MP3}}=E^{(2)}+E^{(3)},\qquad
E_{\mathrm{tot}}=E_{\mathrm{RHF}}+E_{\mathrm{corr}}.
$$

- **Spin:** closed-shell only (spin-orbital residual form; open-shell /
  GHF rejected).
- **Not included:** UMP3 / ROMP3; RI-MP3; SCS-MP3; analytic MP3 gradient /
  Hessian; orbital-optimized MP3; response properties. Frozen core is
  available (`MP3::frozen`, reported in `MP3Result::frozen`) but the pin uses
  `frozen=0`.

The \(E^{(2)}\) of the oracle cross-checks `mp.MP2` to machine precision.

### Implementation

```cpp
ATV::cpu::scf::RHF rhf{ATV::cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 100;
const ATV::cpu::scf::MP3Result mp3 =
    ATV::cpu::scf::MP3<ATV::cpu::scf::RHF>{rhf}.run(frame);
// mp3.e_corr_mp2, mp3.e_corr_mp3, mp3.e_corr, mp3.e_tot [Ha]
// rhf.energy() = E_RHF [Ha] after the internal fresh SCF
```

| Path | Role |
| ---- | ---- |
| `post_hf.h` | `MP3<Method>` |
| `mp3.h` | `MP3Result` public POD |
| `mp3_pack.h` | private `mp3_energy` |
| `src/cpu/scf/mp3.cpp` | spin-orbital doubles residual kernel |

### Validation (vs PySCF)

| Quantity | Pin (Ha) | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 | 1e-7 Ha | reference consistency |
| \(E^{(2)}\) (RMP2) | −0.013138073590 | 1e-6 Ha | diagnostic; matches `mp.MP2` |
| \(E^{(3)}\) | −0.004836072637 | 1e-6 Ha | diagnostic |
| \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\) | **−0.017974146227** | **1e-6 Ha** | gate |
| \(E_{\mathrm{tot}}^{\mathrm{MP3}}\) | −1.134733453623 | 1e-6 Ha | gate |

Cross-check (same geom/basis): FCI/CISD correlation −0.020524527092 Ha
(PySCF `fci.FCI` / `ci.CISD`); MP3 recovers ~87.6% of the correlation and
lies between MP2 and FCI.

### Performance

`OMP_NUM_THREADS=1`, `--method mp3 --geom h2 --basis sto-3g --repeat 5`.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `mp3` (H₂ / STO-3G) | 0.01602 | 0.01241 | **0.774** | ✓ | Median of 5; `OMP_NUM_THREADS=1`; oracle = RHF + diagram sum |

Absolute times are sub-20 ms; the ATV side includes RHF + MP3 + the section
asserts.

### Limitations

- Energy-only RMP3 on CPU; **no** MP3 nuclear gradient or Hessian.
- Closed-shell only: open-shell / GHF / UMP3 rejected (no silent fallback).
- No RI-MP3; full AO ERI materialization in private workspace — fine for
  STO-3G H₂; scales poorly without RI for larger bases / molecule sizes.
- No SCS or other empirical MPn reweightings at third order.
- CUDA MP3 absent.
- Open axes: efficient spatial (non-spin-orbital) third-order path;
  frozen-core cases; CCSD comparison ladder on the same pin; larger basis /
  multi-electron systems.

### References

- Szabo–Ostlund Table 6.2 / pp. 364–368 (third-order diagrams)
- Psi4NumPy `Moller-Plesset/MP3.py` (spatial diagram reduction used by the pin)

## mp4-h2-sto3g — RMP4(SDTQ), H₂ / STO-3G {#mp4-h2-sto3g}

### Summary

Canonical closed-shell fourth-order Møller–Plesset MP4(SDTQ) correlation
energy on a converged RHF reference:
\(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\) with
\(E^{(4)}=E_S^{(4)}+E_D^{(4)}+E_T^{(4)}+E_Q^{(4)}\). The next rung above RMP3
when full CCSD is not required; the gate is total \(E_{\mathrm{corr}}\). The
offline pin rebuilds intermediate-normalization RSPT through order 4 on PySCF
RHF orbitals.

### Method

- **Second and third order:** as in [`mp3-h2-sto3g`](#mp3-h2-sto3g)
  (canonical RMP2 doubles; doubles residual \(u_{ijab}\) with ladder + ring
  and partners).
- **Fourth order (SDTQ):** spin-orbital residual channels for singles (\(S\)),
  doubles (\(D\)), triples (\(T\)), and the disconnected-quadruples /
  renormalization piece

$$
E_Q^{(4)}=-E^{(2)}\,\|\Psi^{(1)}\|^2.
$$

  Total:

$$
E^{(4)}=E_S^{(4)}+E_D^{(4)}+E_T^{(4)}+E_Q^{(4)},\qquad
E_{\mathrm{corr}}^{\mathrm{MP4}}=E^{(2)}+E^{(3)}+E^{(4)},\qquad
E_{\mathrm{tot}}=E_{\mathrm{RHF}}+E_{\mathrm{corr}}.
$$

- **H₂ (2e) specialization:** \(n_{\mathrm{occ}}^{\mathrm{SO}}=2\), so true
  triples and connected quadruples vanish; \(E^{(4)}\) is carried by the
  doubles residual channel plus \(E_Q^{(4)}\).
- **Spin:** closed-shell only (open-shell / GHF rejected).
- **Not included:** UMP4 / ROMP4; RI-MP4; SCS/SOS reweightings; analytic MP4
  gradient / Hessian; orbital-optimized MP4; response properties. Frozen core
  is available (`MP4::frozen`, reported in `MP4Result::frozen`) but the pin
  uses `frozen=0`.

The oracle's \(E^{(2)}\) matches `mp.MP2` to machine precision and
\(E^{(2)}+E^{(3)}\) matches the MP3 pin.

### Implementation

```cpp
ATV::cpu::scf::RHF rhf{ATV::cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.max_iter = 100;
const ATV::cpu::scf::MP4Result mp4 =
    ATV::cpu::scf::MP4<ATV::cpu::scf::RHF>{rhf}.run(frame);
// mp4.e_corr_mp2, mp4.e_corr_mp3, mp4.e_corr_mp4 [Ha]
// mp4.e_corr_s / e_corr_d / e_corr_t / e_corr_q  (E4 channels)
// mp4.e_corr, mp4.e_tot [Ha]
```

| Path | Role |
| ---- | ---- |
| `post_hf.h` | `MP4<Method>` |
| `mp4.h` | `MP4Result` public POD |
| `mp4_pack.h` | private `mp4_energy` |
| `src/cpu/scf/mp4.cpp` | spin-orbital SDTQ residual kernel |

### Validation (vs PySCF)

| Quantity | Pin (Ha) | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 | 1e-7 Ha | reference consistency |
| \(E^{(2)}\) (RMP2) | −0.013138073590 | 1e-6 Ha | diagnostic; matches `mp.MP2` |
| \(E^{(3)}\) | −0.004836072637 | 1e-6 Ha | diagnostic; matches MP3 pin |
| \(E^{(4)}\) | −0.001711078792 | 1e-6 Ha | diagnostic; \(E_S+E_D+E_T+E_Q\) |
| \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\) | **−0.019685225019** | **1e-6 Ha** | gate |
| \(E_{\mathrm{tot}}^{\mathrm{MP4}}\) | −1.136444532416 | 1e-6 Ha | gate |

Cross-check (same geom/basis): FCI/CISD correlation −0.020524527092 Ha; MP4
recovers ~95.9% of the correlation and lies between MP3
(\(E_{\mathrm{corr}}=-0.017974146227\) Ha) and FCI.

### Performance

`OMP_NUM_THREADS=1`, `--method mp4 --geom h2 --basis sto-3g --repeat 3`;
`perf_exempt=false`.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `mp4` (H₂ / STO-3G) | 0.747 | 0.023 | **0.031** | ✓ | Median of 3; `OMP_NUM_THREADS=1`; oracle = det-space RSPT-4 twin on PySCF RHF+ao2mo |

The PySCF twin is the full determinant-space RSPT-4 rebuild (not a shipped
`mp.MP4` module), so its absolute time is higher than the algebraic ATV path.
The ATV side includes RHF + MP4 + the section asserts.

### Limitations

- Energy-only RMP4(SDTQ) on CPU; **no** MP4 nuclear gradient or Hessian.
- Closed-shell only: open-shell / GHF / UMP4 rejected (no silent fallback).
- No RI-MP4; full AO ERI materialization in private workspace.
- No SCS or other empirical MPn reweightings at fourth order.
- CUDA MP4 absent.
- The H₂ gate does not exercise the triples residual channel
  (\(n_{\mathrm{occ}}^{\mathrm{SO}}=2\)); multi-electron / larger-basis pins,
  an efficient spatial fourth-order path, frozen-core cases and a CCSD
  comparison ladder on the same pin ([`ccsd-h2-sto3g`](scf-coupled-cluster.md#ccsd-h2-sto3g))
  are open.

### References

- R. J. Bartlett, *Ann. Rev. Phys. Chem.* **32**, 359 (1981) (many-body PT /
  SDTQ channels)
