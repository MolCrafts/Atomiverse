# Periodic correlated methods on GPW (MP2, RPA, GW)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Post-SCF correlation and quasiparticle methods on a converged periodic GPW
Kohn–Sham reference: second-order Møller–Plesset perturbation theory (MP2,
SCS-MP2, UMP2), direct random-phase-approximation correlation energies (dRPA,
URPA) and one-shot spectral \(G_0W_0\) quasiparticle energies (GW, UGW). They
share one machinery: occupied–virtual MO repulsion integrals \((ia|jb)\) built
from pair densities on the GPW density grid with the plane-wave **RI-GPW**
identity, on real Γ-point orbitals or on complex Bloch orbitals of a
Monkhorst–Pack mesh, closed-shell (RKS) or unrestricted (UKS).

The public API is the concrete `cpu::gpw::GpwDft` (`cpu/gpw/gpw_solver.h`):
construct it with one `Kind` per element, the density cutoff [Ry] and the XC
functional; set members; `build(frame)` and `run(frame)`; then make one
post-SCF call on the converged orbitals — `mp2()` → `Mp2Result`
(`cpu/gpw/mp2.h`), `direct_rpa()` → `RpaResult` (`cpu/gpw/rpa.h`) or
`g0w0()` → `G0W0Result` (`cpu/gpw/g0w0.h`). The spin branch follows the SCF
(`n_occ` → RKS, `n_alpha`/`n_beta` → UKS) and the k branch follows `k_mesh`.
CPU backend only.

**Only the MP2 sections are run by the case:**
[periodic-ri-mp2-si](#periodic-ri-mp2-si) (Γ RI-GPW MP2) and
[periodic-scs-mp2-si](#periodic-scs-mp2-si) (Γ SCS-MP2), both pinned to an
offline CP2K 2026.1 `MP2_GPW` twin. The other sections — canonical Γ MP2,
multi-k MP2, UMP2, dRPA/URPA and G0W0/UGW — document variants of the same
machinery on the same two decks with values recorded offline; each carries the
"*No section of the case runs this variant.*" marker.

Case: `ctest -R '^gpw/periodic-mp2$'` — sources and oracle record in [`regressions/gpw/periodic-mp2/`](../../../regressions/gpw/periodic-mp2/). One section of the case per `##` heading below: `atv_regressions --case gpw/periodic-mp2 --section <name>`.

## Common setup

### Calling sequence

Every section follows the same shape; only the members and the post-SCF call
differ. Geometry is a `cpu::Frame` (positions [Å], Z in `kAtom_atomic_number`,
orthorhombic `box`); basis and pseudopotential come from the lookup
constructors `GaussianBasis(Element, name)` / `GthPotential(Element, name)`,
which throw on an unsupported element or name.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"   // GpwDft; includes mp2.h, rpa.h, g0w0.h
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"

using namespace ATV;
using namespace ATV::cpu::gpw;

const Kind si{molrs::Element::Si,
              GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
              GthPotential(molrs::Element::Si, "GTH-PBE-q4")};

// Γ deck: sc-Si₂
constexpr double a = 5.43;                    // cubic cell [Å]
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_atomic_number)[i] = 14;
    frame.span(kAtom_x)[i] = frame.span(kAtom_y)[i] =
        frame.span(kAtom_z)[i] = i * a / 4.0;
}
frame.box = cpu::Box::cubic(a);
GpwDft m{{si}, /*cutoff_ry=*/30.0, XcKind::PBE};

// Multi-k deck: bct diamond Si₄ (a_t = a/√2, c = a)
const double at = a / std::sqrt(2.0);
const double frac[4][3] = {{0, 0, 0}, {0.5, 0.5, 0.5}, {0.5, 0, 0.25}, {0, 0.5, 0.75}};
cpu::Frame frame4(4);
for (int i = 0; i < 4; ++i) {
    frame4.span(kAtom_atomic_number)[i] = 14;
    frame4.span(kAtom_x)[i] = frac[i][0] * at;
    frame4.span(kAtom_y)[i] = frac[i][1] * at;
    frame4.span(kAtom_z)[i] = frac[i][2] * a;
}
frame4.box = cpu::Box::orthorhombic(at, at, a);
GpwDft mk{{si}, /*cutoff_ry=*/80.0, XcKind::PBE};

// ... set members (per section), then
m.build(frame);
const GpwResult& scf = m.run(frame);   // SCF [Ha] on the result, eV on kEnergy
const Mp2Result& r = m.mp2();          // or m.direct_rpa() / m.g0w0()
```

Members shared by the three post-SCF calls:

| Member | Meaning |
| ------ | ------- |
| `n_occ` / `n_alpha`, `n_beta` | RKS doubly occupied count / UKS occupations (`n_alpha + n_beta > 0` selects UKS; `n_occ` is then ignored) |
| `k_mesh[3]`, `k_time_reversal` | Monkhorst–Pack mesh; `k_time_reversal = false` keeps all points at equal weight (CP2K `FULL_GRID .TRUE.` / `SYMMETRY .FALSE.`) |
| `mp2_ri` | `true` (default): product RI-GPW \((ia|jb)\); `false`: Poisson-per-pair path (Γ only). Isolated (`nonperiodic`) cells always use the Poisson path |
| `mp2_n_virt` | virtual-space cap (per spin for UKS); 0 = all kept virtuals |
| `mp2_os_scale`, `mp2_ss_scale` | Grimme scales applied to `Mp2Result::e_corr` (default 1.0) |

Each call runs at the geometry the last `run()` converged on and returns an
owned result valid until the next call of the same method. Guards (throw, no
silent fallback): `std::logic_error` before `build()` or a converged `run()`,
or when UKS β orbitals are missing; `std::invalid_argument` for `use_ot` (OT
stores no canonical virtuals), a multi-k reference without stored `C_k` /
`C_k_beta`, multi-k with `mp2_ri = false`, an empty virtual space after the
cap, or a vanishing (minimum) HOMO–LUMO gap \(< 10^{-6}\) Ha;
`std::runtime_error` from `direct_rpa()` / `g0w0()` when the dRPA reduction is
ill-conditioned. Build with `-DATV_BUILD_GPW=ON` (with HF + PW, FFTW3; CPU environment `buildenv-gcc/2026.03-mpich`). Correlation energies are reported in **Hartree** on the
result objects (QP gaps, IP and EA in eV); public Frame single-point energies
stay eV on `kEnergy`. Kernels are private (`cpu/gpw/internal/mp2_kernels.h`).

| Path | Role |
| ---- | ---- |
| `cpu/gpw/gpw_solver.h` | `GpwDft::{mp2, direct_rpa, g0w0}` and the members above |
| `cpu/gpw/mp2.h`, `rpa.h`, `g0w0.h` | `Mp2Result`, `RpaResult`, `G0W0Result` value types |
| `src/cpu/gpw/mp2.cpp`, `rpa.cpp`, `g0w0.cpp` | dispatch (RI vs Poisson, Γ vs multi-k, RKS vs UKS) |
| `src/cpu/gpw/internal/mp2_kernels.cpp` | MO-on-grid, RI-GPW \((ia|jb)\), doubles sums, plasmon and spectral kernels |
| `tests/cpu/gpw/test_cpu_gpw_g0w0_kpoints.cpp` | unit tier: multi-k G0W0 kernels invariant under Bloch-MO phases (no full SCF); the MP2 and RPA kernels have no unit test in the tree |

### Systems

Two decks serve every section.

**Γ deck — artificial sc-Si₂** (MP2, SCS-MP2, UMP2, dRPA, URPA, G0W0, UGW at Γ):

| Item | Setting |
| ---- | ------- |
| Crystal | artificial sc-Si₂ (diamond-fragment motif), charge 0 |
| Cell | cubic \(a = 5.43\) Å, periodic XYZ, Γ only |
| Positions [Å] | Si \((0,0,0)\); Si \((a/4,a/4,a/4)=(1.3575,1.3575,1.3575)\) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 (zion 4) → 8 valence e⁻ → \(n_{\mathrm{occ}}=4\) (UKS \(n_\alpha=n_\beta=4\)); \(n_{\mathrm{virt}} = 22\) on this basis |
| XC | PBE (`XcKind::PBE`; RPBE in [ugw-uks-rpbe](#ugw-uks-rpbe)) |
| Cutoff | **30 Ry**, single grid (SCF + `WFC_GPW`) |
| SCF | tol \(10^{-6}\) Ha, mix 0.3, DIIS, **level_shift 0.3 Ha**; `max_iter` 200 in the case sections, 80 in the recorded variants |

Γ-only sc-Si₂ is a small-gap crystal (unshifted KS gap \(\approx 0.015\) Ha):
without the level shift the SCF oscillates for hundreds of iterations. The
shift is a convergence aid only — canonical eigenvalues enter every MP2 / RPA /
GW denominator **unshifted** — and the gap is far above the \(10^{-6}\) Ha
degeneracy guard. The deck does not converge on the k-path (which rejects a
nonzero `level_shift`), so the multi-k sections use the physical bct cell.
Physical diamond Si₈ at production cutoffs exceeds the regression wall budget.

Closed-shell Si₂ is the π² case: the core-Hamiltonian guess has an exactly
degenerate HOMO, so floating-point noise picks the occupied π combination. A
1e-9 Å displacement of atom 2 flips the iteration-0 energy between
−4.56842019 and −4.56824863 Ha and the converged state between two stationary
points, E_SCF ≈ −7.5955445 Ha (≈ 58 iterations) and ≈ −7.5955020 Ha (110–135
level-shifted iterations, the branch the x86 OpenBLAS build takes;
\(\Delta E \approx 4\times10^{-5}\) Ha). Every gate of the case holds on both
branches; `max_iter = 200` keeps the budget independent of the branch.

**Multi-k deck — diamond Si, bct 4-atom cell** (multi-k MP2, UMP2, dRPA, URPA, G0W0, UGW; same geometry as `gpw-kpoints-si`):

| Item | Setting |
| ---- | ------- |
| Cell | \(a_t = a/\sqrt{2}\), \(c = a\); \(a = 5.43\) Å |
| Positions (frac) | \((0,0,0)\), \((\tfrac12,\tfrac12,\tfrac12)\), \((\tfrac12,0,\tfrac14)\), \((0,\tfrac12,\tfrac34)\) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 → 16 valence e⁻ → \(n_{\mathrm{occ}}=8\) (UKS \(n_\alpha=n_\beta=8\)) |
| XC / cutoff | PBE, **80 Ry**, `eps_default` \(10^{-7}\) |
| k | MP **2×1×1**, `k_time_reversal = false` → \(N_k = 2\), \(w_k = 1/2\) |
| SCF | DIIS depth 6, tol \(10^{-7}\), mix 0.3, `max_iter` 60, **no** level shift |
| Correlation | `mp2_ri = true` (required on multi-k), `mp2_n_virt = 2` (per spin for UKS) for the ≤ 60 s budget; CP2K twin `ADDED_MOS 2` |

```cpp
mk.n_occ = 8;               // or mk.n_alpha = mk.n_beta = 8 for UKS
mk.max_iter = 60;
mk.tol = 1e-7;
mk.mix = 0.3;
mk.level_shift = 0.0;       // k-path rejects nonzero
mk.eps_default = 1e-7;
mk.diis_depth = 6;
mk.k_mesh[0] = 2; mk.k_mesh[1] = 1; mk.k_mesh[2] = 1;
mk.k_time_reversal = false; // equal weights
mk.mp2_ri = true;           // required on multi-k
mk.mp2_n_virt = 2;          // e2e virtual cap
```

Sampled points follow the CP2K `SCHEME MONKHORST-PACK` convention
\(\kappa_d = (2r - N_d - 1)/(2N_d)\), \(r = 1,\ldots,N_d\) (even \(N\) is
Γ-avoiding). The multi-k SCF stores complex Bloch coefficients
`GpwResult::C_k` / `C_k_beta` and eigenvalues `eps_k` / `eps_k_beta`. On this
deck the multi-k PBE SCF converges in 7 iterations to
\(E_{\mathrm{SCF}} = -15.5751782413\) Ha (RKS and closed-shell UKS alike) with a
minimum direct KS gap of 1.561168 eV.

### Shared physics

**RI-GPW \((ia|jb)\).** Pair densities \(\rho_{ia}=\psi_i\psi_a\) (complex
Bloch products \(\psi_i^{k_i}[\psi_a^{k_a}]^*\) on multi-k) are collocated on
the SCF density grid. Occupied–virtual pairs are **neutral**
(\(\int \rho_{ia} = S_{ia} = 0\)), so the periodic Coulomb operator with the
G=0 component dropped is divergence-free:

\[
\tilde\rho_{ia}(\mathbf G) = \mathrm{FFT}[\rho_{ia}],
\qquad
B_{ia}(\mathbf G) = \sqrt{\frac{4\pi}{|\mathbf G|^2}}\,\tilde\rho_{ia}(\mathbf G)
\quad (\mathbf G \neq 0),
\qquad
(ia|jb) = \frac{w}{N_g}\sum_{\mathbf G} B_{ia}^*(\mathbf G)\,B_{jb}(\mathbf G)
\]

(equivalently: IFFT \(V_{ia}\) from \(4\pi\tilde\rho_{ia}/|\mathbf G|^2\) and
real-space quadrature \(\sum_g w\, V_{ia}(g)\,\rho_{jb}(g)\); three-index
memory \(\mathcal O(n_{\mathrm{ov}} N_g)\)). This is the Del Ben–Hutter–
VandeVondele layout (CP2K `&WF_CORRELATION` `METHOD MP2_GPW` + `&WFC_GPW`) with
the plane-wave identity in place of a Gaussian cFIT / `RI_AUX` fit: accuracy is
limited by the density-grid cutoff. Multi-k pair densities use a linear real
Poisson solve on Re/Im. The Γ Poisson path (`mp2_ri = false`) solves
\(\hat\nabla^2 V_{ia} = -4\pi\rho_{ia}\) once per pair with the same kernel and
agrees with RI-GPW to \(10^{-10}\) Ha.

**Reference orbitals.** Correlation is evaluated on **KS (PBE) orbitals** —
MP2@KS, dRPA@KS, \(G_0W_0\)@KS, the standard periodic double-hybrid / RPA
ingredient. A true MP2@HF or EXX-based RPA total energy needs periodic exact
exchange (blocked row `periodic-hybrid-gpw`).

**dRPA kernel** (shared by RPA and the GW screening). Singlet Coulomb-only
Casida (\(\alpha_{\mathrm{hf}} = 0\)):

\[
A_{ia,jb} = (\varepsilon_a-\varepsilon_i)\,\delta_{ij}\delta_{ab} + 2\,(ia|jb),
\qquad
B_{ia,jb} = 2\,(ia|jb),
\]

with positive roots \(\Omega_s\) of the Hermitian reduction
\((A-B)^{1/2}(A+B)(A-B)^{1/2}\) and the Furche plasmon correlation energy

\[
E_c^{\mathrm{dRPA}} = \tfrac12\Bigl(\sum_s \Omega_s - \mathrm{Tr}\,A\Bigr),
\qquad
E_{\mathrm{total}} = E_{\mathrm{SCF}} + E_c \;(\text{dRPA@KS, no EXX replacement}).
\]

UKS (URPA) uses the composite α→α ⊕ β→β space with spins explicit (no
closed-shell factor 2). Multi-k uses one Casida block per momentum transfer
\(q\) on the closed MP mesh (transitions \(i_k\to a_{k+q}\), Coulomb at \(q\),
G=0 dropped only at \(q=0\)) and
\(E_c = \sum_q w_q\,\tfrac12(\sum_s\Omega_{q,s} - \mathrm{Tr}\,A_q)\).

**Spectral \(G_0W_0\)** (shared by the GW and UGW sections). One-shot: \(G\)
and \(W\) from the converged KS solution, no self-consistency. \(W\) is the
dRPA screening above; the diagonal correlation self-energy
\(\mathrm{Re}\,\Sigma_c(\omega)\) is the frequency-exact sum over RPA poles and
transition densities (same form as molecular `G0W0<RHF>::run` / PySCF
`GWExact`), with spectral broadening \(\eta = 10^{-8}\) Ha (`kEta` in
`src/cpu/gpw/g0w0.cpp`). Exact-exchange diagonals \(\Sigma_{x,p}\) use
**Martyna–Tuckerman truncated Coulomb**, so the path is G=0-safe without a
periodic-HF operator; \(v_{\mathrm{xc},p}\) is the SCF functional's local
potential on the (BZ-weighted) density. Each MO is solved by Newton iteration
(residual tol \(10^{-8}\) Ha), **not** the linearized \(Z\)-factor form:

\[
\omega = \varepsilon_p + \mathrm{Re}\,\Sigma_{c,p}(\omega) + \Sigma_{x,p} - v_{\mathrm{xc},p}.
\]

A finite KS HOMO–LUMO gap (\(\ge 10^{-6}\) Ha) keeps \((A-B)\) non-singular.
Multi-k uses finite-\(q\) inter-\(k\) \(W\); UKS uses same-spin
\(G_\sigma W\), per-spin MT \(\Sigma_x^\sigma\) and spin-resolved
\(v_{\mathrm{xc}}^\sigma\), and reports the α channel. MT exchange together
with a tiny virtual space can place absolute QP levels above zero, so the
**gap** is the hard observable, not the absolute HOMO/LUMO positions.

| `G0W0Result` field | Meaning | Unit |
| ------------------ | ------- | ---- |
| `qp_energy_ha` / `mf_energy_ha` | QP / KS orbital energies (every MO of the reported spectrum; report \(k\) = that of the minimum direct QP gap) | Ha |
| `sigma_x_ha` / `vxc_ha` | \(\Sigma_{x,p}\) / \(v_{\mathrm{xc},p}\) | Ha |
| `gap_ev` / `mf_gap_ev` | (minimum direct) QP / KS HOMO→LUMO gap | eV |
| `ip_ev` / `ea_ev` | \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) / \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | eV |
| `n_occ` / `n_virt` / `n_kpts` / `ri` / `converged` | counts (UKS: \(n_\alpha\) and \(n_{\mathrm{virt}}^\alpha+n_{\mathrm{virt}}^\beta\)); mesh size (0 at real Γ); RI path; all Newton solves converged | — |

`RpaResult` carries `e_corr`, `e_scf`, `e_total`, `tr_A` [Ha], `n_occ`,
`n_virt` (spin-orbital totals for URPA), `ri`, `n_kpts` and `omega_ha` (all
roots, ascending; multi-k roots of all \(q\) blocks concatenated). `Mp2Result`
carries `e_corr`, unscaled `e_corr_os` / `e_corr_ss`, `e_scf`, `e_total`,
`n_occ`, `n_virt`, `ri`, `n_kpts`.

### Oracle and gates

- **CP2K 2026.1, offline only** (`cp2k.psmp`, host arrhenius1 EPYC, 1 MPI
  rank × OMP=4; rules in `tools/cp2k_oracle/README.md`). CP2K is never
  launched from CMake or ctest. Twin inputs in tree:
  `tools/cp2k_oracle/inputs/periodic-ri-mp2-si.inp`,
  `tools/cp2k_oracle/inputs/periodic-scs-mp2-si.inp`. Operator environment for
  re-pins:

  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  mpirun -np 1 cp2k.psmp -i <case>.inp -o <case>.out
  ```

- **MP2 is a same-algorithm twin.** CP2K `MP2_GPW` assembles \((ia|jb)\) the
  RI-GPW way, so the case sections pin absolute \(E_{\mathrm{corr}}\) to CP2K
  digits at \(10^{-3}\) Ha (grid/SCF twin band) and report a fair wall ratio.
- **RPA and GW are not.** CP2K's default `&RI_RPA` / `&RI_G0W0` paths are
  frequency-integrated (minimax quadrature, Gaussian `RI_AUX`, EXX / periodic
  HF in the total energy or GW step) — not the exact plasmon formula or the
  spectral \(G_0W_0\) with MT exchange on PW-RI \((ia|jb)\). Printed CP2K
  numbers are a near-twin diagnostic, never a hard digit gate, and no CP2K
  digits are filled in without a same-algorithm twin. These variants are gated
  internally (convergence, signs, identities, spectrum sizes, RI ≡ Poisson)
  plus a frozen Atomiverse band or a UKS ≡ RKS dual gate, and their speed is
  **`perf_exempt`**. The multi-k MP2 / UMP2 and Γ UMP2 variants were recorded
  without a CP2K binary on the host and are likewise `perf_exempt`.
- **Fair wall** when a method-class-matched twin exists:
  \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on the same host and core
  count.
- **Unit tier** (not e2e): closed-shell doubles sum ≡ explicit spin-orbital MP2
  sum (\(10^{-12}\)); MO-on-grid ≡ Γ-image-summed AO contraction
  (\(10^{-9}\)); \((pp|pp)\) ≡ \(2\times\) Poisson Hartree energy of the same
  pair density (\(10^{-10}\)); `mo_on_grid_k` at \(\kappa=0\) ≡ real
  `mo_on_grid`; complex OS/SS formula ≡ real `mp2_rhf_energy`; UMP2 ≡
  spin-orbital sum and ≡ RMP2 on identical α=β tensors; cross-spin RI ≡
  Poisson; plasmon kernel ≡ dense A/B path; multi-k kernels at \(N_k = 1\) ≡
  the real Γ plasmon / spectral QP; `direct_rpa_uhf` ≡ `direct_rpa_rhf` on α=β
  tensors; guard throws.

### Shared limitations

- **CPU only**; CUDA GPW correlation is not implemented.
- **@KS references only** — no periodic exact exchange (MP2@HF, EXX-based RPA
  totals, periodic-HF \(\Sigma_x\) all blocked on `periodic-hybrid-gpw`).
- **Energy / QP levels only** — no analytic forces or stress for MP2, RPA or GW.
- **Plane-wave RI identity**, not Gaussian cFIT / `RI_AUX` tables.
- **No OT reference** — canonical virtuals need diagonalization.
- **Capability decks**, not materials numbers: sc-Si₂ at 30 Ry and bct Si₄ at
  80 Ry with `mp2_n_virt = 2`; production work needs diamond Si₈, higher
  cutoffs, denser meshes and the full virtual space.
- GW: spectral one-shot only — no analytic-continuation or
  frequency-integrated GW, no evGW / qsGW / scGW, no BSE.
- RPA: correlation-energy products — not TDHF/TDDFT excitation drivers, not
  RPAX / SOSEX / ACFDT.

### References

- M. Del Ben, J. Hutter, J. VandeVondele, *J. Chem. Theory Comput.* **8**, 4177
  (2012) — RI-GPW MP2 / CP2K `MP2_GPW` (shared \((ia|jb)\) layout).
- C. Møller, M. S. Plesset, *Phys. Rev.* **46**, 618 (1934) — MP2.
- F. Furche, *J. Chem. Phys.* **114**, 5982 (2001), DOI 10.1063/1.1353585 —
  dRPA plasmon formula / Casida reduction.
- X. Ren et al., *New J. Phys.* **14**, 053020 (2012) — RPA in condensed matter.
- M. S. Hybertsen, S. G. Louie, *Phys. Rev. B* **34**, 5390 (1986), DOI
  10.1103/PhysRevB.34.5390 — \(G_0W_0\).
- J. Wilhelm, M. Del Ben, J. Hutter, *J. Chem. Theory Comput.* **12**, 3623
  (2016) — CP2K RI-GW family (frequency-integrated near-twin class).
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  — PBE.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH.
- M. Krack, *Theor. Chem. Acc.* **114**, 145 (2005) — GTH tables.
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.
- Narrative: [GPW chapter](../ch09-gpw.md); molecular correlation
  [chapter](../ch07-correlation.md); molecular twins
  [g0w0-h2-sto3g](scf-green-function.md#g0w0-h2-sto3g),
  [direct-rpa-h2-sto3g](scf-green-function.md#direct-rpa-h2-sto3g),
  [ump2-o2-sto3g](scf-mp2.md#ump2-o2-sto3g); roadmap
  [`docs/roadmap/dft.md`](../../roadmap/dft.md) §10.

## periodic-ri-mp2-si — Γ-point RI-GPW MP2 — sc-Si₂ (PBE reference) {#periodic-ri-mp2-si}

Closed-shell MP2 on the converged Γ-point GPW PBE SCF of the sc-Si₂ deck, with
\((ia|jb)\) from the product RI-GPW path. The section also runs the Poisson
path (`mp2_ri = false`, [periodic-mp2-si](#periodic-mp2-si)) on the same
orbitals: same Coulomb operator, product intermediate layout, identical to
\(10^{-10}\) Ha. Absolute correlation energy is pinned to CP2K 2026.1
`MP2_GPW`.

### Method

Canonical closed-shell doubles sum with OS/SS decomposition (same convention
as molecular `cpu::scf::MP2Result`):

\[
E^{(2)} = -\sum_{ijab}
  \frac{(ia|jb)\,[\,2(ia|jb) - (ib|ja)\,]}
       {\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j}
  = E_{\mathrm{OS}} + E_{\mathrm{SS}}.
\]

Settings: Γ deck, `max_iter = 200`, all kept virtuals (`mp2_n_virt = 0`),
OS/SS scales 1.0.

```cpp
m.n_occ = 4;
m.max_iter = 200;             // degenerate-guess branch budget (see Systems)
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;          // convergence aid; eps unshifted in MP2
m.mp2_ri = true;              // product default
m.build(frame);
const GpwResult& scf = m.run(frame);

m.mp2_n_virt = 0;             // all kept virtuals
const Mp2Result r_ri = m.mp2();   // r_ri.ri == true
m.mp2_ri = false;
const Mp2Result r_can = m.mp2();  // Poisson path, same orbitals; r_can.ri == false
// r_ri.e_corr (< 0) [Ha]; e_corr_os + e_corr_ss == e_corr; e_total == e_scf + e_corr
```

### Validation

CP2K deck: same geometry, basis, PP, PBE, 30 Ry (`MGRID` + `WFC_GPW`),
`EPS_SCF 1e-6`, `LEVEL_SHIFT 0.3`, direct \(P\)-mixing \(\alpha=0.3\),
`ADDED_MOS 22`, `METHOD MP2_GPW`.

| Quantity | Atomiverse gate | CP2K 2026.1 | Tol |
| -------- | --------------- | ----------: | --- |
| \(E_{\mathrm{corr}}\) (RI-GPW) | within band of pin | **−0.276087740659 Ha** | **\(1\times 10^{-3}\) Ha** |
| \(E_{\mathrm{corr}}\) OS (CP2K SO) | — | −0.237578621868 Ha | — |
| \(E_{\mathrm{corr}}\) SS | — | −0.038509118792 Ha | — |
| \(E_{\mathrm{SCF}}\) (PBE) | ~−7.596 Ha (soft; grid/SCF twin) | −7.595338517019 Ha | — |
| \(E_{\mathrm{total}}\) | \(E_{\mathrm{SCF}}+E_{\mathrm{corr}}\) | −7.871426257678 Ha | — |
| \(\lvert E_{\mathrm{corr}}(\mathrm{RI})-E_{\mathrm{corr}}(\mathrm{Poisson})\rvert\) | 0 (same kernel) | — | **\(1\times 10^{-10}\) Ha** |
| OS + SS ≡ \(E_{\mathrm{corr}}\); \(E_{\mathrm{tot}}\equiv E_{\mathrm{SCF}}+E_{\mathrm{corr}}\) | exact | — | \(10^{-12}\) |
| `Mp2Result::ri` | `true` on RI, `false` on Poisson | — | — |
| SCF | converged; \(E_{\mathrm{corr}}<0\), finite | — | hard |

### Performance

| Item | Value |
| ---- | ----- |
| \(T_{\mathrm{CP2K}}\) (full QS ENERGY, SCF ~3.5 min + `MP2_GPW` ~369 s) | **576.7 s** |
| \(T_{\mathrm{ATV}}\) (SCF + RI MP2 + Poisson cross-check, OMP=4) | **~16–30 s** (typical) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **~0.03–0.05** raw; **~0.09** recorded → within 10× (`perf_exempt` false) |

The Atomiverse wall includes the intentional second full MP2 (Poisson
cross-check); the deck is tiny (30 Ry, 2 atoms), so the ratio is not a
production-size claim.

### Limitations

Γ only for this section (multi-k: [periodic-mp2-kpoints](#periodic-mp2-kpoints));
closed-shell (UKS: [periodic-ump2-si](#periodic-ump2-si)).

## periodic-scs-mp2-si — Γ-point RI-GPW SCS-MP2 — sc-Si₂ (PBE reference) {#periodic-scs-mp2-si}

Spin-component-scaled MP2 on the same orbitals and RI-GPW path as
[periodic-ri-mp2-si](#periodic-ri-mp2-si), with Grimme's scales
(*J. Chem. Phys.* **118**, 9095 (2003)):

\[
E_{\mathrm{corr}}^{\mathrm{SCS}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}} + c_{\mathrm{SS}}\,E_{\mathrm{SS}},
\qquad c_{\mathrm{OS}} = 1.2,\; c_{\mathrm{SS}} = 1/3.
\]

```cpp
m.mp2_os_scale = 1.2;
m.mp2_ss_scale = 1.0 / 3.0;
const Mp2Result& r = m.mp2();
// r.e_corr_os / r.e_corr_ss stay unscaled; r.e_corr = 1.2 * OS + (1/3) * SS
```

CP2K twin keywords: `SCALE_S 1.2` (singlet / OS), `SCALE_T 1/3` (triplet /
SS) under `&WF_CORRELATION`; SCF `EPS_SCF 1e-4`, Broyden α = 0.2,
`LEVEL_SHIFT 0.5`, `ADDED_MOS 22` (the small-gap residual plateaus at
~2–4e-5 under diagonalization; energy stable to ~1e-5 Ha versus tighter
attempts). Section gates: \(E_{\mathrm{corr}}^{\mathrm{SCS}}\) =
**−0.297941206394 Ha** ± \(10^{-3}\); \(e_{\mathrm{corr}} \equiv
1.2\,\mathrm{OS} + \tfrac13\,\mathrm{SS}\) (\(10^{-12}\)); unscaled OS/SS
invariant under the scale members versus plain MP2 on the same orbitals
(\(10^{-10}\)); plain OS+SS and \(e_{\mathrm{total}}\) identities
(\(10^{-12}\)); `ri == true`; SCF converged, \(E_{\mathrm{corr}}\) finite and
negative. CP2K printed SO −0.23758796717559, SS −0.03850693735043, total
SCS −0.29794120639418, QS total −7.893970778454608 Ha (SCF
−7.596029572060 Ha); unscaled SO+SS = −0.276094904526 Ha matches the plain
pin to ~7e-6 Ha. Atomiverse gives \(E_{\mathrm{corr}}^{\mathrm{SCS}} =
-0.2982932565\) Ha (\(|\Delta|\) = 3.5e-4 Ha). Fair wall: CP2K full twin
17.81 s versus Atomiverse SCF + SCS-MP2 7.55 s (SCF 7.34 + SCS 0.22, 8
threads) → \(r = 0.424\). Full record in the case `REFERENCE.md`.

## periodic-mp2-si — Γ-point GPW MP2, Poisson path — sc-Si₂ (PBE reference) {#periodic-mp2-si}

*No section of the case runs this variant.*

The canonical (non-RI) Γ MP2 path, selected with `mp2_ri = false` (CP2K
`METHOD DIRECT_CANONICAL` class): collocate \(\rho_{ia}\) on the density grid,
solve the periodic FFT Poisson equation once per pair, and contract with
\(\rho_{jb}\) by grid quadrature,

\[
(ia|jb) = \sum_g w\, V_{ia}(\mathbf r_g)\, \rho_{jb}(\mathbf r_g),
\qquad
\hat\nabla^2 V_{ia} = -4\pi \rho_{ia},
\]

on real lattice-summed AOs \(\chi_\mu(\mathbf r) = \sum_{\mathbf T}
\varphi_\mu(\mathbf r - \mathbf T)\). Exact within the grid resolution for a
tiny cell; cost \(n_{\mathrm{occ}} n_{\mathrm{virt}}\) Poisson solves +
\(\mathcal O(n_{\mathrm{ov}}^2 N_g)\) quadratures. The same doubles sum as
[periodic-ri-mp2-si](#periodic-ri-mp2-si); the case checks this path against
RI-GPW at 1e-10 Ha.

```cpp
m.n_occ = 4;
m.level_shift = 0.3;
m.mp2_ri = false;              // Poisson per pair
m.build(frame);
m.run(frame);
m.mp2_n_virt = 0;
const Mp2Result& r = m.mp2();
```

Recorded values (Γ deck, `max_iter` 80): SCF converged; \(E_{\mathrm{corr}} <
0\); virtual-space monotonicity \(E_{\mathrm{corr}}(22) \le
E_{\mathrm{corr}}(8) < 0\); OS+SS ≡ \(E_{\mathrm{corr}}\); frozen band
\(E_{\mathrm{corr}} = -0.2764\) Ha ± 1e-3 (full value −0.2764036524 Ha, the
reference of the UMP2 dual gate). `perf_exempt` (no CP2K binary on the
recording host).

## periodic-mp2-kpoints — Multi-k RI-GPW MP2 — bct Si₄, MP 2×1×1 {#periodic-mp2-kpoints}

*No section of the case runs this variant.*

Momentum-conserving closed-shell MP2 on complex Bloch MOs of the multi-k PBE
SCF (CP2K `MP2_GPW` + `&KPOINTS`). Equal-weight sum over the sampled mesh:

\[
E^{(2)} = \frac{1}{N_k}\sum_{k_i k_j k_a}\sum_{ijab}
  \frac{(ia|jb)_{k}\,\bigl[2(ia|jb)_{k}-(ib|ja)_{k}\bigr]}
       {\varepsilon_a^{k_a}+\varepsilon_b^{k_b}-\varepsilon_i^{k_i}-\varepsilon_j^{k_j}}
  = E_{\mathrm{OS}}+E_{\mathrm{SS}},
\]

with \(k_b\) fixed by crystal-momentum conservation from \((k_i,k_j,k_a)\).
Multi-k requires `mp2_ri = true` (no multi-k Poisson path). A vanishing
minimum HOMO–LUMO gap on the mesh throws.

```cpp
mk.n_occ = 8;                 // multi-k deck members (Common setup)
mk.build(frame4);
mk.run(frame4);               // stores C_k / eps_k / k_frac / weights
const Mp2Result& r = mk.mp2();   // r.ri && r.n_kpts == 2
```

| Quantity | Atomiverse (OMP=4) | Tol | Notes |
| -------- | -----------------: | --- | ----- |
| \(n_{\mathrm{kpts}}\); \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) | 2; 8 / 2 | exact | `C_k.size() == n_kpts`; CP2K `ADDED_MOS 2` |
| \(E_{\mathrm{SCF}}\) | −15.5751782413 Ha (7 iters) | setup | |
| \(E_{\mathrm{corr}}\) | **−0.0657637511 Ha** | **1e-4** | frozen Atomiverse band; CP2K digits pending an offline run |
| \(E_{\mathrm{corr}}^{\mathrm{OS}}\) / \(E_{\mathrm{corr}}^{\mathrm{SS}}\) | −0.0598553603 / −0.0059083908 Ha | 1e-12 identity | unscaled |
| \(E_{\mathrm{total}}\) | −15.6409419924 Ha | 1e-12 | \(E_{\mathrm{SCF}}+E_c\) |
| Wall (SCF + MP2) | **3.6 s** | ≤ 60 s | `perf_exempt` (no CP2K binary on host) |

No UMP2 here ([periodic-ump2-kpoints](#periodic-ump2-kpoints)); absolute CP2K
digits are pending an offline re-pin.

## periodic-ump2-si — Γ-point GPW UMP2 — sc-Si₂ (UKS PBE reference) {#periodic-ump2-si}

*No section of the case runs this variant.*

Unrestricted MP2 on a Γ UKS PBE reference with equal-spin occupations
\(n_\alpha = n_\beta = 4\) — the sc-Si₂ deck re-hosted as UKS so the UMP2
kernel is exercised against an identity: when \(C^\alpha\approx C^\beta\),
**UMP2 ≡ RMP2**. With \(v_{ia,jb}^{\sigma\sigma'}=(i_\sigma a_\sigma|j_{\sigma'} b_{\sigma'})\)
(molecular / PySCF UMP2 convention):

\[
E_{\mathrm{SS}}^\sigma
  =\frac14\sum_{ijab}
   \frac{\big[v_{ia,jb}^{\sigma\sigma}-v_{ib,ja}^{\sigma\sigma}\big]^2}
        {\varepsilon_a^\sigma+\varepsilon_b^\sigma-\varepsilon_i^\sigma-\varepsilon_j^\sigma},
\qquad
E_{\mathrm{OS}}
  =\sum_{ijab}
   \frac{\big(v_{ia,jb}^{\alpha\beta}\big)^2}
        {\varepsilon_a^\alpha+\varepsilon_b^\beta-\varepsilon_i^\alpha-\varepsilon_j^\beta},
\]

\[
E_{\mathrm{corr}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}} + c_{\mathrm{SS}}\,(E_{\mathrm{SS}}^\alpha+E_{\mathrm{SS}}^\beta),
\qquad E_{\mathrm{tot}} = E_{\mathrm{SCF}} + E_{\mathrm{corr}}.
\]

`e_corr_os` / `e_corr_ss` are unscaled; `n_occ` / `n_virt` are spin-orbital
totals \(n_\alpha+n_\beta\) and \(n_{v\alpha}+n_{v\beta}\); `mp2_n_virt`
caps each spin. Three tensors are built (αα, ββ, αβ).

```cpp
m.n_alpha = 4;                 // UKS, equal-spin closed shell
m.n_beta  = 4;
m.level_shift = 0.3;
m.mp2_ri = true;
m.build(frame);
m.run(frame);
m.mp2_n_virt = 0;              // all kept virtuals per spin
const Mp2Result& r = m.mp2();  // r.n_occ == 8; r.ri == true; r.n_kpts == 0
```

| Quantity | Value [Ha] | Tol |
| -------- | ---------: | --- |
| \(E_{\mathrm{corr}}\) reference (closed-shell Γ MP2, [periodic-mp2-si](#periodic-mp2-si)) | **−0.2764036524** | — |
| \(E_{\mathrm{corr}}\) (UMP2@UKS) dual gate | same target | **2×10⁻³** (UKS numerical drift) |
| \(E_{\mathrm{corr}} < 0\); \(E_{\mathrm{corr}}(\mathrm{all}) \le E_{\mathrm{corr}}(n_v{=}8) < 0\) | — | hard |
| OS + SS ≡ \(E_{\mathrm{corr}}\); \(E_{\mathrm{tot}} \equiv E_{\mathrm{SCF}}+E_{\mathrm{corr}}\) | identity | **1×10⁻¹²** |
| \(n_{\mathrm{occ}}\) | **8** | exact |
| CP2K absolute (`UKS`, `MULTIPLICITY 1`) | pending re-pin | — |

`perf_exempt` (no CP2K binary on the recording host); wall target ≤ 60 s. The
pin exercises equal-spin occupations only; high-spin open-shell UMP2 runs the
same kernel without a recorded value.

## periodic-ump2-kpoints — Multi-k GPW UMP2 — bct Si₄, MP 2×1×1 {#periodic-ump2-kpoints}

*No section of the case runs this variant.*

Momentum-conserving UMP2 on complex Bloch MOs of the multi-k UKS PBE SCF
(\(n_\alpha = n_\beta = 8\), multiplicity 1; CP2K twin `MP2_GPW` +
`&KPOINTS` + `UKS`):

\[
E^{(2)} = N_k^{-1}\sum_{k_i k_j k_a}\sum_{ijab}\frac{\cdots}{\Delta_{ij}^{ab}(k)},
\qquad k_b \equiv k_i + k_j - k_a
\]

(fractional wrap into the sampled set), with complex-modulus spin channels

\[
E_{\mathrm{SS}}^\sigma = \tfrac14\sum_{ijab}\frac{\lvert v_{iajb}^\sigma - v_{ibja}^\sigma\rvert^2}{\Delta},
\qquad
E_{\mathrm{OS}} = \sum_{ijab}\frac{\lvert v_{i_\alpha a_\alpha j_\beta b_\beta}\rvert^2}{\Delta}
\]

(no exchange in OS). With \(C^\alpha \equiv C^\beta\) the channels recover
multi-k RMP2. Kernel `mp2_uhf_energy_kpoints`; `mp2_n_virt = 2` per spin →
reported `n_virt = 4`.

```cpp
mk.n_alpha = 8;
mk.n_beta  = 8;               // other members: multi-k deck
mk.build(frame4);
mk.run(frame4);               // C_k + C_k_beta
const Mp2Result& r = mk.mp2();   // r.n_kpts == 2; r.ri
```

| Quantity | Atomiverse (OMP=4) [Ha] | Tol | Notes |
| -------- | ----------------------: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k UKS PBE) | −15.5751782413 | setup | 7 iters; matches RKS multi-k |
| \(E_{\mathrm{corr}}\) | **−0.0657637511** | **2e-3** vs [periodic-mp2-kpoints](#periodic-mp2-kpoints) | equal to the RMP2 value |
| \(E_{\mathrm{corr}}^{\mathrm{OS}}\) / \(E_{\mathrm{corr}}^{\mathrm{SS}}\) | −0.0598553603 / −0.0059083908 | 1e-12 identity | |
| \(E_{\mathrm{total}}\) | −15.6409419924 | identity | |
| Counts | `n_occ = 16`, `n_virt = 4`, `n_kpts = 2`, `ri` | exact | `C_k.size() == C_k_beta.size() == n_kpts` |
| Wall (SCF + UMP2) | **~3.7 s** | ≤ 60 s | `perf_exempt` |

Unit tier: multi-k UMP2 with α=β recovers multi-k RMP2 OS/SS.

## periodic-rpa-si — Γ-point GPW direct RPA — sc-Si₂ (dRPA@PBE) {#periodic-rpa-si}

*No section of the case runs this variant.*

dRPA correlation energy on the closed-shell Γ PBE reference of the sc-Si₂
deck: the shared Coulomb-only Casida kernel on the \((ia|jb)\) tensor of
[periodic-ri-mp2-si](#periodic-ri-mp2-si), reduced with the Furche plasmon
formula — the GPW analogue of molecular `DirectRPA<RHF>::run()`. The plasmon
energy is exact on the discrete spectrum and equals (non-DF limit) the
imaginary-frequency integral
\(\frac{1}{2\pi}\int_0^\infty\mathrm{d}\omega\,\mathrm{Tr}[\ln(1-\chi_0 v)+\chi_0 v]\).
Full spectrum \(n_{\mathrm{states}} = n_{\mathrm{occ}}\cdot n_{\mathrm{virt}}
= 4 × 22 = 88\).

```cpp
m.n_occ = 4;
m.level_shift = 0.3;
m.mp2_ri = true;               // product RI-GPW (ia|jb)
m.build(frame);
m.run(frame);
m.mp2_n_virt = 0;
const RpaResult& r = m.direct_rpa();   // r.e_corr, r.e_total, r.tr_A, r.omega_ha, r.ri
```

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (PBE, Γ) | **−7.5955445070 Ha** | setup | |
| **\(E_c^{\mathrm{dRPA}}\)** | **−0.2658539776 Ha** | **\(1\times 10^{-3}\) Ha** | frozen Atomiverse band (no CP2K plasmon twin) |
| \(E_{\mathrm{total}}\) | −7.8613984846 Ha | identity \(10^{-12}\) | |
| Tr \(A\) | 70.122806 Ha | — | diagnostic |
| \(n_{\mathrm{occ}}/n_{\mathrm{virt}}\) | 4 / 22 (\(n_{\mathrm{ov}}=88\)) | exact | |
| \(\Omega_{\min}/\Omega_{\max}\) | 0.029896 / 3.347032 Ha | all \(\Omega_s>0\) | |
| \(\lvert E_c^{\mathrm{RI}}-E_c^{\mathrm{Poisson}}\rvert\) | ≤ \(10^{-10}\) Ha | hard | `mp2_ri` on/off |

CP2K twin deck: same geometry / basis / PP / cutoff / level shift with
`&RI_RPA` (minimax, 8 quadrature points, Gaussian `RI_AUX`) and EXX
total-energy assembly — diagnostic only. Wall (OMP=4, EPYC): ~22 s total (SCF
≈ 20 s, RI-RPA ≈ 1 s, Poisson cross-check ≈ 0.3 s); `perf_exempt`.

## periodic-rpa-kpoints — Multi-k k-diagonal direct RPA — bct Si₄, MP 2×1×1 {#periodic-rpa-kpoints}

*No section of the case runs this variant.*

k-diagonal dRPA: each sampled \(k\) builds its own Casida block from
**same-\(k\)** (vertical, \(q=0\)) pairs
\(\rho_{ia}^{(k)} = \psi_i^{(k)}\psi_a^{(k)*}\),

\[
A_{ia,jb}(k) = (\varepsilon_{a k}-\varepsilon_{i k})\,\delta_{ij}\delta_{ab} + 2\,(ia|jb)_k,
\qquad B_{ia,jb}(k) = 2\,(ia|jb)_k,
\qquad
E_c = \sum_k w_k\,\tfrac12\Bigl(\sum_s \Omega_{k,s} - \mathrm{Tr}\,A_k\Bigr),
\]

with spectrum size \(N_k\cdot n_{\mathrm{occ}}\cdot n_{\mathrm{virt}}\).
`GpwDft::direct_rpa` on a multi-k reference runs the finite-\(q\) inter-\(k\)
screening of [periodic-rpa-finite-q](#periodic-rpa-finite-q); no public member
selects the k-diagonal approximation, so the values below are a record of that
approximation on the multi-k deck, not a reproducible call.

| Quantity | Value | Tol | Notes |
| -------- | ----: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k, PBE, 80 Ry) | −15.5751782413 Ha | setup | 7 iters |
| **\(E_c\)** (2k, \(n_{\mathrm{virt}}=2\)) | **≈ −0.0163 Ha** | **5×10⁻³ Ha** | band covers OMP orbital scatter of the multi-k SCF |
| \(E_c\) (OMP=1, deterministic) | −0.0157519268 Ha | info | |
| \(E_c\) (OMP=4, sample range) | −0.0154 … −0.0173 Ha | info | |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\); \(N_k\) / `ri` | 8 / 2; 2 / true | exact | |
| \(n_\Omega\) | \(2\cdot 8\cdot 2 = 32\) | exact | k-diagonal spectrum |
| Wall (SCF + RPA) | ~3 s | ≤ 60 s | `perf_exempt` |

The k-diagonal sum misses inter-\(k\) dielectric coupling; compare
\(E_c = -0.0736273682\) Ha from the finite-\(q\) path on the same deck.

## periodic-rpa-finite-q — Finite-\(q\) inter-\(k\) RI-GPW dRPA — bct Si₄, MP 2×1×1 {#periodic-rpa-finite-q}

*No section of the case runs this variant.*

The multi-k dRPA path of `GpwDft::direct_rpa`: for every transfer \(q\) on the
same closed MP mesh as \(k\), the Coulomb-only Casida spectrum of transitions
\(i_{\mathbf k}\to a_{\mathbf{k}+\mathbf q}\) with RI-GPW Coulomb at \(q\),
Furche-reduced and BZ-averaged (shared kernel, Common setup). At \(N_k = 1\)
it collapses to the real Γ plasmon formula of
[periodic-rpa-si](#periodic-rpa-si) (unit-pinned). Spectrum size

\[
n_\Omega = N_q\,N_k\,n_{\mathrm{occ}}\,n_{\mathrm{virt}} = N_k^2\,n_{\mathrm{ov}},
\]

64 on this deck; `tr_A` is \(\sum_q w_q\,\mathrm{Tr}\,A_q\).

```cpp
mk.n_occ = 8;                 // multi-k deck members
mk.build(frame4);
mk.run(frame4);
const RpaResult& r = mk.direct_rpa();   // r.e_corr, r.omega_ha, r.tr_A, r.n_kpts, r.ri
```

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k PBE, 80 Ry) | −15.5751782413 Ha (7 iters) | SCF gates | |
| \(E_c\) (\(n_{\mathrm{virt}}=2\)) | **−0.0736273682 Ha** | **5e-3 Ha** | frozen Atomiverse band (covers OMP orbital scatter); not a CP2K digit |
| \(E_{\mathrm{tot}}=E_{\mathrm{SCF}}+E_c\) | identity | **1e-12** | |
| \(E_c\) | \(<0\), finite | hard | |
| \(n_\Omega\) | **64** (\(=2\cdot2\cdot8\cdot2\)) | exact | |
| All \(\Omega_s\) | \(>0\), finite | hard | |
| Flags | `ri=true`, `n_kpts=2`, \(n_{\mathrm{occ}}=8\), \(n_{\mathrm{virt}}=2\) | exact | |
| Wall (SCF + RPA, OMP=4) | ~3 s | ≤ 60 s | `perf_exempt` |

CP2K twin deck: `MONKHORST-PACK 2 1 1`, `FULL_GRID`, `SYMMETRY .FALSE.`,
`ADDED_MOS 2` ↔ `mp2_n_virt=2`, `&RI_RPA` (diagnostic only).

## periodic-urpa-si — Γ-point unrestricted direct RPA (URPA@UKS) — sc-Si₂ {#periodic-urpa-si}

*No section of the case runs this variant.*

Spin-conserving dRPA on a Γ UKS PBE reference with closed-shell occupations
\(n_\alpha = n_\beta = 4\) (multiplicity 1). Same-spin OV tensors
\(V^{\alpha\alpha}\), \(V^{\beta\beta}\) and cross-spin \(V^{\alpha\beta}\) come
from `mo_eri_ovov{,_ri}` / `mo_eri_ovov_ab{,_ri}`; the Casida blocks span the
composite α→α ⊕ β→β space with spins explicit, and `internal::direct_rpa_uhf`
applies the Furche formula. When \(C^\alpha \approx C^\beta\), **URPA ≡ RKS
dRPA** — the gate is a dual gate against [periodic-rpa-si](#periodic-rpa-si),
not a high-spin open-shell pin. \(\mathrm{Tr}\,A\) is ≈ 2 × the RKS value when
α=β; `omega_ha` has length \(n_{\mathrm{ov}}^\alpha+n_{\mathrm{ov}}^\beta\).

```cpp
m.n_alpha = 4;
m.n_beta = 4;                  // UKS closed-shell occupations
m.level_shift = 0.3;
m.mp2_ri = true;
m.build(frame);
m.run(frame);
m.mp2_n_virt = 0;              // per spin
const RpaResult& r = m.direct_rpa();   // r.omega_ha composite α⊕β; r.ri == true
```

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E_c\) (URPA) | **−0.2658538251 Ha** | — | Δ vs RKS ≈ \(1.5\times 10^{-7}\) Ha |
| \(E_c\) (RKS dRPA, [periodic-rpa-si](#periodic-rpa-si)) | **−0.2658539776 Ha** | **\(2\times 10^{-3}\) Ha** band | dual gate |
| \(E_{\mathrm{SCF}}\) (UKS-PBE) | **−7.5955445033 Ha** | soft | matches RKS to ~\(4\times 10^{-9}\) |
| \(\mathrm{Tr}\,A\) (composite) | 136.534225 | soft | ≈ 2 × RKS Tr A |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) / \(n_\Omega\) | 8 / 44 / 176 | hard | spin-orbital composite |
| RI vs Poisson \(E_c\) | \(|\Delta| \le 10^{-8}\) Ha | hard | `mp2_ri` on/off |
| SCF converged; all \(\Omega_s > 0\); \(E_c < 0\); \(E_{\mathrm{tot}} = E_{\mathrm{SCF}} + E_c\) | — | hard | |

CP2K twin deck: UKS `MULTIPLICITY 1` + `&RI_RPA` MINIMAX (diagnostic only).
Wall (8 OpenMP, EPYC): SCF ≈ 14 s + RI-URPA ≈ 0.7 s + Poisson cross-check ≈
2.7 s → **~18 s**; `perf_exempt`.

## periodic-urpa-kpoints — Multi-k unrestricted direct RPA (URPA@UKS) — bct Si₄ {#periodic-urpa-kpoints}

*No section of the case runs this variant.*

Finite-\(q\) inter-\(k\) spin-conserving dRPA on the multi-k UKS reference
(\(n_\alpha = n_\beta = 8\)): for each transfer \(q\), transitions
\(i_k^\sigma \to a_{k+q}^\sigma\) over the composite α block then β block,
Coulomb at \(q\) via RI-GPW, Furche-averaged over \(q\)
(`internal::direct_rpa_uhf_energy_kpoints`). With
\(n_{\mathrm{ov}}^\alpha = n_{\mathrm{ov}}^\beta = 8\cdot 2 = 16\) (composite
32) and roots concatenated over all ordered \((k,q)\) pairs,
\(n_\Omega = N_k^2 \cdot 32 = 128\). When \(C^\alpha \approx C^\beta\) the
result matches multi-k RKS dRPA ([periodic-rpa-finite-q](#periodic-rpa-finite-q))
bit for bit.

```cpp
mk.n_alpha = 8;
mk.n_beta  = 8;               // multi-k deck members; mp2_n_virt = 2 per spin
mk.build(frame4);
mk.run(frame4);
const RpaResult& r = mk.direct_rpa();   // multi-k URPA
```

| Quantity | Atomiverse (OMP=8) | Tol | Notes |
| -------- | -----------------: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k UKS-PBE) | −15.5751782413 Ha | setup | 7 iters; \(\langle S^2\rangle=8\) as recorded |
| **\(E_c\)** (2k URPA, n_virt=2/spin) | **−0.0736273682 Ha** | **5e-3 Ha** vs multi-k RKS dRPA (same value) | dual gate |
| Tr \(A\) (composite BZ average) | 17.508325 Ha | diagnostic | |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) / \(n_\Omega\) | 16 / 4 / 128 | exact | spin-orbital counts |
| Wall (SCF + URPA) | ~2.7 s | ≤ 60 s | `perf_exempt` |

Hard gates: SCF converged; `C_k.size()==C_k_beta.size()==n_kpts`;
`n_kpts==2`; `ri`; \(E_c\) finite and strictly negative; all \(\Omega>0\);
\(e_{\mathrm{total}} \equiv e_{\mathrm{scf}}+e_{\mathrm{corr}}\). Unit tier:
`UhfKpointsNk1DualGatesRhf`, `UhfFiniteQTwoKSpectrumShape`. Single-rank pin;
open-shell (\(n_\alpha\neq n_\beta\)) multi-k URPA runs the same kernel without
a recorded value.

## periodic-gw-si — Γ-point spectral G0W0 — sc-Si₂ (PBE reference) {#periodic-gw-si}

*No section of the case runs this variant.*

Closed-shell spectral \(G_0W_0\)@PBE at Γ on the sc-Si₂ deck: dRPA screening
on the RI-GPW \((ia|jb)\) of [periodic-rpa-si](#periodic-rpa-si), spectral
\(\Sigma_c\), MT \(\Sigma_x\), Newton QP equation (Common setup). QP levels
for every MO; primary observable the HOMO–LUMO QP gap.

```cpp
m.n_occ = 4;
m.level_shift = 0.3;
m.mp2_ri = true;
m.build(frame);
m.run(frame);
m.mp2_n_virt = 0;              // all kept virtuals in dRPA
const G0W0Result& gw = m.g0w0();   // gw.gap_ev [eV], gw.qp_energy_ha [Ha], gw.converged, gw.ri
```

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| SCF / Newton / `ri` / \(n_{\mathrm{occ}}=4\) | converged / `converged` / true / exact | hard | |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}} < \varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\); finite shifts vs KS | required | hard | |
| \(E_{\mathrm{SCF}}\) (PBE, Γ) | **−7.5955445070 Ha** | info | |
| KS gap | **0.410147 eV** | info | |
| **QP gap** | **5.719545 eV** | **0.05 eV** | frozen Atomiverse band |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | −0.13897686 Ha | info | Δ vs KS +0.071804 Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | +0.07121256 Ha | info | Δ vs KS +0.123312 Ha |
| IP / EA (QP) | 3.7818 / 1.9378 eV | info | |

CP2K twin deck: RI-G0W0 + minimax quadrature + Gaussian `RI_AUX` under
`&WF_CORRELATION` (diagnostic only). Wall ≈ **24 s** (SCF ≈ 23 + G0W0 ≈ 0.6;
4 OMP, EPYC); `perf_exempt`. Self-consistent GW remains open
([excited-states roadmap](../../roadmap/excited-states.md) §5;
periodic §10 in [dft.md](../../roadmap/dft.md)).

## periodic-gw-kpoints — Multi-k k-diagonal spectral G0W0 — bct Si₄, MP 2×1×1 {#periodic-gw-kpoints}

*No section of the case runs this variant.*

Spectral \(G_0W_0\)@PBE on the multi-k deck with **k-diagonal** screening:
dRPA poles at each \(k\) from same-\(k\) pairs
\(\rho_{ia}^{(k)} = \psi_i^{(k)}\psi_a^{(k)*}\) (RI-GPW Poisson on complex pair
densities), \(\Sigma_{c,pp}^{(k)}(\omega)\) as the spectral sum over those poles,
MT \(\Sigma_x^{(k)}\), PBE \(v_{\mathrm{xc}}\) on the BZ-weighted density, and
the per-\(k\) Newton equation
\(\omega = \varepsilon_p^{(k)} + \mathrm{Re}\,\Sigma_{c,p}^{(k)}(\omega) + \Sigma_{x,p}^{(k)} - v_{\mathrm{xc},p}^{(k)}\).
Primary observable: minimum direct QP gap over the mesh. `GpwDft::g0w0` on a
multi-k reference runs the finite-\(q\) screening of
[periodic-gw-finite-q](#periodic-gw-finite-q); no public member selects the
k-diagonal approximation, so the values below are a record of it, not a
reproducible call.

| Quantity | Value | Tol | Notes |
| -------- | ----: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k, PBE, 80 Ry) | **−15.5751782413 Ha** | — | 7 iters |
| KS min-direct gap | **1.561168 eV** | — | diagnostic |
| **min-direct QP gap** | **8.009033 eV** | **0.5 eV** | primary band |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) / \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) (report \(k\)) | **+0.167405** / **+0.461731 Ha** | — | MT \(x\) + tiny virt space shift absolute levels |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) / \(n_k\) | 8 / 2 / 2 | exact | |
| Wall (SCF + G0W0, OMP=4, EPYC) | **~34 s** | ≤ 60 s | `perf_exempt` |

Hard gates: SCF converged; `C_k.size()==n_kpts`; `n_kpts==2`; `ri`; Newton
converged; finite positive min-direct QP/KS gaps; HOMO QP < LUMO QP. CP2K twin
deck: PBE, 80 Ry, `MONKHORST-PACK 2 1 1`, `FULL_GRID .TRUE.` /
`SYMMETRY .FALSE.`, `ADDED_MOS 2`, RI-G0W0 with `CORR_MOS_OCC 8` /
`CORR_MOS_VIRT 2` (diagnostic only).

## periodic-gw-finite-q — Finite-\(q\) inter-\(k\) spectral G0W0 — bct Si₄, MP 2×1×1 {#periodic-gw-finite-q}

*No section of the case runs this variant.*

The multi-k path of `GpwDft::g0w0`. For each transfer \(q\) on the same closed
MP mesh as \(k\), the Coulomb-only Casida spectrum of transitions
\(i_k\to a_{k+q}\) (RI-GPW Poisson, complex Bloch; the dielectric of
[periodic-rpa-finite-q](#periodic-rpa-finite-q)) gives \(X+Y\) amplitudes and
spectral transition potentials \(W_s^q\); the self-energy at target \(k\)
contracts intermediate states at \(k-q\) with weight \(1/N_k\) (private kernel
`spectral_g0w0_rhf_kpoints`). At \(N_k = 1\) with real-valued complex orbitals
it recovers the real Γ spectral path (unit-tested).

```cpp
mk.n_occ = 8;                 // multi-k deck members; mp2_ri required
mk.build(frame4);
mk.run(frame4);
const G0W0Result& gw = mk.g0w0();   // gw.gap_ev, gw.mf_gap_ev, gw.ip_ev, gw.ea_ev; gw.n_kpts == 2
```

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| SCF | converged; `n_kpts = 2`; `C_k.size() = 2` | hard | |
| \(E_{\mathrm{SCF}}\) | **−15.5751782413 Ha** (7 iters) | setup | |
| KS min-direct gap | **1.561168 eV** | finite > 0 | |
| **QP min-direct gap** | **7.529834 eV** | **0.5 eV** | frozen Atomiverse band; not a CP2K digit |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) / \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | +0.171486 / +0.448203 Ha | HOMO < LUMO | report \(k\) |
| Newton | `converged == true` | hard | all orbits, all \(k\) |
| RI / counts | `ri == true`; `n_occ = 8`; `n_virt = 2` | hard | |
| Wall (SCF + G0W0, OMP=4) | ~10 s | ≤ 60 s | `perf_exempt` |

Finite-\(q\) \(W\) softens the gap relative to the k-diagonal approximation
(~8.01 eV, [periodic-gw-kpoints](#periodic-gw-kpoints)). CP2K twin deck:
multi-k GPW RKS PBE, `MONKHORST-PACK 2 1 1`, `FULL_GRID` / no symmetry,
`ADDED_MOS 2`, `&RI_G0W0` with `CORR_MOS_OCC 8` / `CORR_MOS_VIRT 2`
(diagnostic only).

## periodic-ugw-si — Γ-point UGW@UKS — sc-Si₂ (spectral G0W0, PBE) {#periodic-ugw-si}

*No section of the case runs this variant.*

Unrestricted spectral \(G_0W_0\) on a Γ UKS PBE reference with closed-shell
occupations \(n_\alpha = n_\beta = 4\): spin-conserving dRPA over α→α ⊕ β→β
(same ovov tensors as Γ UMP2), same-spin \(G_\sigma W\), per-spin MT
\(\Sigma_x^\sigma\), spin-resolved PBE \(v_{\mathrm{xc}}^\sigma\); α-channel
report. When \(C^\alpha \approx C^\beta\), **UGW ≡ RGW** to numerical noise
(\(\Delta_{\mathrm{gap}} \sim 2\times 10^{-5}\) eV on this deck), which is the
gate against [periodic-gw-si](#periodic-gw-si). The β spectrum is used in the
screening but not exported as a second block.

```cpp
m.n_alpha = 4;
m.n_beta = 4;                  // UKS closed-shell occupations
m.level_shift = 0.3;
m.mp2_ri = true;
m.build(frame);
m.run(frame);
m.mp2_n_virt = 0;              // per spin
const G0W0Result& gw = m.g0w0();   // gw.n_occ == n_α; gw.n_virt == n_vα + n_vβ
```

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| SCF | converged | — | UKS-PBE, tol 1e-6, max 80 iters |
| \(E_{\mathrm{SCF}}\) | **−7.5955445033 Ha** | ~4e-9 vs RKS | |
| KS gap α | **0.410157 eV** | — | diagnostic |
| **QP gap α** | **5.719545 eV** dual gate | **0.05 eV** | vs RKS [periodic-gw-si](#periodic-gw-si) |
| UGW QP gap | **5.719564 eV** | \(\Delta\sim 2\times 10^{-5}\) eV | |
| Newton / gap sign / order / `ri` / \(n_{\mathrm{occ}}=4\) | converged / > 0 / \(\varepsilon_H^{\mathrm{QP}} < \varepsilon_L^{\mathrm{QP}}\) / true | hard | |

Unit tier: `direct_rpa_uhf` ≡ `direct_rpa_rhf` on α=β tensors; UKS unconverged
guard. CP2K twin deck: UKS `MULTIPLICITY 1` + `&RI_G0W0` (periodic HF in the
GW step; diagnostic only). Wall ~14 s (SCF ≈ 13 + UGW ≈ 1; 8 OMP);
`perf_exempt`.

## periodic-ugw-kpoints — Multi-k unrestricted spectral G0W0 (UGW) — bct Si₄ {#periodic-ugw-kpoints}

*No section of the case runs this variant.*

UGW on the multi-k UKS PBE reference (\(n_\alpha = n_\beta = 8\)) with
finite-\(q\) inter-\(k\) \(W\) on `C_k` / `C_k_beta` (the dielectric of
[periodic-gw-finite-q](#periodic-gw-finite-q), spin-conserving). Report: α
min-direct QP gap over the mesh. `mp2_n_virt = 2` per spin → reported
`n_virt = 4`.

```cpp
mk.n_alpha = 8;
mk.n_beta = 8;                // multi-k deck members
mk.build(frame4);
mk.run(frame4);
const G0W0Result& gw = mk.g0w0();   // gw.n_kpts == 2; gw.gap_ev = min_k direct α QP gap
```

| Quantity | Atomiverse (OMP=8) | Tol | Notes |
| -------- | -----------------: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k UKS-PBE) | **−15.5751782413 Ha** | info | 7 iters; ≡ RKS multi-k |
| KS min-direct gap α | **1.561168 eV** | > 0 | ≡ RKS multi-k |
| **min-direct QP gap α** | **7.698404 eV** | \|gap − 7.529834\| ≤ **0.5 eV** | Δ vs RKS finite-\(q\) G0W0 ([periodic-gw-finite-q](#periodic-gw-finite-q)) ≈ 0.17 eV |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) / \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | +0.158037 / +0.440948 Ha | HOMO < LUMO | report \(k\) |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) / `n_kpts` / `ri` / Newton | 8 / 4 / 2 / true / converged | exact | `C_k.size()==C_k_beta.size()==n_kpts` |
| Wall (SCF + UGW) | **~16 s** | ≤ 60 s | `perf_exempt` |

Unit tier: multi-k UHF at \(N_k=1\) with \(C^\alpha=C^\beta\) ≡ multi-k RHF
spectral QP.

## ugw-uks-rpbe — Γ-point UGW@UKS-RPBE — sc-Si₂ {#ugw-uks-rpbe}

*No section of the case runs this variant.*

Γ UGW on the sc-Si₂ deck with the **RPBE** spin GGA in both the UKS SCF and
the UGW \(v_{\mathrm{xc}}^\sigma\) diagonals (shared grid kernel
`xc_grid_rpbe_spin`, `xc_grid_gga.h`), dual-gated against closed-shell
\(G_0W_0\)@RPBE and required to differ from the PBE UGW gap. RPBE is
Hammer–Hansen–Nørskov exchange
\(F_x(s) = 1 + \kappa\bigl(1 - e^{-\mu s^2/\kappa}\bigr)\) with PBE
\(\kappa,\mu\) and PBE correlation (libxc `GGA_X_RPBE` + `GGA_C_PBE`) — **not**
Zhang–Yang revPBE. CP2K 2026.1 needs nested `&GGA_X_RPBE` / `&GGA_C_PBE`
blocks; `XC_FUNCTIONAL RPBE` is not a valid enumeration (as in
[gpw-uks-rpbe](gpw-uks.md#gpw-uks-rpbe)).

```cpp
GpwDft m{{si}, 30.0, XcKind::RPBE};   // Hammer RPBE + PBE c (not revPBE)
m.n_alpha = 4;
m.n_beta = 4;
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;
m.mp2_ri = true;
m.build(frame);
m.run(frame);
m.mp2_n_virt = 0;
const G0W0Result& gw = m.g0w0();      // dual-gate twin: same deck with n_occ = 4
```

| Quantity | Value / gate | Tol |
| -------- | ------------ | --- |
| \|gap(UGW@RPBE) − gap(RKS G0W0@RPBE)\| | UGW ≡ RGW | **≤ 0.05 eV** |
| \|gap(UGW@RPBE) − 5.719545 eV\| (PBE UGW, [periodic-ugw-si](#periodic-ugw-si)) | functional integrity | **≥ 1e-3 eV** |
| SCF + Newton converged; gap > 0; HOMO QP < LUMO QP; finite shifts; `ri` | hard | — |
| QP gap UGW@RPBE | **5.674272 eV** | α channel |
| QP gap RKS G0W0@RPBE | **5.674251 eV** | \(\lvert\Delta\rvert \approx 2\times 10^{-5}\) eV |
| \(E_{\mathrm{SCF}}\) (UKS-RPBE) | **−7.6173910139 Ha** | matches RKS to ~7e-9 Ha |
| KS gap α | **0.384145 eV** | diagnostic |

The PBE value differs by \(\gg 1\) meV. Wall ≈ 22 s (UKS SCF + UGW + RKS SCF +
G0W0, 8 OMP); `perf_exempt`. CP2K twin deck needs
`ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,BASIS_RI_MOLOPT,GTH_POTENTIALS} .` in the
work directory (diagnostic only). Reference: B. Hammer, L. B. Hansen,
J. K. Nørskov, *Phys. Rev. B* **59**, 7413 (1999). Related:
[gpw-uks-o2-gamma](gpw-uks.md#gpw-uks-o2-gamma) (Γ UKS energy),
[gpw-pbesol-si](gpw-rks.md#gpw-pbesol-si).
