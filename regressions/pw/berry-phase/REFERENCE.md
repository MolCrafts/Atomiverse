# pw/berry-phase

One case; every section below is run by `atv_regressions --case pw/berry-phase` and keeps its own oracle record.


## Reference — Berry-phase (King–Smith–Vanderbilt) polarization, bulk Si

### Method

Modern theory of polarization on the CPU plane-wave method: restricted
Kohn–Sham LDA SCF, then non-self-consistent diagonalization of the occupied
bands on strings of k-points along each reciprocal lattice vector, and the
electronic polarization from the string-averaged Berry phase

\[
\varphi_i = \Bigl\langle \operatorname{Im}\ln\prod_{j=0}^{J-1}
  \det S^{(j)} \Bigr\rangle_{\mathrm{strings}},\qquad
S^{(j)}_{mn} = \langle u_{m,k_j}\,|\,u_{n,k_{j+1}}\rangle ,
\]

closing the loop through the periodic-gauge coefficient shift
\(c_{n,k_0+\mathbf b}(\mathbf G)=c_{n,k_0}(\mathbf G+\mathbf b)\)
(King-Smith & Vanderbilt, *Phys. Rev. B* **47**, 1651 (1993); Resta,
*Rev. Mod. Phys.* **66**, 899 (1994)).

Public path:

```cpp
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};
cpu::pw::BerryPolarization berry;   // members: k_perp, k_string, directions
auto r = berry.compute(method, frame);
// r.phase_el / r.p_el_frac / r.p_ion_frac / r.p_frac [e] / r.dipole_e_ang [e Å]
```

No public `*Config` bag; no factories.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |

### Literature pins (symmetry-exact, parameter-free)

1. **Centrosymmetric formal polarization.** A crystal with inversion
   symmetry has formal polarization on the lattice \(\{0, \tfrac12 Q\}\)
   (quantum \(Q = e\mathbf a_i/V\); Spaldin, *J. Solid State Chem.* **195**,
   2 (2012), §3; Resta & Vanderbilt, in *Physics of Ferroelectrics*, Springer
   (2007)). Diamond Si is centrosymmetric → every total \(p_i\) sits on the
   half-quantum lattice.
2. **Born effective charge of an elemental crystal is exactly zero.** The
   acoustic sum rule \(\sum_a Z^*_a = 0\) (Pick, Cohen & Martin, *Phys.
   Rev. B* **1**, 910 (1970)) plus the site-symmetry equivalence of the two
   diamond sublattices give \(Z^* = 0\) per atom. A sublattice displacement
   \(\delta\) must therefore leave \(p_3\) unchanged: the electronic Berry
   phase cancels the bare ionic \(Z^\mathrm{val}\delta = 0.08\) exactly. A
   sign error or a dead electronic term shows as \(|Z^*| \approx 4\) or 8.

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4)(1,1,1)\); displaced case: atom 1 \(+0.02\,\mathbf a_3\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry |
| SCF k-mesh | \(2\times2\times2\) (`PW::k_mesh = 2`) |
| XC | `XCType::LSDA` |
| SCF tol | \(10^{-6}\) |
| Strings | `k_perp = 2` (4 strings/direction), `k_string = 4` |

### Acceptance (hard-coded in `reg_berry-polarization.cpp`)

1. Both SCF runs converge.
2. Ionic bookkeeping exact: ideal \(p^\mathrm{ion} = (1,1,1)\) (tol 1e-9);
   displaced \(\Delta p^\mathrm{ion}_3 = 4\times0.02 = 0.08\) (tol 1e-9).
3. Centrosymmetric pin: distance of every ideal-Si \(p_i\) to the nearest
   multiple of 0.5 < **0.05** \(e\).
4. Born-charge pin: \(|Z^*| = |\Delta p_3|/\delta < 0.5\).

Measured on this host (2026-07-25, Release): ideal phases
\(\varphi = 0.000000\) per direction (quantization exact to \(10^{-6}\)),
\(p = \pm1\) (≡ half-quantum lattice); displaced \(\Delta p_3 = +0.000000\),
\(Z^* = +0.0000\) — the ASR cancellation is exact to print precision.

### Root-cause fixes shipped with this case (shared PW solver)

The Berry phase is first-order sensitive to eigenVECTOR quality and exposed
two latent bugs invisible to energy-window regressions:

1. **`enumerate_g_vectors` index-bound clipping** — the per-axis bound
   \(\pm(g_\mathrm{max}/|\mathbf b_i| + 1)\) ignored the k-offset of the
   sphere centre and the obliquity of the reciprocal basis, silently
   dropping sphere-edge G-vectors (label-equivalent spheres at fractional
   0.125 vs 0.875 had 271 vs 264 G's). Fixed with exact index windows
   \(-f_i \pm g_\mathrm{max}|\mathbf a_i|/2\pi\); FFT-grid sizing unchanged.
2. **Davidson subspace Hermiticity** — the subspace Hamiltonian
   \(\langle v_\mu|H|v_\nu\rangle\) was truncated to its real part and
   diagonalized with a real Jacobi solver. At any non-TRIM k-point this
   corrupts the Ritz vectors at first order (occupied subspace failed
   \(|\det\langle u^*|u\rangle| = 1\) at the 0.6 level) while leaving
   eigenvalues plausible. Fixed with a complex Hermitian Jacobi
   (`jacobi_eigh_hermitian`); string phases quantize to \(10^{-6}\) after
   the fix. The CUDA band solver has an analogous truncation and is out of
   scope for this loop (flagged for the CUDA backlog).

All 17 `test_cpu_pw_*` unit tests and all 14 `regressions/pw/*` cases were
re-run green after the fixes.

### CP2K note

CP2K computes the same observable as a Berry-phase dipole
(`&DFT / &PRINT / MOMENTS { PERIODIC T }`, Γ-limit Resta formula) with
GTH + Gaussian basis — absolute phases are not directly comparable to the
ONCV plane-wave twin; the symmetry pins above (lattice membership, ASR
\(Z^*=0\)) are basis-independent and are the scientific content. Offline twin
input echo: `tools/cp2k_oracle/inputs/si_berry_polarization_note.inp`.

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~18 s (Release, ecut 15 / k=2; 2 SCF + 4 direction sweeps of 16 NSCF diagonalizations) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time the same geometry with the oracle
input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) here, require
\(r \le 10\).


## Reference — Collinear-spin Berry-phase polarization, bulk Si

### Method

King–Smith–Vanderbilt formal polarization on the **collinear spin** CPU
plane-wave path (`cpu::pw::PW` with `spin_polarized = true`, fixed
occupations). Each spin channel has its own occupied manifold; string
phases \(\varphi_{i,s}\) are averaged independently and

\[
p^\mathrm{el}_i = \frac{\varphi_{i,\uparrow}+\varphi_{i,\downarrow}}{2\pi}
\quad\text{(spin factor \(f=1\) per channel)}.
\]

Total formal polarization wraps with quantum \(1\) (nspin=2 convention).
Ionic bookkeeping is spin-independent.

Public path:

```cpp
cpu::pw::PW method{cpu::pw::UpfPseudopotential{"..."}, /*ecut_ry=*/15.0,
                   /*k_mesh=*/2, cpu::scf::XCType::LSDA};
method.spin_polarized = true;
method.multiplicity = 1;          // N↑−N↓ = 0 for diamond Si
cpu::pw::BerryPolarization berry; // k_perp, k_string, directions
auto r = berry.compute(method, frame);
```

No public `*Config` bag; no factories. Closed-shell path
(`spin_polarized = false`) is unchanged — see
`regressions/pw/berry-phase`.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |

### Literature pins (symmetry-exact, parameter-free)

Same basis-independent pins as `berry-polarization`:

1. **Centrosymmetric formal polarization** (Spaldin, *J. Solid State Chem.*
   **195**, 2 (2012)): diamond Si → every total \(p_i\) on the lattice
   \(\{0,\tfrac12 Q\}\) (distance to nearest multiple of 0.5 < 0.05).
2. **Born effective charge of an elemental crystal is exactly zero**
   (Pick, Cohen & Martin, *Phys. Rev. B* **1**, 910 (1970)): sublattice
   displacement \(\delta\) leaves total \(p_3\) unchanged
   (\(|Z^*|=|\Delta p_3|/\delta < 0.5\)).

With multiplicity 1 and even \(N_e\), the collinear path must recover the
same physics as the closed-shell KSV strings (modulo the smaller nspin=2
polarization quantum branch choice).

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4)(1,1,1)\); displaced: atom 1 \(+0.02\,\mathbf a_3\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry |
| SCF k-mesh | \(2\times2\times2\) |
| XC | `XCType::LSDA` |
| Spin | collinear, multiplicity 1 (\(N_\uparrow=N_\downarrow=4\)) |
| SCF tol | \(10^{-6}\) |
| Strings | `k_perp = 2`, `k_string = 4` |

### Acceptance (hard-coded in `reg_berry-spin-si.cpp`)

1. Both collinear SCF runs converge.
2. Ionic bookkeeping exact: ideal \(p^\mathrm{ion}=(1,1,1)\) (tol 1e-9);
   displaced \(\Delta p^\mathrm{ion}_3=0.08\) (tol 1e-9).
3. Centrosymmetric pin: distance of every ideal \(p_i\) to nearest
   multiple of 0.5 < **0.05**.
4. Born-charge pin: \(|Z^*|<0.5\).

### CP2K note

Offline twin echoes the closed-shell Berry note with an explicit UKS /
LSD spin section for documentation — absolute phases still not comparable
to ONCV PW; symmetry pins are the scientific content.

Input echo: `tools/cp2k_oracle/inputs/si_berry_spin_note.inp`.

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No loadable CP2K/QE PW binary on the build host for a fair wall twin this shift (same host constraint as `berry-polarization`) |
| ATV twin wall (informative) | **32.05 s** (Release, ecut 15 / k=2; 2 collinear SCF + string sweeps) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

Measured on this host (2026-07-26, Release): ideal phases
\(\varphi=0\) per direction (quantization exact to print precision),
\(p=0\) on the nspin=2 quantum-1 branch (≡ half-quantum lattice via
Spaldin pin); displaced \(\Delta p_3=+0.000000\), \(Z^*=+0.0000\).
