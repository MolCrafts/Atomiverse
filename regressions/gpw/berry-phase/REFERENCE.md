# gpw/berry-phase

One case; every section below is run by `atv_regressions --case gpw/berry-phase` and keeps its own oracle record.


The canonical batched exact Fourier-overlap implementation was validated on
2026-10-25 (build12, Release, Apple M4, OpenMP/OpenBLAS 8/8, profiling enabled).
Both existing sections pass their unchanged numerical gates and strict
whole-section 60 s bound. Equal short wavevectors reuse image-local Fourier
integrals, while each ket k retains its Bloch phase and matrix sum order.
The resulting AO operators are shared by the occupied UKS manifolds.

### Current native validation

| Section | Whole section | Ideal total polarization | Displaced Δp_z | Born charge |
| --- | --- | --- | --- | --- |
| `periodic-berry-gpw` | **10.070322 s** | (0, 0, 0) at printed precision | +0.000069 e | +0.0034 e |
| `periodic-berry-gpw-uks` | **15.201491 s** | (0, 0, 0) at printed precision | +0.000069 e | +0.0035 e |

The timer covers all three ideal-cell directions and the displaced-cell z
direction, including construction and assertions. RKS then UKS run in the
same process. Per-axis profiles separate SCF and exact AO/MO link work;
every axis uses the ordinary `method.run(frame)` path.

[Raw CTest output](../../../tools/cp2k_oracle/inputs/gpw-berry-phase/native-validation.txt),
[metadata/hashes](../../../tools/cp2k_oracle/inputs/gpw-berry-phase/native-validation.json)
and [reproduction instructions](../../../tools/cp2k_oracle/inputs/gpw-berry-phase/README.md)
archive the actual measurements. These native values are validation evidence;
the literature symmetry/ASR pins remain the oracle. Earlier native measurements
are retained as historical diagnostics. No fresh CP2K wall ratio is inferred
from the historical external runs below.

### Exact Gaussian Fourier overlaps

Berry and Wannier links share `prim_overlap_fourier` in the existing GPW
one-electron integral kernel. It evaluates

\[
F_{ab}(\mathbf q)=\int a(\mathbf r)b(\mathbf r)
  e^{-i\mathbf q\cdot\mathbf r}\,d^3r
= c_a c_b\left(\frac{\pi}{p}\right)^{3/2}
  e^{-|\mathbf q|^2/(4p)-i\mathbf q\cdot\mathbf P}
  \prod_{d=x,y,z}\sum_{t=0}^{l_{a,d}+l_{b,d}}
    E_t^{l_{a,d},l_{b,d}}(-i q_d)^t,
\]

where \(p=\alpha_a+\alpha_b\),
\(\mathbf P=(\alpha_a\mathbf A+\alpha_b\mathbf B)/p\), and the existing
McMurchie–Davidson coefficients \(E_t\) include the Gaussian product's
center-separation attenuation. The wavevector \(\mathbf q\) is the short,
unwrapped ket-minus-bra reciprocal step in Bohr⁻¹; primitive coordinates
are in Bohr and the Fourier sign is **negative**. Periodic image phases
retain the ket's Bloch convention and wrapped k labels.

Every angular term contributes. In particular a same-center s/pₓ product
has zero ordinary overlap but Fourier integral
\(-i q_x F_{ss}(\mathbf q)/(2p)\). Multiplying the ordinary overlap by only
a scalar damping/center phase loses that term and the higher p/d moments.
The shared kernel uses the exact Hermite polynomial instead.

The added pure `GpwOneElectron.Fourier*` tests target closed-form s/s,
odd s/p and d/p, p/p and d/d second/fourth moments, a separated-center p/d
integral, zero-wavevector overlap identities, and conjugation/exchange
identities. They do not solve an SCF or use native snapshots as expected
values. In build13, all 17 `GpwOneElectron` tests (including seven primitive
Fourier tests) and all three `GpwFourierLinks` tests passed. The batched tests
cover analytic image sums for equal/distinct wavevectors, split/batch
identity, mixed-angular translation/conjugation covariance, and invalid-domain
exception propagation. The [actual unit-output extract](../../../tools/cp2k_oracle/inputs/gpw-berry-phase/native-math-validation.txt)
and [metadata/hashes](../../../tools/cp2k_oracle/inputs/gpw-berry-phase/native-math-validation.json)
record all **1771/1771** registered units passing in **46.66 s**. Build13 adds
test registration; native mathematical code is unchanged from the build12
regression measurements above.



## Reference — Berry-phase (King–Smith–Vanderbilt) polarization, bulk Si (GPW)

### Method

Modern theory of polarization on the CPU GPW method: multi-k restricted
Kohn–Sham LDA SCF on strings of k-points along each reciprocal lattice
vector (AO-basis Bloch MOs), and the electronic polarization from the
string-averaged Berry phase

\[
\varphi_i = \Bigl\langle \operatorname{Im}\ln\prod_{j=0}^{J-1}
  \det M^{(j)} \Bigr\rangle_{\mathrm{strings}},\qquad
M^{(j)}=C^\dagger(k_j)\,S^{(k_j,k_{j+1})}\,C(k_{j+1}),
\]

with phased AO overlaps \(S\) carrying \(e^{-i\mathbf b\cdot\mathbf r}\)
and the short reciprocal step \(b=2\pi/(L_i J)\) on wrapped k-labels
(King-Smith & Vanderbilt, *Phys. Rev. B* **47**, 1651 (1993); Resta,
*Rev. Mod. Phys.* **66**, 899 (1994)).

Public path:

```cpp
cpu::gpw::GpwDft method{{si_kind}, /*cutoff_ry=*/80.0,
                        cpu::gpw::XcKind::PADE};  // + knobs: n_occ, …
method.build(frame);                     // positions [Å], Z, orthorhombic box [Å]
cpu::gpw::BerryPolarization berry;       // members: k_perp, k_string, directions
auto r = berry.compute(method, frame);
// r.phase_el / r.p_el_frac / r.p_ion_frac / r.p_frac [e] / r.dipole_e_ang [e Å]
```

No public `*Config` bag; no factories.

### Pseudopotential / basis

| Item | Value |
| ---- | ----- |
| GTH | `GTH-PBE-q4` / PADE twin class (`GTH-PADE-q4` in CP2K) |
| Basis | `DZVP-MOLOPT-GTH` (ATV tables) / `DZVP-MOLOPT-SR-GTH` (CP2K) |
| Valence | \(Z^\mathrm{val}=4\) per Si |

### Literature pins (symmetry-exact, parameter-free)

1. **Centrosymmetric formal polarization.** A crystal with inversion
   symmetry has formal polarization on the lattice \(\{0, \tfrac12 Q\}\)
   (quantum \(Q = e\mathbf a_i/V\); Spaldin, *J. Solid State Chem.* **195**,
   2 (2012), §3; Resta & Vanderbilt, in *Physics of Ferroelectrics*, Springer
   (2007)). Diamond Si is centrosymmetric → every total \(p_i\) sits on the
   half-quantum lattice.
2. **Born effective charge of an elemental crystal is exactly zero.** The
   acoustic sum rule \(\sum_a Z^*_a = 0\) (Pick, Cohen & Martin, *Phys.
   Rev. B* **1**, 910 (1970)) plus the site-symmetry equivalence of the
   diamond sublattices give \(Z^* = 0\) per atom. A sublattice displacement
   \(\delta\) must therefore leave \(p_z\) unchanged: the electronic Berry
   phase cancels the bare ionic \(Z^\mathrm{val}\delta = 0.08\) exactly.

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 4-atom BCT orthorhombic cell |
| Lattice constant \(a\) | 5.43 Å → cell \((a/\sqrt2,\,a/\sqrt2,\,a)\) |
| Basis atoms (frac) | \((0,0,0)\), \((½,½,½)\), \((½,0,¼)\), \((0,½,¾)\); displaced: atom 1 \(+0.02\,\mathbf c\) |
| Density cutoff | 80 Ry |
| XC | `XcKind::PADE` |
| SCF tol | \(10^{-7}\) Ha |
| Strings | `k_perp = 1` (1 string/direction), `k_string = 4` |
| Occupied | `n_occ = 8` (closed shell) |

Ionic bookkeeping (exact): ideal \(p^\mathrm{ion}=(4,4,6)\);
displaced \(\Delta p^\mathrm{ion}_z = 4\times0.02 = 0.08\).

Measured (Release, OMP=8, 2026-08-01): ideal phases \(\varphi=0\) per
direction (quantized), \(p=(0,0,0)\); displaced \(\Delta p_z=+5.5\times10^{-5}\),
\(Z^*=+0.0028\).

### Acceptance (hard-coded in `reg_periodic-berry-gpw.cpp`)

1. All three ideal-cell directional SCFs and the displaced-cell z SCF converge.
2. Ionic bookkeeping exact (tol 1e-9).
3. Centrosymmetric pin: distance of every ideal-Si \(p_i\) to the nearest
   multiple of 0.5 < **0.08** \(e\).
4. Born-charge pin: \(|Z^*| = |\Delta p_z|/\delta < 0.5\).
5. Entire section completes in ≤60 s at the registered 8-thread width.

### CP2K twin

Matched algorithm class: Quickstep GPW + `&MOMENTS PERIODIC T` on the same
BCT Si geometry / GTH-PADE / MOLOPT / PADE / 80 Ry cutoff / \(2\times2\times2\)
k-mesh. Input:
`tools/cp2k_oracle/inputs/si_periodic_berry_gpw.inp`.

Absolute Berry phases need not match the ATV string discretization number-
for-number (quantum ambiguity + Resta Γ-limit vs finite string mesh); the
symmetry pins above are the dual-gate correctness content. Fair wall is the
same single twin ENERGY job (1 MPI × fixed OMP) vs **one** ATV string-mesh
SCF+Berry direction (not the multi-gate regression wall).

#### Offline pin record

| Field | Value |
| ----- | ----- |
| Binary | `cp2k.psmp` (source `…/cp2k/install/env.sh`) |
| Version | CP2K 2026.1 `git:5e54ba2` |
| Input | `tools/cp2k_oracle/inputs/si_periodic_berry_gpw.inp` |
| Command | `OMP_NUM_THREADS=8 mpirun -np 1 cp2k.psmp -i si_periodic_berry_gpw.inp` |
| Output line | `ENERGY\| Total FORCE_EVAL ( QS ) energy [hartree] -15.914904501447854` |
| Moments path | `qs_scf_post_moments` / `qs_moment_kpoints` / `Periodic Dipole Matrix Elements` |
| Host | arrhenius1 (2026-08-01) |
| `T_oracle` | **1.18 s** (wall of the single ENERGY job) |
| `T_atv` | **11.3 s** (one string-mesh SCF+Berry direction, OMP=8; multi-gate reg wall ≈45 s / 4 SCFs) |
| `ratio` | \(T_\mathrm{atv}/T_\mathrm{oracle} = 11.3/1.18 \approx \mathbf{9.6}\) |

### Historical external performance record

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| Gate | \(r = T_\mathrm{atv}/T_\mathrm{cp2k} \le 10\) on the matched twin |
| `within_10x` | **true** (\(r \approx 9.6\)) |


## Reference — Collinear UKS Berry-phase (KSV) polarization, bulk Si (GPW)

### Method

King–Smith–Vanderbilt formal polarization on the **collinear UKS** CPU GPW
path (`cpu::gpw::GpwDft` with `n_alpha`/`n_beta`, fixed occupations). Each
spin channel has its own occupied manifold; string phases
\(\varphi_{i,s}\) are formed independently and

\[
p^\mathrm{el}_i = \frac{\varphi_{i,\uparrow}+\varphi_{i,\downarrow}}{2\pi}
\quad\text{(spin factor \(f=1\) per channel)}.
\]

Total formal polarization wraps with quantum \(1\) (nspin=2 convention).
Ionic bookkeeping is spin-independent. Multi-k string-mesh SCF uses
AO-basis Bloch MOs and phased AO overlaps \(S\) carrying
\(e^{-i\mathbf b\cdot\mathbf r}\) (King-Smith & Vanderbilt, *Phys. Rev. B*
**47**, 1651 (1993); Resta, *Rev. Mod. Phys.* **66**, 899 (1994)).

Public path:

```cpp
cpu::gpw::GpwDft method{{si_kind}, /*cutoff_ry=*/80.0, cpu::gpw::XcKind::PADE};
method.n_alpha = 8;
method.n_beta = 8;               // multiplicity 1 for diamond Si
method.n_occ = 0;
method.build(frame);             // positions [Å], Z, orthorhombic box [Å]
cpu::gpw::BerryPolarization berry; // k_perp, k_string, directions
auto r = berry.compute(method, frame);
```

No public `*Config` bag; no factories. RKS and UKS use the same exact
Gaussian Fourier-overlap kernel. The spin-independent AO link is built once
and applied to both occupied UKS manifolds. Each direction independently
runs the canonical `method.run(frame)` path on its own k-mesh with the
original convergence gates.

### Pseudopotential / basis

| Item | Value |
| ---- | ----- |
| GTH | `GTH-PBE-q4` / PADE twin class (`GTH-PADE-q4` in CP2K) |
| Basis | `DZVP-MOLOPT-GTH` (ATV tables) |
| Valence | \(Z^\mathrm{val}=4\) per Si |

### Literature pins (symmetry-exact, parameter-free)

1. **Centrosymmetric formal polarization.** A crystal with inversion
   symmetry has formal polarization on the lattice \(\{0, \tfrac12 Q\}\)
   (quantum \(Q = e\mathbf a_i/V\); Spaldin, *J. Solid State Chem.* **195**,
   2 (2012), §3; Resta & Vanderbilt, in *Physics of Ferroelectrics*, Springer
   (2007)). Diamond Si is centrosymmetric → every total \(p_i\) sits on the
   half-quantum lattice.
2. **Born effective charge of an elemental crystal is exactly zero.** The
   acoustic sum rule \(\sum_a Z^*_a = 0\) (Pick, Cohen & Martin, *Phys.
   Rev. B* **1**, 910 (1970)) plus the site-symmetry equivalence of the
   diamond sublattices give \(Z^* = 0\) per atom. A sublattice displacement
   \(\delta\) must therefore leave \(p_z\) unchanged: the electronic Berry
   phase cancels the bare ionic \(Z^\mathrm{val}\delta = 0.08\) exactly.

With multiplicity 1 and even \(N_e\), the collinear path must recover the
same physics as the closed-shell KSV strings (modulo the smaller nspin=2
polarization quantum branch choice).

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 4-atom BCT orthorhombic cell |
| Lattice constant \(a\) | 5.43 Å → cell \((a/\sqrt2,\,a/\sqrt2,\,a)\) |
| Basis atoms (frac) | \((0,0,0)\), \((½,½,½)\), \((½,0,¼)\), \((0,½,¾)\); displaced: atom 1 \(+0.02\,\mathbf c\) |
| Density cutoff | 80 Ry |
| XC | `XcKind::PADE` |
| SCF tol | \(10^{-7}\) Ha |
| Spin | collinear UKS, \(N_\uparrow=N_\downarrow=8\) |
| Strings | `k_perp = 1` (1 string/direction), `k_string = 4` |

Ionic bookkeeping (exact): ideal \(p^\mathrm{ion}=(4,4,6)\);
displaced \(\Delta p^\mathrm{ion}_z = 4\times0.02 = 0.08\).

Measured (Release, OMP=8, 2026-08-01): ideal phases \(\varphi=0\) per
direction (quantized), \(p=(0,0,0)\); displaced \(\Delta p_z=+5.3\times10^{-5}\),
\(Z^*=+0.0026\).

### Acceptance (hard-coded in `reg_periodic-berry-gpw-uks.cpp`)

1. All three ideal-cell directional UKS SCFs and the displaced-cell z SCF converge.
2. Ionic bookkeeping exact (tol 1e-9).
3. Centrosymmetric pin: distance of every ideal-Si \(p_i\) to the nearest
   multiple of 0.5 < **0.08** \(e\).
4. Born-charge pin: \(|Z^*| = |\Delta p_z|/\delta < 0.5\).
5. Entire section completes in ≤60 s at the registered 8-thread width.

### CP2K twin

Matched algorithm class: Quickstep GPW + `LSD` / `MULTIPLICITY 1` +
`&MOMENTS PERIODIC T` (Berry-phase / Resta operator) on the same BCT Si
geometry / GTH-PADE / MOLOPT / PADE / 80 Ry cutoff. Input:
`tools/cp2k_oracle/inputs/si_periodic_berry_gpw_uks.inp`.

**Why Γ for the twin.** CP2K 2026.1 `qs_moment_kpoints` (PERIODIC moments
under `&KPOINTS`) asserts closed shell only
(`CPASSERT(SIZE(matrix_ks_kp,1)==1)` in `qs_moments.F`). The collinear
UKS path therefore uses `qs_moment_berry_phase` at Γ, which loops over
`nspins`. Same method class as ATV dual-channel KSV (modern theory of
polarization); k-mesh is a twin discretization choice, not a different
representation (not GTH↔NC-UPF). Absolute phases need not match the ATV
string mesh number-for-number; the symmetry pins above are the dual-gate
correctness content.

Fair wall is the same single twin ENERGY job (1 MPI × fixed OMP) vs
**one** ATV string-mesh SCF+Berry direction (not the multi-gate regression
wall ≈ 98 s / 4 SCFs).

#### Offline pin record

| Field | Value |
| ----- | ----- |
| Binary | `cp2k.psmp` (source `…/cp2k/install/env.sh`) |
| Version | CP2K 2026.1 `git:5e54ba2` |
| Input | `tools/cp2k_oracle/inputs/si_periodic_berry_gpw_uks.inp` |
| Command | `OMP_NUM_THREADS=8 mpirun -np 1 cp2k.psmp -i si_periodic_berry_gpw_uks.inp` |
| Output line | `ENERGY\| Total FORCE_EVAL ( QS ) energy [hartree] -15.304828965893670` |
| Moments path | `qs_moment_berry_phase` / “Dipole vectors are based on the periodic (Berry phase) operator” / `Dipole moment [Debye] … Total= 0.0000000` |
| Host | arrhenius1 (2026-08-01) |
| `T_oracle` | **98.68 s** (wall of the single ENERGY+MOMENTS job) |
| `T_atv` | **21.09 s** (one string-mesh SCF+Berry direction, OMP=8) |
| `ratio` | \(T_\mathrm{atv}/T_\mathrm{oracle} = 21.09/98.68 \approx \mathbf{0.214}\) |

### Historical external performance record

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| Gate | \(r = T_\mathrm{atv}/T_\mathrm{cp2k} \le 10\) on the matched twin |
| `within_10x` | **true** (\(r \approx 0.214\)) |
