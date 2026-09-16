# Reference — Berry-phase (King–Smith–Vanderbilt) polarization, bulk Si (GPW)

## Method

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
cpu::gpw::GpwDft method;                 // cell, cutoff, n_occ, xc, …
method.build(atoms);
cpu::gpw::BerryPolarization berry;       // members: k_perp, k_string, directions
auto r = berry.compute(method, atoms);
// r.phase_el / r.p_el_frac / r.p_ion_frac / r.p_frac [e] / r.dipole_e_ang [e Å]
```

No public `*Config` bag; no factories.

## Pseudopotential / basis

| Item | Value |
| ---- | ----- |
| GTH | `GTH-PBE-q4` / PADE twin class (`GTH-PADE-q4` in CP2K) |
| Basis | `DZVP-MOLOPT-GTH` (ATV tables) / `DZVP-MOLOPT-SR-GTH` (CP2K) |
| Valence | \(Z^\mathrm{val}=4\) per Si |

## Literature pins (symmetry-exact, parameter-free)

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

## Twin geometry and settings

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

## Acceptance (hard-coded in `reg_periodic-berry-gpw.cpp`)

1. Both SCF runs converge.
2. Ionic bookkeeping exact (tol 1e-9).
3. Centrosymmetric pin: distance of every ideal-Si \(p_i\) to the nearest
   multiple of 0.5 < **0.08** \(e\).
4. Born-charge pin: \(|Z^*| = |\Delta p_z|/\delta < 0.5\).

## CP2K twin

Matched algorithm class: Quickstep GPW + `&MOMENTS PERIODIC T` on the same
BCT Si geometry / GTH-PADE / MOLOPT / PADE / 80 Ry cutoff / \(2\times2\times2\)
k-mesh. Input:
`tools/cp2k_oracle/inputs/si_periodic_berry_gpw.inp`.

Absolute Berry phases need not match the ATV string discretization number-
for-number (quantum ambiguity + Resta Γ-limit vs finite string mesh); the
symmetry pins above are the dual-gate correctness content. Fair wall is the
same single twin ENERGY job (1 MPI × fixed OMP) vs **one** ATV string-mesh
SCF+Berry direction (not the multi-gate regression wall).

### Offline pin record

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

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| Gate | \(r = T_\mathrm{atv}/T_\mathrm{cp2k} \le 10\) on the matched twin |
| `within_10x` | **true** (\(r \approx 9.6\)) |
