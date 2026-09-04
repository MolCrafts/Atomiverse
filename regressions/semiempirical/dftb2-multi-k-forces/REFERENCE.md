# dftb2-multi-k-forces — analytic Bloch-sum forces + cell stress (2×1×1 MP mesh)

```text
oracle: literature
oracle_version: Elstner et al., Phys. Rev. B 58, 7260 (1998); Hourahine et al.,
  J. Chem. Phys. 152, 124101 (2020) — Bloch-sum SCC-DFTB2 forces
family_oracle: dftb+
```

**W does not enter.** `multiplicity = 1` (closed shell); `spin_w_ha` is
ignored. This spec (`dftb-data-00-honesty`) did **not** move the energy.
The numbers below are the current ATV twin (transcribed mio-1-1 C–C, Ewald
\(\gamma\)), matching `reg_dftb2-multi-k-forces.cpp`. They supersede the
2026-07-29 REFERENCE snapshot −104.6601079822 eV, which was already stale
versus the cpp pin; that move is not a spin-constant effect and is not
labelled as one.

Absolute agreement with DFTB+ is **not** claimed (no offline 2×1×1 force
twin was re-taken here). The scientific pin is the **module-boundary
identity**: analytic multi-k \(F=-\nabla E\) of the Bloch-phased,
lattice-summed functional, vs re-SCC FD of that same functional. Absolute
\(E\) / \(F_x\) are frozen as an ATV-twin snapshot so silent Hamiltonian
drift fails the gate.

## Method

Self-consistent-charge DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260
(1998), DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260))
on a Monkhorst–Pack k-mesh; periodic force assembly as in DFTB+ (Hourahine
et al., *J. Chem. Phys.* **152**, 124101 (2020), DOI
[10.1063/1.5143190](https://doi.org/10.1063/1.5143190)).

Energy (unchanged, `dftb2-kpoints`):

\[
E=\sum_k w_k\,\mathrm{Re}\,\mathrm{Tr}\!\left[P(k)H^0(k)\right]
 +\tfrac12\sum_{AB}\gamma_{AB}\Delta q_A\Delta q_B+E_{\mathrm{rep}},
\qquad
H^0(k)_{\mu a,\nu b}=\sum_{\mathbf T}e^{2\pi i\,\kappa\cdot n_{\mathbf T}}
   h_{\mu\nu}(\mathbf R_b+\mathbf T-\mathbf R_a).
\]

Gradient (this case). Substituting the Bloch sum and using \(P(k)^\dagger=P(k)\)
collapses the band term onto **real** image-resolved density blocks

\[
\tilde P_{ij}(a,b,\mathbf T)=\sum_k w_k\,
  \mathrm{Re}\!\left[P(k)_{i_a,j_b}e^{-2\pi i\,\kappa\cdot n_{\mathbf T}}\right],
\qquad
\tilde W\ \text{likewise from}\ W(k)=2\sum_o\varepsilon_o C_{\cdot o}C^{\dagger}_{o\cdot},
\]

so each ordered pair \(a<b\) and image \(\mathbf T\) with bond
\(\mathbf d=\mathbf R_b+\mathbf T-\mathbf R_a\) contributes

\[
\frac{\partial E}{\partial d_\alpha}=
2\sum_{ij}\Big[\tilde P_{ij}\big(\partial_\alpha h_{ij}
+\tfrac12(V_a+V_b)\partial_\alpha s_{ij}\big)-\tilde W_{ij}\partial_\alpha s_{ij}\Big]
+\Delta q_a\Delta q_b\,\partial_\alpha\gamma_{ab}
+\partial_\alpha E^{\mathrm{rep}}_{ab},
\]

i.e. Hellmann–Feynman \(+\) Pulay \(+\) k-independent SCC \(\gamma\) \(+\)
repulsive — the **same** pair form the Γ path uses, with \(2P\to\tilde P\) and
\(2W\to\tilde W\). At Γ it is an identity, not an approximation, and the
\(\Delta q\) contracted by the \(\gamma\) term is the k-integrated Mulliken
population.

The \(\gamma\) and repulsive lattice sums are k-independent and reused verbatim
from the Γ-PBC path (`dftb2-periodic-forces`). Self-images
(\(A=A+\mathbf T\), \(\mathbf T\ne 0\)) have bond \(\mathbf T\), independent of
every nuclear coordinate, hence contribute zero nuclear force.

**Cell stress.** The Monkhorst–Pack mesh is defined in **fractional**
coordinates, so an affine strain \(\mathbf r\to(1+\varepsilon)\mathbf r\) leaves
every \(\kappa\) and every Bloch phase untouched and enters only through the
image bonds \(\mathbf d\to(1+\varepsilon)\mathbf d\). The intensive stress is
therefore the pair virial of exactly the terms above,

\[
\sigma_{\alpha\beta}=-\frac{1}{\Omega}\sum_{\text{bonds}}
   \frac{\partial E}{\partial d_\alpha}\,d_\beta ,
\]

Voigt \((xx,yy,zz,xy,xz,yz)\) in eV/Å³, mechanical convention
\(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\). Self-images carry **no**
force but a real strain response (\(|\mathbf T|\) scales with the cell), so the
virial — unlike the force — walks them explicitly. Landing this closed the
matching hole in the Γ virial as well (see *Closed while landing this case*).

**Reality of the k-sum.** \(P(-k)=\overline{P(k)}\) for a real real-space
Hamiltonian and \(e^{-i\theta(-k)}=\overline{e^{-i\theta(k)}}\), so a ±k pair
contributes \(w\,\mathrm{Re}(\cdot)\) exactly and the imaginary part cancels.
The implementation **throws** if it does not: the residue is accumulated and
gated (`1e-9`) for the two cases where it must vanish on its own — an unfolded
(`k_time_reversal = false`) mesh, and self-paired k (\(\kappa_d\in\{0,\pm\tfrac12\}\),
e.g. Γ in an odd mesh).

## Data

In-tree transcribed mio-1-1 C–C (`DftbSkSet::MioSketch`, historical name).

## Geometry

Off-equilibrium 1-D carbon dimer crystal (two C per orthorhombic cell) — same
cell as `dftb2-periodic-forces`, so the Γ twin is directly comparable:

| Atom | Z | x [Å] | y [Å] | z [Å] |
|------|---|-------|-------|-------|
| C    | 6 | 0.00  | 0.00  | 0.00  |
| C    | 6 | 1.20  | 0.00  | 0.00  |

Cell: orthorhombic \(L_x = 2.8\) Å, \(L_y = L_z = 12.0\) Å.

With \(r(\mathrm{C–C}) = 1.20 \ne L_x/2\), the left/right lattice images sit at
1.20 Å and 1.60 Å (plus farther shells inside \(r_{\mathrm{cut}} = 4.5\) Å), so
\(F_x \ne 0\). The equilibrium twin \(r = 1.4\) has vanishing force by symmetry
and is not used here.

## k-mesh

- `k_mesh = {2, 1, 1}` (sampling along the chain axis)
- `k_time_reversal = true` (default; forces are invariant under the flag —
  unit-gated in `tests/cpu/semiempirical/test_dftb2.cpp`
  `KMeshForcesAreTimeReversalInvariant`)
- Convention \(\kappa_d = (2r - N_d - 1)/(2N_d)\), \(r = 1..N_d\)
  (CP2K / DFTB+ `SupercellFolding`)

Offline DFTB+ twin input shape:
`KPointsAndWeights = SupercellFolding { 2 0 0; 0 1 0; 0 0 1; 0 0 0 }`,
`Forces = Yes`.

## SCC settings

- `scc_tol = 1e-12`
- `max_scc = 80`
- `with_forces = true`
- `with_stress = true`
- `sk_set = MioSketch` (default), `d3 = false`, `charge = 0`,
  `multiplicity = 1`

## Pins (hard-coded gates)

| Gate | Criterion |
|------|-----------|
| Energy finite | \(\|E\| > 10^{-3}\) eV |
| Forces finite | all components finite; \(\|F\|^2 > 10^{-8}\) |
| Translation | \(\|\sum_i F_i\| \le 10^{-10}\) eV/Å per Cartesian |
| **Force FD identity** | analytic \(F_{0,x}\), \(F_{1,x}\) vs re-SCC central FD of the **multi-k** energy (\(h = 10^{-4}\) Å) within **1×10⁻⁶ eV/Å** |
| **Stress FD identity** | analytic \(\sigma_{xx}\), \(\sigma_{yy}\) vs re-SCC **affine strain**-FD of the multi-k energy (\(h = 2\times10^{-4}\)) within **1×10⁻⁶ eV/Å³**; all six components finite |
| Multi-k ≠ Γ | \(\|F_{0,x}^{2\times1\times1} - F_{0,x}^{\Gamma}\| > 10^{-3}\) eV/Å |
| Energy pin | \(E = -91.8503968025\) eV ± 0.05 |
| Force pin | \(F_{0,x} = -11.1318394253\) eV/Å ± 0.05 |

Measured (2026-09-04, `-DCMAKE_BUILD_TYPE=Release`, 1 thread) — **unchanged
by published \(W\)** (`multiplicity = 1`):

```text
DFTB2 multi-k (2x1x1) C-chain forces: E=-91.8503968025 eV
  F[0]=(-11.1318394253, -0.0000000000,  0.0000000000) eV/A
  F[1]=( 11.1318394253,  0.0000000000, -0.0000000000) eV/A
  F0x analytic=-11.1318394253  fd=-11.1318416498  |d|=2.225e-06
  F1x analytic= 11.1318394253  fd= 11.1318416501  |d|=2.225e-06
  sigma=(-2.4281077193e-02, ~0, ~0, ~0, ~0, ~0) eV/A^3
  sigma_xx analytic=-2.4281077193e-02  strain-fd=-2.4281094545e-02  |d|=1.735e-08
  sigma_yy analytic= 0                 strain-fd= 0                 |d|=2e-32
  Γ (1x1x1): E=-93.1115376115 eV  F0x=-14.8265853692 eV/A  ΔF0x=3.694746
```

\(\sigma_{yy}=\sigma_{zz}=0\) and the shears are numerical zeros — correct for a
1-D chain along \(x\) with 12 Å of vacuum on the transverse axes (every
transverse image is beyond \(r_{\mathrm{cut}}\)).

The Γ control (\(F_{0,x}=-14.8265853692\) eV/Å, \(E=-93.1115376115\) eV) is
the same geometry as `reg_dftb2-periodic-forces` / `reg_dftb3-periodic-forces`
(homonuclear ⇒ DFTB3 reduces to DFTB2). The 3.69 eV/Å shift is the physical
effect of sampling the 1-D band beyond Γ.

## Public API

```cpp
cpu::semiempirical::DFTB2 dftb;
dftb.k_mesh = {2, 1, 1};       // members only — no *Config, no make_*/build_*
dftb.with_forces = true;
dftb.with_stress = true;
dftb.scc_tol = 1e-12;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
```

## Closed while landing this case — the Γ pair virial dropped self-images

`analytic_forces_restricted` walked ordered pairs \(a<b\) only, for the force
**and** the virial. That is correct for the force (a self-image bond is
\(\mathbf T\), independent of every nuclear coordinate) and **wrong** for the
virial (\(|\mathbf T|\) scales with the cell). It was invisible to
`dftb2-analytic-stress`, whose geometry is a molecule in a 20 Å box where every
self-image is beyond \(r_{\mathrm{cut}}\) — its own Gap 2 named the missing
periodic case.

Measured on this cell before the fix (Γ, \(L_x = 2.8\) Å < \(r_{\mathrm{cut}} = 4.5\) Å):

```text
sigma_xx analytic -0.31716963   strain-FD -0.37019871   |d| 5.3e-02 eV/A^3   (14%)
```

after adding the self-image \(H^0/S/\gamma/E_\mathrm{rep}\) virial terms
(factor 1 per ordered image for the electronic block, ½ for \(\gamma_{AA}\) and
\(E_\mathrm{rep}\)): \(|d| < 10^{-6}\) eV/Å³, unit-gated by
`tests/cpu/semiempirical/test_dftb2.cpp`
`PeriodicAnalyticStressMatchesStrainFdCChain`. The molecular
`dftb2-analytic-stress` pin is bit-unchanged, as expected.

## Gaps

1. **Restricted closed shell only.** Collinear open shell at k ≠ Γ needs a dual
   α/β Bloch Fock pair plus the spin-shift \(\partial S\) term; `multiplicity > 1`
   (or an odd electron count) with a multi-k mesh throws.
2. **`ThreeOb` has no Bloch path.** The 3ob radial-table route is molecular
   (no lattice sum at all), so `sk_set = ThreeOb` with a multi-k mesh throws.
3. **Integer band filling.** \(n_{\mathrm{occ}}\) lowest bands are filled at
   every k with weight 2 — no Fermi smearing, no fractional occupation. For a
   mesh where band 4/5 cross the Fermi level between k-points the density is
   discontinuous in the nuclear coordinates and the analytic gradient is not
   \(-\nabla E\). The gated geometry has an open gap at both sampled k-points
   (which the \(10^{-6}\) FD identity demonstrates); a genuinely metallic mesh
   is out of scope.
4. **No absolute DFTB+ number.** No 2×1×1 force twin was re-taken in this
   spec; see the note at the top. W does not enter.
5. **D3 is still a molecular cluster sum** (pre-existing `DFTB2::d3`
   limitation), so `d3 = true` with a k-mesh mixes a periodic electronic
   energy with an isolated-cluster dispersion term. Untouched here.

## Perf

```text
perf_exempt: true
perf_exempt_reason: no DFTB+ 2x1x1 force twin re-taken in this spec
```

Runtime: < 0.1 s (2-atom cell, 1 irreducible k-point after ±k folding, 5 SCC
solves total including the finite differences).
