# Reference — Collinear-spin Berry-phase polarization, bulk Si

## Method

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
cpu::pw::PW method;
method.spin_polarized = true;
method.multiplicity = 1;          // N↑−N↓ = 0 for diamond Si
cpu::pw::BerryPolarization berry; // k_perp, k_string, directions
auto r = berry.compute(method, frame);
```

No public `*Config` bag; no factories. Closed-shell path
(`spin_polarized = false`) is unchanged — see
`regressions/pw/berry-polarization`.

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |

## Literature pins (symmetry-exact, parameter-free)

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

## Twin geometry and settings

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

## Acceptance (hard-coded in `reg_berry-spin-si.cpp`)

1. Both collinear SCF runs converge.
2. Ionic bookkeeping exact: ideal \(p^\mathrm{ion}=(1,1,1)\) (tol 1e-9);
   displaced \(\Delta p^\mathrm{ion}_3=0.08\) (tol 1e-9).
3. Centrosymmetric pin: distance of every ideal \(p_i\) to nearest
   multiple of 0.5 < **0.05**.
4. Born-charge pin: \(|Z^*|<0.5\).

## CP2K note

Offline twin echoes the closed-shell Berry note with an explicit UKS /
LSD spin section for documentation — absolute phases still not comparable
to ONCV PW; symmetry pins are the scientific content.

Input echo: `tools/cp2k_oracle/inputs/si_berry_spin_note.inp`.

## Performance

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
