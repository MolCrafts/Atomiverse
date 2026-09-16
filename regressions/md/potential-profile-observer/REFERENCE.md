# Reference — slab electrostatic potential profile \(\Psi(z)\) (E10)

## oracle

```text
oracle: literature
oracle_version: planar Poisson / parallel-plate capacitor (Jackson Classical Electrodynamics ch.1 continuum); MetalWalls JOSS 5(53) 2373 (2020) potential-profile analysis tools; Scalfi et al. Phys. Chem. Chem. Phys. 22, 10480 (2020) EDL profiles
```

No MetalWalls binary and no LAMMPS built-in planar Poisson \(\Psi(z)\) compute
on the agent host. The offline pin is the closed-form planar Poisson
integration (MD metal units) on a synthetic two-sheet parallel-plate geometry
(proxy for a const-pot electrode slab charge density profile).

## Physical statement

For planar symmetry the electrostatic potential satisfies

\[
\frac{\mathrm{d} E_z}{\mathrm{d} z}
  = 4\pi\,k_{\mathrm{e}}\,\rho(z)
  = \frac{\rho(z)}{\varepsilon_0},
\qquad
E_z = -\frac{\mathrm{d}\Psi}{\mathrm{d} z},
\]

with \(k_{\mathrm{e}} = 14.399645\,\mathrm{eV\cdot Å\cdot e^{-2}}\)
(`units::coulomb`, LAMMPS metal `qqr2e`), \(\rho\) [e/Å³], \(E_z\) [V/Å],
\(\Psi\) [V]. Gauge: \(\Psi(z_{\min})=0\), \(E_z(z_{\min})=0\).

Ideal infinite parallel-plate capacitor with surface charge \(\pm\sigma\):

\[
E_{\mathrm{gap}} = 4\pi k_{\mathrm{e}}\,\sigma
  \qquad\text{(between plates; 0 outside)}.
\]

## Offline pin geometry

| Quantity | Value |
| --- | ---: |
| \(L_x = L_y\) | \(10\) Å |
| \(L_z\) | \(20\) Å |
| Area \(A\) | \(100\) Å² |
| Bin window | \(z\in[0,20)\) Å |
| \(n_{\mathrm{bins}}\) | \(20\) |
| \(\Delta z\) | \(1\) Å |
| Left sheet | \(Q=+1\,e\) at \(z=5.5\) Å (bin 5) |
| Right sheet | \(Q=-1\,e\) at \(z=14.5\) Å (bin 14) |
| \(\sigma = Q/A\) | \(0.01\) e/Å² |

## Closed-form / discrete pin

\[
E_{\mathrm{gap}}^{\mathrm{pin}}
  = 4\pi\times 14.399645\times 0.01
  = 1.80951275785204\,\mathrm{V/Å}.
\]

Piecewise-constant \(\rho\) integration (implementation algorithm) with the
two sheets as uniform density in bins 5 and 14:

| Quantity | Pin value |
| --- | ---: |
| \(E_z\) mid-gap (bin 9, \(z=9.5\)) | \(1.80951275785204\) V/Å |
| \(E_z\) vacuum left (bin 0) | \(0\) |
| \(E_z\) vacuum right (bin 19) | \(0\) |
| \(\Psi\) bin 0 | \(0\) V |
| \(\Psi\) bin 19 (= potential drop) | \(-16.28561482066836\) V \(= -E_{\mathrm{gap}}\times 9\) |
| \(\rho\) bin 5 | \(0.01\) e/Å³ |
| \(\rho\) bin 14 | \(-0.01\) e/Å³ |

Recorded offline (Python, same \(k_{\mathrm{e}}\) / \(\pi\) as ATV):

```text
PIN_E_GAP           1.80951275785204
PIN_PSI_DROP        -16.28561482066836
PIN_RHO_LEFT        0.01
PIN_RHO_RIGHT       -0.01
```

Continuum cross-check: mid-gap \(E\) matches \(4\pi k_{\mathrm{e}}\sigma\)
exactly under this binning; potential drop equals \(E_{\mathrm{gap}}\) times
center-to-center plate separation (\(9\) Å).

## Tolerances

| Check | Tol | Rationale |
| --- | ---: | --- |
| \(E_{\mathrm{gap}}\), \(\Psi\) drop, \(\rho\) bins | \(1\times 10^{-12}\) absolute | Exact double arithmetic of discrete scheme |
| Continuum \(E\) identity | \(1\times 10^{-12}\) | Algebraic identity with `units::coulomb` |

## Perf

```text
perf_exempt: true
perf_exempt_reason: Analysis Observer is O(N_atoms) charge binning + O(n_bins) planar Poisson integration on a 2-atom synthetic capacitor. No MetalWalls or LAMMPS twin binary for this named potential-profile post-process on the agent host; fair wall vs an oracle binary is not applicable.
```
