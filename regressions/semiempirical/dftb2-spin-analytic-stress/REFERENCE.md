# Reference — collinear spin-DFTB2 analytic cell stress (O₂ triplet, NO doublet)

```text
oracle: dftb+
oracle_version: 24.1 (energy + forces). Cluster DFTB+ has no cell, so it
  cannot report stress; the stress gate is analytic vs strain-FD of the
  same ATV energy.
```

Offline twin: `tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-stress/`
(`geo.gen`, `dftb_in.hsd`, archived `dftb.out.txt`). Cluster (`C`) O₂ with
the regression's tilted bond \(|R|=1.30\) Å along
\(\hat u=(0.6,0.7,\sqrt{0.15})\). ATV uses `DftbBoundary::cluster` and a
cubic 20 Å box only to define \(\Omega\) for the intensive virial — that is
still a cluster Hamiltonian. A DFTB+ `S` geometry would add Ewald images ATV
does not use, so the twin stays `C`. **No DFTB+ stress number is claimed
and none is invented.**

Same spin block as `spin-dftb2-o2`. ctest never launches DFTB+.

## Method

Collinear unrestricted SCC-DFTB2:

\[
E = \mathrm{Tr}(\rho^\alpha H^0)+\mathrm{Tr}(\rho^\beta H^0)
  + \tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\Delta q_B
  + \tfrac12\sum_A W_A m_A^2
  + E_{\mathrm{rep}}
\]

Intensive cell stress (mechanical / DFTB+ / PW–GPW convention):

\[
\sigma_{\alpha\beta}=-\frac{1}{\Omega}\frac{\partial E}{\partial\varepsilon_{\alpha\beta}},
\qquad \text{Voigt } (xx,yy,zz,xy,xz,yz)\ \ [\mathrm{eV/\AA^3}]
\]

evaluated **closed form** at the SCC density as the pair virial of the same
bond terms as the analytic nuclear force (Hellmann–Feynman, Pulay including
the \(\pm W_A m_A\) half of \(\varepsilon^\sigma_A\), SCC \(\gamma\),
repulsive). \(E_{\mathrm{spin}}\) has no explicit strain dependence; the
implicit \(m_A(\varepsilon)\) response reduces variationally to that Pulay
term. It is included; it is not assumed to be zero.

## Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2` (`sk_set = MioSketch`, transcribed
  mio-1-1), `boundary = cluster`,
  `spin_w_ha = basis_data::kDftbSpinConstantsPbeHa`.

## Geometry (Å)

Both molecules use the tilted bond
\(\hat u = (0.6,\,0.7,\,\sqrt{0.15})\) so all six Voigt components are
non-trivial.

**Case 1 — O₂ triplet (headline).** `charge = 0`, `multiplicity = 3`,
\(r = 1.30\) Å:

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| O | 8 | 0.78 | 0.91 | 0.503487835 |

**Case 2 — NO doublet (discriminator).** `charge = 0`, `multiplicity = 2`,
\(r = 1.20\) Å. O₂ is degenerate: \(\Delta q\equiv 0\) and the spin shift is
uniform, so a gate built on O₂ alone would pass an implementation that never
wrote the spin virial. NO restores both \(\Delta q\neq 0\) and
\(W_{\mathrm{N}}m_{\mathrm{N}}\neq W_{\mathrm{O}}m_{\mathrm{O}}\). No DFTB+
twin is archived for NO (this spec's twin is O₂).

Isolated cluster, cubic 20 Å box (\(\Omega=8000\) Å³; every lattice image is
beyond the SK cutoff). `scc_tol = 1e-12`, `max_scc = 200`,
`with_forces = true`, `with_stress = true`.

## Pins (hard-coded gates)

Per case:

1. **Energy:** finite, \(|E| > 10^{-3}\) eV.
2. **Translation:** \(|\sum_i F_i|\le 10^{-6}\) eV/Å per Cartesian.
3. **Stress:** all six Voigt components finite.
4. **Strain-FD identity:** analytic \(\sigma_v\) vs central affine strain-FD of
   the same converged \(E\) (\(h=2\times10^{-4}\)), **all six** components,
   within **1×10⁻⁸ eV/Å³**.

Measured ATV (2026-09-04, Release, 1 thread):

```text
O2 triplet   E = -176.77425642 eV
  sigma[0] analytic=-2.87321165e-04  fd=-2.87321148e-04  |d|=1.753e-11
  sigma[1] analytic=-3.91076031e-04  fd=-3.91075973e-04  |d|=5.730e-11
  sigma[2] analytic=-1.19717152e-04  fd=-1.19717153e-04  |d|=6.681e-13
  sigma[3] analytic=-6.70416052e-04  fd=-6.70415688e-04  |d|=3.641e-10
  sigma[4] analytic=-3.70930029e-04  fd=-3.70929985e-04  |d|=4.402e-11
  sigma[5] analytic=-4.32751701e-04  fd=-4.32751627e-04  |d|=7.429e-11
NO doublet   E = -152.83740700 eV
  sigma[0] analytic=-2.67630972e-04  fd=-2.67630947e-04  |d|=2.476e-11
  sigma[1] analytic=-3.64275489e-04  fd=-3.64275402e-04  |d|=8.689e-11
  sigma[2] analytic=-1.11512905e-04  fd=-1.11512907e-04  |d|=1.906e-12
  sigma[3] analytic=-6.24472267e-04  fd=-6.24471689e-04  |d|=5.780e-10
  sigma[4] analytic=-3.45510099e-04  fd=-3.45510034e-04  |d|=6.500e-11
  sigma[5] analytic=-4.03095115e-04  fd=-4.03095004e-04  |d|=1.115e-10
```

## Offline DFTB+ 24.1 (O₂ only)

From archived `dftb.out.txt`. Same \(|R|=1.30\) Å as the linear force twin,
so the energy is identical (orientation does not enter a cluster Hamiltonian):

| Quantity | DFTB+ | Converted |
| -------- | ----- | --------- |
| Total energy | −6.4963341184 H | **−176.774256878772 eV** |
| Energy SPIN | −0.0280000000 H | −0.761918814888 eV |
| \(F_1\) | (0.057307719745, 0.066859006948, 0.036991973513) Ha/Bohr | (2.94688143, 3.43802837, 1.90220376) eV/Å |
| \(F_2\) | opposite | opposite |

\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{DFTB+}}\rvert=4.6\times10^{-7}\) eV
(≤ \(10^{-3}\) ⇒ energy/forces `oracle: dftb+`). The DFTB+ force lies along
\(\hat u\) with magnitude 4.91146906 eV/Å, matching the linear 1.30 Å twin.
**No stress tensor is present in `detailed.out` or `results.tag`.**

## Gaps (still open)

1. **D3 dispersion is absent from the stress** unless `d3 = true` (the
   closed-shell D3 virial lives on `dftb2-analytic-stress`).
2. **No periodic open-shell stress twin.** This geometry is a molecule in a
   20 Å box; self-image virial is unit-gated elsewhere
   (`dftb2-multi-k-forces`).
3. **Cluster DFTB+ cannot pin \(\sigma\).** Closing an absolute stress pin
   would need a periodic Hamiltonian ATV actually uses, not a dummy `S` box.

## Citations

- M. Elstner et al., Phys. Rev. B **58**, 7260 (1998). DOI 10.1103/PhysRevB.58.7260.
- C. Köhler, G. Seifert, U. Gerstmann, M. Elstner, H. Overhof, Th. Frauenheim,
  Phys. Chem. Chem. Phys. **3**, 5109 (2001). DOI 10.1039/b105782k.
- C. Köhler, Th. Frauenheim, B. Hourahine, G. Seifert, M. Sternberg,
  J. Phys. Chem. A **111**, 5622 (2007). DOI 10.1021/jp068802p.
- B. Hourahine et al., J. Chem. Phys. **152**, 124101 (2020). DOI 10.1063/1.5143190.

## Perf

```text
perf_exempt: true
perf_exempt_reason: DFTB+ 24.1 cluster run has no stress to wall-clock
  against; analytic virial replaces 12 multi-SCC strain evaluations
```

## Units

Public ATV: energy **eV**, positions **Å**, forces **eV/Å**, stress
**eV/Å³** Voigt \((xx,yy,zz,xy,xz,yz)\).
