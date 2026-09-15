# Periodic Γ-point GPW MP2 — sc-Si₂ / DZVP-MOLOPT-GTH (canonical, PBE reference)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-mp2-si` |
| Regression | `regressions/gpw/periodic-mp2-si/` |
| Public API | `cpu::gpw::GpwDft::mp2(atoms)` → `cpu::gpw::Mp2Result` (`cpu/gpw/mp2.h`); members `mp2_n_virt`, `mp2_os_scale`, `mp2_ss_scale` |
| Roadmap | roadmap §10 ○ Periodic MP2 foothold; unblocks `periodic-rpa-si` |

## Summary

**Second-order Møller–Plesset correlation under periodic boundary conditions**,
computed on top of a converged Γ-point closed-shell GPW SCF with the
occupied–virtual MO repulsion integrals \((ia|jb)\) evaluated **the Quickstep
way** (Del Ben, Hutter, VandeVondele, *JCTC* **8**, 4177 (2012); CP2K
`&WF_CORRELATION` MP2 `METHOD DIRECT_CANONICAL`): collocate the pair density
\(\rho_{ia} = \psi_i\psi_a\) on the density grid, solve the periodic FFT
Poisson equation once per pair, and contract with \(\rho_{jb}\) by grid
quadrature. v1 is canonical (non-RI) — exact within the grid resolution for a
tiny cell; the RI factorization with a cFIT auxiliary basis is a scale-up
follow-on. The same MO-ERI machinery + denominators are what the periodic RPA
foothold consumes.

## Method

Γ-point periodic MOs come from the converged GPW RKS solve — real
lattice-summed AOs \(\chi_\mu(\mathbf r) = \sum_{\mathbf T}
\varphi_\mu(\mathbf r - \mathbf T)\), canonical eigenvalues \(\varepsilon_p\).
Occupied–virtual pair densities are **neutral**
(\(\int \rho_{ia} = S_{ia} = 0\)), so the periodic Coulomb operator with the
G=0 component dropped is divergence-free at Γ:

\[
(ia|jb) = \sum_g w\, V_{ia}(\mathbf r_g)\, \rho_{jb}(\mathbf r_g),
\qquad
\hat\nabla^2 V_{ia} = -4\pi \rho_{ia}.
\]

Canonical closed-shell doubles sum with OS/SS decomposition (identical
convention to molecular `cpu::scf::MP2Result`):

\[
E^{(2)} = -\sum_{ijab}
  \frac{(ia|jb)\,[\,2(ia|jb) - (ib|ja)\,]}
       {\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j}
      = E_{\mathrm{OS}} + E_{\mathrm{SS}} .
\]

The correlation is evaluated on KS (PBE) orbitals — the standard periodic
foothold (RPA/double-hybrid ingredient); a true MP2@HF reference awaits
periodic exact exchange (blocked row `periodic-hybrid-gpw`). Cost:
\(n_{\mathrm{occ}} n_{\mathrm{virt}}\) Poisson solves +
\(\mathcal O(n_{\mathrm{ov}}^2 N_g)\) quadratures.

## Public API

```cpp
cpu::gpw::GpwDft m;                    // concrete method — members as params
m.cell[0] = m.cell[1] = m.cell[2] = a; // Bohr
m.cutoff_ry = 30.0;
m.n_occ = 4;
m.xc = cpu::gpw::XcKind::PBE;
m.level_shift = 0.3;                   // convergence aid (eps unshifted)
m.build(atoms);
m.run(atoms);                          // converged Γ RKS

m.mp2_n_virt = 0;                      // 0 = all kept virtuals
const cpu::gpw::Mp2Result& r = m.mp2(atoms);
// r.e_corr (< 0), r.e_corr_os / r.e_corr_ss, r.e_total = r.e_scf + r.e_corr
```

Guards: `std::logic_error` before `build()`/converged `run()`;
`std::invalid_argument` for UKS, for `use_ot` (no canonical virtuals), and
for a vanishing HOMO–LUMO gap (< 1e-6 Ha). No public `*Config`, no factory —
kernels are private (`cpu/gpw/internal/mp2_kernels.h`).

## Validation

- **Unit tier** (`tests/cpu/gpw/test_cpu_gpw_mp2.cpp`): closed-shell doubles
  sum ≡ explicit spin-orbital MP2 sum (1e-12); MO-on-grid ≡ explicit
  Γ-image-summed AO contraction (1e-9); \((pp|pp)\) ≡ 2 × Poisson Hartree
  energy of the same pair density (1e-10); guard throws.
- **E2E** (`regressions/gpw/periodic-mp2-si/`): periodic Γ sc-Si₂
  (a = 5.43 Å, DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE, 30 Ry): SCF converged;
  E_corr < 0; virtual-space monotonicity E_corr(22) ≤ E_corr(8) < 0;
  OS+SS ≡ E_corr; frozen first-green band E_corr = −0.2764 Ha ± 1e-3 until
  the offline CP2K twin re-pin (`tools/cp2k_oracle/inputs/periodic-mp2-si.inp`).
  Fair wall vs CP2K is **perf_exempt** (no CP2K binary on the pin host).

## Follow-ons

RI factorization (cFIT aux tables), k-point sampling, UMP2, `periodic-rpa-si`
(direct consumer), physical diamond-Si₈ deck at production cutoffs.
