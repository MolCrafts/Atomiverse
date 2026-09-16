# Reference — Γ-point canonical GPW MP2 on periodic Si₂ (DZVP-MOLOPT-GTH)

## Method

Second-order Møller–Plesset correlation energy on top of a converged
**periodic Γ-point closed-shell GPW PBE** SCF, with the occupied–virtual MO
repulsion integrals \((ia|jb)\) computed the Quickstep way (Del Ben, Hutter,
VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012); CP2K
`&WF_CORRELATION` MP2 `METHOD DIRECT_CANONICAL`):

1. pair density \(\rho_{ia}(\mathbf r) = \psi_i(\mathbf r)\psi_a(\mathbf r)\)
   collocated on the SCF density grid (Γ lattice-image-summed AOs);
2. one periodic FFT Poisson solve per \((i,a)\) pair (G=0 dropped —
   divergence-free at Γ because \(\int\rho_{ia} = S_{ia} = 0\));
3. \((ia|jb) = \sum_g w\, V_{ia}(g)\, \rho_{jb}(g)\) grid quadrature;
4. canonical closed-shell doubles sum
   \(E^{(2)} = -\sum_{ijab} (ia|jb)[2(ia|jb) - (ib|ja)]/\Delta_{ij}^{ab}\)
   with OS/SS decomposition (molecular `MP2Result` convention).

v1 is canonical (non-RI): for this tiny cell the full tensor needs only
`n_occ·n_virt` = 88 Poisson solves. MP2 on KS (PBE) orbitals — a true MP2@HF
reference awaits periodic exact exchange (blocked row `periodic-hybrid-gpw`).

Atomiverse public path (no Config bag, no factory):

```cpp
cpu::gpw::GpwDft m;            // members below; build(); run(atoms)
const cpu::gpw::Mp2Result& r = m.mp2(atoms);   // Γ-point GPW MP2
```

## System

| Item | Value |
|------|--------|
| Crystal | **artificial sc-Si₂**: 2 Si in a simple-cubic cell (diamond-fragment motif) |
| Cell | cubic **a = 5.43 Å** (10.26126 Bohr), fully periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4, a/4, a/4) |
| Basis | **DZVP-MOLOPT-GTH** (Si; VandeVondele–Hutter 2007) |
| Pseudopotential | **GTH-PBE-q4** (Krack 2005) → 8 valence e⁻, n_occ = 4 |
| XC (SCF reference) | **PBE** |
| Density cutoff | **30 Ry** (single grid; e2e ≤ 60 s budget) |
| SCF | tol 1e-6 Ha, mix 0.3, DIIS 6, **level_shift 0.3 Ha**, max_iter 80 |
| MP2 | all 22 kept virtuals (lindep 1e-8); OS/SS scales 1.0 |

**Note (small gap):** Γ-only sc-Si₂ is an artificial small-gap crystal —
converged KS gap ≈ **0.015 Ha** (unshifted eigenvalues; the level shift is a
convergence aid only). SCF without the shift oscillates > 300 iters. The gap
is far above the `GpwDft::mp2` 1e-6 Ha degeneracy guard, so the canonical
denominators are well conditioned. A physical diamond-Si₈ cube exceeds the
regression budget (≈ 16× the per-iteration cost) — production-depth follow-on.

## First-green values (Atomiverse, this deck, 2026-07-25)

| Quantity | Value [Ha] |
|----------|------------|
| E_SCF (PBE, periodic Γ) | **−7.5955445070** |
| E_corr (MP2, unscaled) | **−0.2764036524** |
| E_corr OS | −0.2378754296 |
| E_corr SS | −0.0385282228 |
| E_total | −7.8719481594 |
| E_corr (n_virt = 8) | −0.1668178815 |

Wall (4 OpenMP threads, EPYC login node): SCF ≈ 15 s, MP2 ≈ 0.5 s.

## Offline oracle status

**No CP2K binary / module on the worker host at pin time** (`cp2k`,
`cp2k.psmp`, `cp2k.ssmp` not on PATH; `module avail cp2k` empty — same status
as every prior wave). Absolute CP2K digits are therefore **not invented** and
**no live oracle** runs at ctest/regression time.

Pin strategy (binding, mirrors `gpw-pbesol-si`):

1. **Hard internal gates** (regression asserts all):
   - SCF converged; E_corr finite and **strictly negative**;
   - **virtual-space monotonicity** E_corr(22 virt) ≤ E_corr(8 virt) < 0
     (every added virtual contributes a non-positive term);
   - OS + SS ≡ E_corr and E_total ≡ E_SCF + E_corr to 1e-12;
   - unit tier separately pins the closed-shell sum against an explicit
     spin-orbital MP2 sum and the (ia|jb) machinery against the Poisson
     Hartree energy (`tests/cpu/gpw/test_cpu_gpw_mp2.cpp`).
2. **Frozen first-green band** |E_corr − (−0.2764036524)| ≤ 1e-3 Ha —
   setup-integrity band, not a third-party oracle.
3. **Offline re-pin:** when CP2K is available, run
   `tools/cp2k_oracle/inputs/periodic-mp2-si.inp` (same deck: PBE, 30 Ry,
   DZVP-MOLOPT-GTH, GTH-PBE-q4, Γ, canonical MP2 over all virtuals) and
   replace the band with the CP2K `MP2 Energy` / total-energy digits, citing
   the CP2K release (prefer `CP2K 2024.1+`; note EPS_DEFAULT and any
   RI/canonical difference).

## Perf

`perf_exempt` — no CP2K binary on the pin host for a fair wall twin.
Regression wall ≈ 16 s single process (4 threads), under the ≤ 60 s cap.
