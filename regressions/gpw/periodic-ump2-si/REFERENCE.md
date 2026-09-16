# Reference — Γ-point unrestricted GPW MP2 (UMP2) on periodic Si₂

## Method

Second-order Møller–Plesset correlation on top of a converged **periodic
Γ-point UKS GPW PBE** SCF (equal-spin closed-shell occupations
\(n_\alpha = n_\beta = 4\)). Occupied–virtual MO integrals use the product
**RI-GPW** path (Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.*
**8**, 4177 (2012); CP2K `&WF_CORRELATION` `METHOD MP2_GPW`):

1. collocate spin-resolved pair densities
   \(\rho_{ia}^\sigma(\mathbf r)=\psi_i^\sigma(\mathbf r)\psi_a^\sigma(\mathbf r)\);
2. plane-wave Coulomb intermediates (G=0 dropped — neutral pairs);
3. same-spin αα / ββ doubles
   \(E_{\mathrm{SS}}^\sigma=\tfrac14\sum[(ia|jb)-(ib|ja)]^2/\Delta\)
   and opposite-spin αβ
   \(E_{\mathrm{OS}}=\sum(i_\alpha a_\alpha|j_\beta b_\beta)^2/\Delta\)
   (molecular / PySCF UMP2 convention).

When \(C^\alpha\approx C^\beta\) and occupations match the closed-shell
reference, **UMP2 ≡ RMP2** — dual-gated against the
`periodic-mp2-si` first-green correlation energy.

Atomiverse public path (no Config bag, no factory):

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 4; m.n_beta = 4;   // UKS
m.xc = XcKind::PBE;
m.build(atoms);
m.run(atoms);
const cpu::gpw::Mp2Result& r = m.mp2(atoms);  // UMP2@UKS
```

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same as `periodic-mp2-si`) |
| Cell | cubic **a = 5.43 Å**, fully periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4, a/4, a/4) |
| Basis | **DZVP-MOLOPT-GTH** |
| Pseudopotential | **GTH-PBE-q4** → 8 valence e⁻ |
| XC (SCF reference) | **PBE**, UKS \(n_\alpha=n_\beta=4\) |
| Density cutoff | **30 Ry** |
| SCF | tol 1e-6 Ha, mix 0.3, **level_shift 0.3 Ha**, max_iter 80 |
| MP2 | RI-GPW product (`mp2_ri = true`); all kept virtuals per spin |

## Dual-gate / first-green band

| Quantity | Value [Ha] | Notes |
|----------|------------|--------|
| E_corr (RMP2 first-green, `periodic-mp2-si`) | **−0.2764036524** | closed-shell twin |
| Regression band | \|E_corr(UMP2) − (−0.2764036524)\| ≤ **2e-3** | UKS numerical drift |
| Internal | E_corr < 0; OS+SS ≡ E_corr; E_total ≡ E_SCF+E_corr; virtual cap monotone | hard |

Unit tier (`tests/cpu/gpw/test_cpu_gpw_mp2.cpp`): UMP2 ≡ spin-orbital sum;
UMP2 ≡ RMP2 on identical α=β tensors; cross-spin RI ≡ Poisson.

## Offline oracle status

**No CP2K module on the worker host at pin time** (`module avail CP2K`
empty). Absolute CP2K UMP2 digits are **not invented**; **no live oracle**
in ctest.

Offline twin input: `tools/cp2k_oracle/inputs/periodic-ump2-si.inp`
(UKS MULTIPLICITY 1, same cell/basis/PP/cutoff, `&WF_CORRELATION` MP2).
When CP2K is available, replace the dual-gate band with CP2K
`MP2 Energy` / total energy and cite the release.

## Perf

`perf_exempt` — no CP2K binary on the pin host for a fair wall twin.
Target wall ≤ 60 s single process (SCF ≈ RKS deck + UMP2 3 tensor builds).
