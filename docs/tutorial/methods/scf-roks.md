# Restricted open-shell Kohn–Sham DFT (ROKS)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Restricted open-shell Kohn–Sham keeps a spin-pure high-spin open-shell
determinant — exact \(\langle S^2\rangle = S(S+1)\) — where UHF/UKS break
spin symmetry. One spatial MO set has closed / open / virtual spaces with
split occupation \(n_\alpha \ge n_\beta\); SCF couples the unrestricted KS
Fock pair into one Roothaan effective Fock (Guest–Saunders / PySCF canonical
coupling, `get_roothaan_fock`), and the total energy is the coupling-free UKS
functional on that constrained RO density. It is the KS analogue of
[ROHF](scf-rohf.md). The public method is
`cpu::scf::ROKS{basis, n_alpha, n_beta, xc}` (members `tol`, `max_iter`) with
`run(frame)` for the energy and `forces(frame)` → `GradientResult` [eV/Å].

Both sections run on CPU: the energy and the analytic gradient of triplet O₂.

Case: `ctest -R '^scf/roks$'` — sources and oracle record in [`regressions/scf/roks/`](../../../regressions/scf/roks/). One section of the case per `##` heading below: `atv_regressions --case scf/roks --section <name>`.

## Common setup and oracle

| Field | Value |
| ----- | ----- |
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075); cubic box 20 Å |
| Basis | STO-3G, Cartesian (`cart=True`) |
| Method | ROKS(9,7) / LSDA (`XCType::LSDA` ≡ PySCF `LDA,VWN`) |
| Spin | triplet, \(\langle S^2\rangle = 2\) exactly |
| `conv_tol` | \(1\times 10^{-12}\), `max_iter = 200` |
| Oracle | PySCF **2.14.0** `dft.ROKS`, offline (`docs/tutorial/appendix-pyscf-oracle.md`); never in ctest |
| Build | `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU build environment `buildenv-gcc/2026.03-mpich`) |

```cpp
cpu::Frame frame(2);   // O₂ along z
frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 8;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_z)[1] = 1.2075;
frame.box = cpu::Box::cubic(20.0);

cpu::scf::ROKS roks{cpu::scf::BasisSet::STO3G, /*n_alpha=*/9, /*n_beta=*/7, cpu::scf::XCType::LSDA};
roks.tol = 1e-12;
roks.max_iter = 200;
```

## References

- Roothaan, *Rev. Mod. Phys.* **32**, 179 (1960)
- Guest & Saunders, *Mol. Phys.* **28**, 819 (1974)

## roks-o2-sto3g — ROKS energy, triplet O₂ / LSDA / STO-3G {#roks-o2-sto3g}

### Practice

```cpp
roks.run(frame);   // O₂ r = 1.2075 Å
```

### Results

#### Accuracy vs PySCF 2.14.0 (`dft.ROKS`, `xc='LDA,VWN'`)

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{ROKS}}\) | **−147.193715448518** | **1e-6** |
| \(\langle S^2\rangle\) | 2.0 (exact high-spin) | identity |

#### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `rohf` | 0.45 s | 0.59 s | **1.31** | ✓ |

## roks-grad-o2-sto3g — ROKS analytic gradient, triplet O₂ / LSDA / STO-3G {#roks-grad-o2-sto3g}

### Summary

ROKS analytic nuclear forces on high-spin O₂. `ROKS::forces` returns a
`GradientResult` in eV/Å; the section converts to Ha/Bohr for the pin
(PySCF `dft.ROKS.nuc_grad_method`). Fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 1.117\). The ROHF counterpart is
the `o2_rohf_grad` section of [scf/rohf](scf-rohf.md).

### Method

**Gradient composition.**

- The electronic gradient reuses the **unrestricted KS** path: Coulomb \(J\),
  hybrid \(K\) (if any), and XC grid response on the split-occupation
  \(\alpha/\beta\) densities from the shared spatial orbitals (PySCF
  `grad/uks.py` / `grad_elec` shape).
- The ROKS-specific piece is the **energy-weighted density** (same Lagrangian
  form as ROHF):

\[
W = O_\alpha F_\alpha O_\alpha + O_\beta F_\beta O_\beta,
\]

  with **KS spin Focks** \(F_\sigma\), not the Roothaan effective Fock and not
  \(\sum_i \varepsilon_i C_i C_i^\top\). This matches PySCF `make_rdm1e` on
  the KS Fock pair. A naive effective-Fock eigenvalue \(W\) fails the pin by
  \(\gtrsim 10^{-1}\) Ha/Bohr (the same pathology as for ROHF).

**Scope.**

- Functional: LSDA only; no hybrid or GGA on this section.
- Analytic nuclear gradient only; no Hessian, no ROMP2 forces.
- XC grid discretization: PySCF analytic vs FD of the same PySCF energy differ
  by \(\sim 1\times 10^{-5}\) Ha/Bohr here; ATV analytic tracks the **FD**
  surface more closely (see Validation).

### Implementation

```cpp
const cpu::scf::GradientResult res = roks.forces(frame);
// res.forces(A, d) in eV/Å; convert for Ha/Bohr pins:
// units::convert(..., units::Force::eVPerAngstrom, units::Force::HartreePerBohr)
```

Headers: `roks.h`, `methods.h`, `gradient_result.h`, `frame.h`.

### Validation (vs PySCF)

Oracle: `dft.ROKS` + `nuc_grad_method().kernel()`.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{ROKS}}\) | −147.193715448518 Ha | \(1\times 10^{-6}\) Ha | as in `roks-o2-sto3g` |
| \(\partial E/\partial z\) [atom 0] | **+0.16292653404646806** Ha/Bohr | element-wise **\(2\times 10^{-5}\)** Ha/Bohr | gate |
| \(\partial E/\partial z\) [atom 1] | **−0.16292653404646806** Ha/Bohr | same | equal–opposite |
| \(\partial E/\partial x\), \(\partial E/\partial y\) | ~0 | same | diatomic on \(z\) |
| Translational invariance \(\lvert\sum_A g_A\rvert_{\max}\) | — | \(1\times 10^{-8}\) Ha/Bohr | gate |

The nominal element-wise tol is \(1\times 10^{-5}\) Ha/Bohr; the gate is
**\(2\times 10^{-5}\)** because PySCF analytic and central FD of the PySCF
energy (\(h=10^{-4}\) Bohr, atom 0 \(z\)) differ by **1.004e-5** Ha/Bohr on
this system (LDA grid response). Atomiverse analytic matches that FD value to
\(\sim 7\times 10^{-9}\) Ha/Bohr and sits \(\sim 1.005\times 10^{-5}\) from
the PySCF analytic pin. A wrong \(W\) still fails by \(\ge 10^{-1}\).

Oracle re-pin:

```bash
python <pyscf pin> roks \
    --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --conv-tol 1e-12 --grad
```

### Performance

Login node, `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=close`, median of \(n=3\).

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `roks_grad` (SCF + analytic grad) | **0.1767** | **0.1974** | **1.117** | ✓ | matched method |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\).

```bash
python <pyscf fair-bench> --method roks_grad --geom o2 \
    --basis sto-3g --spin 2 --xc LDA,VWN --repeat 3 -- <section binary>
```

**Twin caveats.** The section binary includes full element-wise +
translational checks; the PySCF side is SCF + `nuc_grad_method` only. Walls
are short (\(\sim 0.2\) s), so OS jitter can be material at this scale. The
energy section `roks-o2-sto3g` uses a **ROHF** twin (\(r\sim 1.31\)); do not
conflate that with this matched `roks_grad` pair.

### Limitations

- Single textbook case: O₂ triplet, STO-3G cart, LSDA only; no hybrid ROKS
  gradient, no GGA/meta-GGA force pin.
- The grid-response residual vs PySCF analytic forces the \(2\times 10^{-5}\)
  gate rather than a hard \(1\times 10^{-5}\) match to analytic PySCF.
- `ROKS::mp2()`, `fci()`, SOSCF, level shift, density fitting throw /
  unsupported (header contract).
- CUDA ROKS / ROKS forces: not available.
- No Hessian / vibrational analysis on the ROKS density (the ROHF Hessian is
  [`rohf-hessian-o2`](scf-rohf.md#rohf-hessian-o2)).

### References

- Johnson, Gill, Pople, *J. Chem. Phys.* **98**, 5612 (1993) (KS XC grid gradient)
- PySCF `pyscf/grad/rohf.py` (`make_rdm1e`), `pyscf/grad/uks.py` (`grad_elec`);
  `dft.ROKS`
