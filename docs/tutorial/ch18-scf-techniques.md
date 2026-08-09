# Chapter 18. SCF Convergence

Roothaan–Hall iteration is a fixed-point search: nothing guarantees that it
converges. This chapter covers three staged controls—initial guess, Fock/density
mixing, and second-order orbital updates—each written as **Theory → Practice →
Results**.

See also [Ch. 2](ch02-rhf.md) and [Ch. 3](ch03-open-shell.md).

> **Build flag.** Everything in this chapter lives behind `ATV_BUILD_HF`, which
> is **OFF by default** — with it off the `cpu::scf::…` headers, sources, tests
> and regressions are excluded from the build entirely.
>
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release \
>       -DATV_BUILD_HF=ON -DBLA_VENDOR=OpenBLAS
> cmake --build build -j$(nproc)
> ```

---

## 18.1 Why iterations fail

### Theory

Two common failure modes:

1. **Charge oscillation.** For nearly degenerate configurations \(A\) and \(B\),
   \(F[P_A]\) prefers \(B\) and \(F[P_B]\) prefers \(A\). Density flips every
   step. Small HOMO–LUMO gaps and stretched bonds favour this.
2. **Overshoot.** Updates are too large; a bare-nuclear (Hcore) first step often
   pushes charge too hard toward the nuclei.

Classical remedies:

- **Damping** \(P\leftarrow(1-\alpha)P_{\mathrm{new}}+\alpha P_{\mathrm{old}}\)
  (`damping_factor`; \(\alpha\) = fraction of *old* density kept).
- **Level shift** (Saunders–Hillier): raise virtual levels by \(\lambda\). After
  convergence the engine re-diagonalises the unshifted Fock matrix.

Recommended staging: good guess (§18.2) → far-region mixer (§18.3) → CDIIS or
SOSCF near SCF (§18.3–18.4).

### Practice

```cpp
cpu::scf::RHF rhf;
rhf.damping_factor = 0.5;  // retain half the previous density
rhf.level_shift = 0.2;     // Hartree; 0 disables
rhf.run(frame);
```

### Results

No single universal number: damping and shift trade stability for speed. Use
them when the residual oscillates or diverges; turn them off (or reduce them)
once the iteration is stable enough for CDIIS.

---

## 18.2 Initial guesses

### Theory

SCF is local. The guess chooses the basin and the starting distance to a
stationary point.

| Guess | Idea |
|---|---|
| Hcore | Diagonalise \(T+V_{ne}\). Always available; poorest. |
| MINAO / SAD | Superposition of atomic densities (default). |
| Extended Hückel | Empirical \(F\); one diagonalisation → orbitals (H–Ar tables). |
| SAP | Superposition of atomic potentials; density-free (\(Z\le 118\)). |

Open-shell spin polarisation still needs mixing/occupation after a
spin-symmetric guess.

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Def2SVP, /*charge=*/0, /*tol=*/1e-9};
rhf.init_guess = cpu::scf::InitGuess::Hueckel; // or SAP, MINAO, Hcore
rhf.run(frame);
```

### Results

Water / def2-SVP / RHF / \(10^{-9}\):

| Guess | Iterations (this engine) | Note |
|---|---:|---|
| Hcore | 16 | baseline |
| MINAO | 16 | default density guess |
| Hückel | 14 | fewer than Hcore |
| SAP | 14 | fewer than Hcore |

Compare *ordering* across codes, not absolute iteration counts (Hückel
parameterisations differ). Regression: `h2o_scf_convergence`.

---

## 18.3 Mixing: CDIIS, EDIIS, ADIIS

### Theory

**CDIIS** (Pulay) extrapolates Fock matrices from commutator errors
\(e_i=X^\top(F_iP_iS-SP_iF_i)X\). Near SCF, convergence is nearly quadratic.
Far from SCF, large coefficients can overshoot.

**EDIIS** / **ADIIS** minimise a model on the simplex \(c_i\ge 0\),
\(\sum c_i=1\) (convex combinations). They are **far-region** tools and stall
near \(10^{-4}\) residual. Intended use: ADIIS/EDIIS first, then **permanent**
switch to CDIIS when \(\max|e|\) falls below a threshold.

### Practice

```cpp
cpu::scf::UHF uhf(5, 5);
uhf.mixer = cpu::scf::MixerKind::ADIIS_CDIIS;
uhf.mixer_switch_tol = 1e-2;
uhf.run(frame);
```

Default `MixerKind::CDIIS` leaves the historical path unchanged.

### Results

Water stretched \(\times 2.6\), def2-SVP, UHF, Hcore guess (broken-symmetry
diradical target, \(\langle S^2\rangle\approx 1.97\)):

| Path | Outcome | \(E\) / Ha vs PySCF |
|---|---|---|
| Pure CDIIS | stalls / oscillates within 60 steps | — |
| ADIIS→CDIIS | converges ~38 steps | \(-75.718966\) within \(10^{-6}\) |

---

## 18.4 Second-Order SCF

### Theory

Minimise energy on the orbital manifold:
\(C(\kappa)=C\,e^{\kappa}\), \(\kappa^\top=-\kappa\), with independent
parameters only in the occupied–virtual block. Newton steps use the orbital
Hessian \({}^1(A+B)\) (same operator as stability analysis). **CIAH** solves
an augmented eigenproblem with matrix-free \(\sigma\)-products (Sun). Trust
radius and energy backtracking stabilise the macro-iterations.

Cost per Newton step exceeds one CDIIS extrapolation. Staging: DIIS until
\(\max|e|<\texttt{soscf\_start}\), then SOSCF. Implemented for RHF/UHF only
(not KS).

### Practice

```cpp
cpu::scf::UHF uhf(5, 5);
uhf.soscf = true;
uhf.soscf_start = 0.25;
uhf.soscf_max_micro = 12;
uhf.soscf_trust = 0.4;
uhf.run(frame);
// uhf.soscf_iterations() — Newton macro-steps (diagnostic)
```

### Results

| System | Quantity | Engine | PySCF `mf.newton()` |
|---|---|---:|---:|
| 2.6× H₂O def2-SVP UHF + SOSCF | \(E\) / Ha | −75.71896632262 | −75.71896632262 |
| 2.6× H₂O def2-SVP UHF + SOSCF | Newton macros | ~9 | ~8 (comparable seed) |
| N₂ \(R=2.5\) Å 6-31G RHF + SOSCF | \(E\) / Ha | −108.211223079 | −108.211223079 |
| N₂ \(R=2.5\) Å 6-31G RHF + SOSCF | Newton macros | 4 | 4 |

---

## Further reading

- Jensen, *Introduction to Computational Chemistry*, SCF techniques.
- Pulay, *Chem. Phys. Lett.* **73**, 393 (1980); Kudin et al., *J. Chem. Phys.*
  **116**, 8255 (2002); Hu & Yang, *J. Chem. Phys.* **132**, 054109 (2010).
- Sun, arXiv:1610.08423 (CIAH).
- Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory*, Ch. 10.
