# scf/mp2

One case; every section below is run by `atv_regressions --case scf/mp2` and keeps its own oracle record.

The `scs-mp2-h2o-sto3g`, `frozen-core-mp2-h2o` and `mp2-water-chain` sections accept
`--device=cpu|cuda|both`. Their CUDA implementation uses the same geometries,
bases, frozen-core choices and external PySCF energy pins below, with a
1e-6 Hartree tolerance on the correlated total, correlation and available
spin-channel pins. CUDA additionally checks Frame total-energy output in eV,
the read-only Frame entry, nested accumulation and explicit rejection of
unsupported force requests. These are energy gates; they do not establish
CUDA MP2 gradients, device residency or performance.

## Eight-water canonical RMP2 / STO-3G

The `mp2-water-chain` section runs on CPU and CUDA. Molecule `w=0..7`
has O(4w, 0, 0.1173), H(4w, 0.7572, -0.4692),
H(4w, -0.7572, -0.4692), in Angstrom. This extends the single-water pin
to 56 AOs and tests the translated, separated system used by the scaling
benchmark. All five energies use a 1e-6 Hartree absolute gate.

Offline oracle: PySCF 2.11.0, `cart=True`, `basis='sto-3g'`, RHF
`init_guess='1e'`, `conv_tol=1e-12`, `max_cycle=200`, `diis_space=8`,
`diis_start_cycle=1`, `damp=0`; then `mp.MP2(mf).kernel()`.
An independent MINAO-started solve agrees within 2e-10 Hartree on the
correlated total. Oracle runs are manual; the executable has no PySCF
dependency. The local oracle build has no OpenMP, so its wall time is not
a performance comparison.

| Quantity | Hartree |
| --- | ---: |
| HF | -599.6949297548834 |
| Correlation | -0.2851551624659191 |
| Opposite spin | -0.2688466882176113 |
| Same spin | -0.016308474248307774 |
| Total | -599.9800849173494 |


## Reference — H2O RMP2/STO-3G analytic nuclear gradient

Oracle protocol: offline PySCF pin (`tools/pyscf_oracle/pin_scf.py`); numbers
hard-coded below. **Never** import PySCF from CMake/ctest.

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| Basis | STO-3G, **Cartesian** (`cart=True`) |
| Method | RHF → canonical RMP2 (no frozen core) |
| Convergence | `conv_tol = 1e-12` |
| PySCF version | 2.14.0 |
| Command | `python tools/pyscf_oracle/pin_scf.py mp2 --geom water --basis sto-3g --grad --conv-tol 1e-12` |
| **E_HF** | **−74.963023138463 Ha** |
| **E_corr** | **−0.035545651647 Ha** |
| **E_MP2** | **−74.998568790110 Ha** |
| Gate | element-wise 1e-5 Ha/Bohr; translational invariance 1e-8 Ha/Bohr |

### Pinned analytic gradient dE/dR [Ha/Bohr]

`mp.MP2(mf).run(); m2.nuc_grad_method().kernel()`. Molecule in the yz-plane →
x components numerically zero (~1e-16).

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O  | 0 | 0 | −9.528798092138e-02 |
| H  | 0 | −3.796971346463e-02 | +4.764399046069e-02 |
| H  | 0 | +3.796971346463e-02 | +4.764399046069e-02 |

Σ over atoms ≈ (0, 0, 0) to ~1e-15 Ha/Bohr.

### Why 1e-5 Ha/Bohr

PySCF and Atomiverse both deliver **analytic** RMP2 gradients (Lagrangian
1-RDM + non-separable 2-RDM + Z-vector on ¹(A+B)). Agreement to ~1e-6–1e-8 is
expected on this minimal system; the regression gate is a looser 1e-5 Ha/Bohr
to cover packing / AO-slice assembly noise without masking a missing response
or 2-RDM term (those errors are typically ≥1e-3 Ha/Bohr).

Literature: Pople, Krishnan, Schlegel, Binkley, *Int. J. Quantum Chem. Symp.*
**13**, 225 (1979); Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
Theory*, response / gradient chapters; PySCF `pyscf.grad.mp2`.


## H2O frozen-core RMP2/6-31G* — e_corr + e_tot pins

Closed-shell Møller–Plesset second-order (RMP2) with **frozen O 1s** on
water / 6-31G*. Axis extension after plain RMP2
(`regressions/scf/mp2`) and `ump2-o2-sto3g`: same observable pair
(`e_corr`, `e_tot`) with the frozen-core axis (lowest occupied MO excluded
from the correlation sum).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py mp2-fc`.

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) — pin_scf `water` preset |
| Basis | 6-31G* (`6-31g*`) |
| Angular convention | Cartesian (`cart=True`; Pople `*` → Cartesian \(d\), Atomiverse default) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.MP2(mf, frozen=1).run()` — freeze lowest occupied MO (O 1s) |
| AO count | 19 (Cartesian 6-31G* on H₂O) |
| Occupied MOs | 5; correlating occupied = 4 after freeze |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py mp2-fc \
    --geom water --basis 6-31g* --frozen 1 --conv-tol 1e-12
```

Equivalent explicit script:

```python
import pyscf
from pyscf import gto, scf, mp

mol = gto.M(
    atom="""
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""",
    basis="6-31g*",
    cart=True,
    unit="Angstrom",
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()                    # -76.010504988257 Ha
m2 = mp.MP2(mf, frozen=1).run()       # freeze O 1s
## m2.e_corr  -> -0.186119540118 Ha
## m2.e_tot   -> -76.196624528375 Ha
## Full (unfrozen) RMP2 e_corr ≈ -0.188521600925 Ha for comparison
```

### Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −76.010504988257 | 1e-8 Ha (reference only; not the product gate) |
| \(E_{\mathrm{corr}}\) (frozen-core RMP2) | −0.186119540118 | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) = \(E_{\mathrm{RHF}}+E_{\mathrm{corr}}\) | −76.196624528375 | **1e-6 Ha** |

Cross-check: unfrozen RMP2 on the same geometry/basis/cart/tol gives
\(E_{\mathrm{corr}}\approx-0.188521600925\,\mathrm{Ha}\)
(\(E_{\mathrm{tot}}\approx-76.199026589182\,\mathrm{Ha}\)); freezing O 1s
raises \(E_{\mathrm{corr}}\) by \(\sim 2.4\,\mathrm{mHa}\), as expected for
excluding core correlation on a single first-row heavy atom.

### Notes

- Product regression `reg_frozen-core-mp2-h2o.cpp` asserts **only** the
  hard-coded pins above — no live PySCF, no reading this file at ctest time.
  Public path: `rhf.frozen = 1; auto r = rhf.mp2(frame);` (member, not a
  Config bag).
- PySCF's `frozen=1` freezes the lowest occupied spatial MO (O 1s on water);
  this matches the usual “frozen core” chemistry convention for H₂O (no
  core orbitals on H).
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`posthf-eri-block-mp2`, `ump2-o2-sto3g`); slug = queue id
  `frozen-core-mp2-h2o`.


## H2O SCS-MP2/STO-3G — e_corr pin

Spin-component-scaled second-order Møller–Plesset (SCS-MP2) on water /
STO-3G. Axis extension after plain RMP2 (`regressions/scf/mp2`)
and `frozen-core-mp2-h2o`: same RMP2 doubles sum, reweighted by Grimme’s
opposite-spin / same-spin scales.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py scs-mp2`.

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) — pin_scf `water` preset |
| Basis | STO-3G (`sto-3g`) |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.MP2(mf).run()` then Grimme SCS: \(E_{\mathrm{corr}}^{\mathrm{SCS}} = \tfrac13 E_{\mathrm{SS}} + 1.2\,E_{\mathrm{OS}}\) |
| AO count | 7 |
| Occupied spatial MOs | 5 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py scs-mp2 \
    --geom water --basis sto-3g --conv-tol 1e-12
```

Equivalent explicit script:

```python
import pyscf
from pyscf import gto, scf, mp

mol = gto.M(
    atom="""
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""",
    basis="sto-3g",
    cart=True,
    unit="Angstrom",
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()                 # -74.963023138463 Ha
m2 = mp.MP2(mf).run()
## m2.e_corr_ss -> -0.002029831562 Ha
## m2.e_corr_os -> -0.033515820085 Ha
## m2.e_corr    -> -0.035545651647 Ha  (plain RMP2)
## m2.emp2_scs  -> -0.040895594623 Ha  (= e_ss/3 + 1.2*e_os)
## m2.e_tot_scs -> -75.003918733086 Ha
```

### Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138463 | 1e-7 Ha (reference only; not the product gate) |
| \(E_{\mathrm{OS}}\) (unscaled) | −0.033515820085 | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{SS}}\) (unscaled) | −0.002029831562 | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{corr}}\) (plain RMP2) | −0.035545651647 | 1e-6 Ha (diagnostic; OS+SS) |
| \(E_{\mathrm{corr}}^{\mathrm{SCS}}\) | **−0.040895594623** | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}^{\mathrm{SCS}}\) | −75.003918733086 | 1e-6 Ha |

Cross-check: \(E_{\mathrm{corr}}^{\mathrm{SCS}} = \tfrac13 E_{\mathrm{SS}} + 1.2\,E_{\mathrm{OS}}\)
recovers the pin to machine precision; SCS is more negative than plain RMP2
on this system by \(\sim 5.3\,\mathrm{mHa}\) (OS up-weighted, SS down-weighted).

### Notes

- Product regression `reg_scs-mp2-h2o-sto3g.cpp` asserts the hard-coded
  SCS \(E_{\mathrm{corr}}\) (and \(E_{\mathrm{tot}}\)) pins — no live PySCF, no
  reading this file at ctest time. Public path:
  ```cpp
  rhf.mp2_os_scale = 1.2;
  rhf.mp2_ss_scale = 1.0 / 3.0;
  auto r = rhf.mp2(frame);   // r.e_corr is SCS; r.e_corr_os/ss unscaled
  ```
  Members, not a Config bag (iron law).
- Grimme, S. *J. Chem. Phys.* **118**, 9095 (2003). Scales match PySCF’s
  `emp2_scs` property (`e_corr_ss/3 + e_corr_os*1.2`).
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`posthf-eri-block-mp2`, `frozen-core-mp2-h2o`, `ump2-o2-sto3g`); slug = queue id
  `scs-mp2-h2o-sto3g`.


## O2 triplet UMP2/STO-3G — e_corr + e_tot pins

Open-shell Møller–Plesset second-order (UMP2) on a UHF reference for
triplet O₂ / STO-3G. Axis extension after plain RMP2
(`regressions/scf/mp2`): same observable pair (`e_corr`, `e_tot`)
on an unrestricted open-shell reference.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py ump2`.

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | O₂, charge 0, spin \(2S=2\) (triplet, \(S=1\)) |
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) — same as `o2_rohf_sto3g` |
| Basis | STO-3G |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart flag is inert but recorded) |
| SCF | UHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.UMP2(mf).run()` (no frozen core) |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ump2 \
    --geom o2 --basis sto-3g --spin 2 --conv-tol 1e-12
```

Equivalent explicit script:

```python
import pyscf
from pyscf import gto, scf, mp

mol = gto.M(
    atom="O 0 0 0; O 0 0 1.2075",
    basis="sto-3g",
    spin=2,
    cart=True,
    unit="Angstrom",
)
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()           # -147.633946785450 Ha
m2 = mp.UMP2(mf).run()
## m2.e_corr  -> -0.096393144130 Ha
## m2.e_tot   -> -147.730339929580 Ha
## mf.spin_square() -> (2.003410857681, ~3.002)  # residual UHF contamination
```

### Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{UHF}}\) | −147.633946785450 | 1e-8 Ha (reference only; not the product gate) |
| \(E_{\mathrm{corr}}\) (UMP2) | −0.096393144130 | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) = \(E_{\mathrm{UHF}}+E_{\mathrm{corr}}\) | −147.730339929580 | **1e-6 Ha** |

Cross-check vs existing ROHF/UHF pins in `o2_rohf_sto3g/REFERENCE.md`:
UHF total energy matches that table (−147.633946785450 Ha) to the shared
digits; ⟨S²⟩ − 2 ≈ 3.410858×10⁻³ (same contamination).

### Notes

- Product regression `reg_ump2-o2-sto3g.cpp` asserts **only** the hard-coded pins
  above — no live PySCF, no reading this file at ctest time. ROHF density
  warm-start is required so UHF lands on the same minimum as PySCF.
- UMP2 is spin-contaminated when the UHF reference is; the pins are the
  raw (unprojected) UMP2 energies as returned by PySCF `mp.UMP2`.
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`posthf-eri-block-mp2`, `frozen-core-mp2-h2o`); slug = queue id `ump2-o2-sto3g`.


## Reference — O₂ UMP2/STO-3G analytic nuclear gradient

Oracle protocol: offline PySCF pin (`tools/pyscf_oracle/pin_scf.py`); numbers
hard-coded below. **Never** import PySCF from CMake/ctest.

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) — same as `ump2-o2-sto3g` / `o2_rohf_sto3g` |
| Basis | STO-3G, **Cartesian** (`cart=True`) |
| Method | UHF (spin \(2S=2\) triplet) → canonical UMP2 (no frozen core) |
| Convergence | `conv_tol = 1e-12` |
| PySCF version | 2.14.0 |
| Command | `python tools/pyscf_oracle/pin_scf.py ump2 --geom o2 --basis sto-3g --spin 2 --conv-tol 1e-12 --grad` |
| **E_UHF** | **−147.633946785450 Ha** |
| **E_corr** | **−0.096393144130 Ha** |
| **E_UMP2** | **−147.730339929580 Ha** |
| Gate | element-wise 1e-5 Ha/Bohr; translational invariance 1e-8 Ha/Bohr |

### Pinned analytic gradient dE/dR [Ha/Bohr]

`mp.UMP2(mf).run(); m2.nuc_grad_method().kernel()`. Collinear diatomic on z →
x,y components numerically zero (~1e-16).

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O  | 0 | 0 | +1.665132663664e-01 |
| O  | 0 | 0 | −1.665132663664e-01 |

Σ over atoms ≈ (0, 0, 0) to ~1e-15 Ha/Bohr.

### Why 1e-5 Ha/Bohr

PySCF and Atomiverse both deliver **analytic** UMP2 gradients (spin-resolved
Lagrangian 1-RDM + non-separable 2-RDM + unrestricted Z-vector on (A+B)).
Agreement to ~1e-6–1e-8 is expected on this minimal system; the regression gate
is a looser 1e-5 Ha/Bohr to cover packing / AO-slice assembly noise without
masking a missing response or 2-RDM term (those errors are typically ≥1e-3
Ha/Bohr).

Literature: Pople, Krishnan, Schlegel, Binkley, *Int. J. Quantum Chem. Symp.*
**13**, 225 (1979); Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
Theory*, response / gradient chapters; PySCF `pyscf.grad.ump2`.

### Multi-minima note

Cold UHF from the core Hamiltonian often lands on a higher local solution for O₂. MINAO, PySCF's default, reaches the lower one. The
regression warm-starts from ROHF densities (same path as `ump2-o2-sto3g`) so
UHF lands on the PySCF minimum used for the pin.

### Fair wall twin (perf gate)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/pyscf_oracle/fair_bench.py --method ump2_grad --geom o2 \
    --basis sto-3g --spin 2 --repeat 3 -- ./build/regressions/reg_ump2-grad-o2-sto3g
```

| Side | median wall (s) |
|---|---|
| PySCF 2.14.0 UHF+UMP2+grad | 0.296 |
| Atomiverse `reg_ump2-grad-o2-sto3g` | 0.404 |
| **ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\)** | **1.37** |
| within_10x | **true** |


## O2 triplet ROMP2/STO-3G — e_corr + e_tot pins

Restricted open-shell Møller–Plesset second-order (ROMP2) on an ROHF
reference for triplet O₂ / STO-3G. Axis extension after `ump2-o2-sto3g`:
same geometry/basis/spin, but the **spin-pure** ROHF determinant instead of
a spin-contaminated UHF one. Product gate: `e_corr` (and consistency of
`e_tot`).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py romp2`.

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | O₂, charge 0, spin \(2S=2\) (triplet, \(S=1\)) |
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) — same as `o2_rohf_sto3g` |
| Basis | STO-3G |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart flag is inert but recorded) |
| SCF | ROHF (9α, 7β), `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.UMP2(mf_rohf).run()` (PySCF converts ROHF→UHF view then UMP2) |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Method note (PySCF convention)

PySCF has no separate `mp.ROMP2` class in 2.14.0. Calling `mp.MP2` /
`mp.UMP2` on an ROHF object:

1. Converts ROHF → UHF (`addons.convert_to_uhf`): shared spatial \(C\),
   occupations \(n_\alpha / n_\beta\), and **semi-canonical** orbital
   energies \(\varepsilon_i^\sigma = (C^\dagger F_\sigma C)_{ii}\) tagged as
   `mo_energy.mo_ea` / `mo_eb` (not the Roothaan \(F_{\mathrm{eff}}\)
   eigenvalues).
2. Runs raw (unprojected) UMP2 on that UHF-shaped view.

Atomiverse `ROHF::mp2` implements the same: true spin Focks
(`F_alpha_uncoupled_` / `F_beta_uncoupled_`) → diagonal MO energies →
`ump2_energy` kernel. Spin purity of the ROHF reference is preserved
(\(\langle S^2\rangle = S(S+1)\)); only the denominators are spin-resolved.

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py romp2 \
    --geom o2 --basis sto-3g --spin 2 --conv-tol 1e-12
```

Equivalent explicit script:

```python
import pyscf
from pyscf import gto, scf, mp

mol = gto.M(
    atom="O 0 0 0; O 0 0 1.2075",
    basis="sto-3g",
    spin=2,
    cart=True,
    unit="Angstrom",
)
mf = scf.ROHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()           # -147.632166990682 Ha
m2 = mp.UMP2(mf).run()
## m2.e_corr  -> -0.096718716392 Ha
## m2.e_tot   -> -147.728885707074 Ha
## mf.spin_square() -> (2.0, 3.0)
```

### Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{ROHF}}\) | −147.632166990682 | 1e-8 Ha (reference only; not the product gate) |
| \(E_{\mathrm{corr}}\) (ROMP2) | −0.096718716392 | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) = \(E_{\mathrm{ROHF}}+E_{\mathrm{corr}}\) | −147.728885707074 | **1e-6 Ha** |

Cross-check vs `o2_rohf_sto3g/REFERENCE.md`: ROHF total matches
(−147.632166990682 Ha). Cross-check vs `ump2-o2-sto3g`: UHF is lower
(−147.633946785450) and UMP2 \(E_{\mathrm{corr}}\) differs
(−0.096393144130) because the reference orbitals/energies differ.

### Performance

Fair wall-time vs PySCF (same geometry/basis/method class; median of ≥3
runs). Command:

```bash
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method romp2 --geom o2 --basis sto-3g --spin 2 --repeat 3 -- \
    ./build/regressions/reg_romp2-o2-sto3g
```

| Side | Median wall (s) | Notes |
|---|---|---|
| PySCF 2.14.0 | 0.118 | `ROHF` + `UMP2` (OMP_NUM_THREADS=1) |
| Atomiverse | 0.162 | `reg_romp2-o2-sto3g` (OMP_NUM_THREADS=1) |
| **ratio** \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **1.37** | target ≤ 10.0 ✓ |

Recorded 2026-07-23 with `fair_bench.py --method romp2 --repeat 3`.
`within_10x=true`. No focused optimize pass required.

Gate: `within_10x` ⇔ ratio ≤ 10.0. Correct-but-slow is **not** shippable.

### Notes

- Product regression `reg_romp2-o2-sto3g.cpp` asserts **only** the hard-coded
  pins above — no live PySCF, no reading this file at ctest time.
- Domain path under `regressions/scf/`; slug = queue id `romp2-o2-sto3g`.


## H2O RMP2/STO-3G through the block-assembled dense ERI — e_corr / e_hf pins

Plain (canonical, non-RI) second-order Møller–Plesset on water / STO-3G, driven
through the public `RHF::mp2` path. Canonical RMP2 consumes the full N⁴ AO ERI
tensor materialized by `IntegralWorkspace::eri_dense`; this case is the end-to-end
witness for spec **scf-review-07-posthf-perf item 17** — the dense ERI tensor
is re-assembled from a per-AO scalar `native::eri` loop to a `native::eri_block`
shell-block engine (amortized Boys/R-table + OpenMP, 8-fold-symmetry scatter),
and the correlation energy must reproduce the cited PySCF value unchanged.

### What it gates

This witness uses the PySCF `water` preset geometry and pins the **exact**
PySCF RHF total energy **and** the plain-RMP2 correlation energy, so it gates
the block-assembled dense-ERI numbers rather than merely their sign. It is the
plain-RMP2 pin of the suite.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py`.

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) — pin_scf `water` preset |
| Basis | STO-3G (`sto-3g`), 7 AO, Cartesian (no \(d\); `cart=True` is a no-op) |
| Occupied spatial MOs | 5 |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.MP2(mf).run()` (plain RMP2, unit scales) |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23 (RHF + RMP2 channels), venv `~/work/.x86_64` |

Geometry, basis, SCF settings and PySCF version are identical to
`regressions/scf/mp2` (same `water` preset); the plain-RMP2
`e_corr` here is that case's `m2.e_corr` (before the Grimme SCS reweighting).

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py scs-mp2 \
    --geom water --basis sto-3g --conv-tol 1e-12
```

Equivalent explicit script:

```python
import pyscf
from pyscf import gto, scf, mp

mol = gto.M(
    atom="""
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""",
    basis="sto-3g",
    cart=True,
    unit="Angstrom",
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()                 # -74.963023138463 Ha
m2 = mp.MP2(mf).run()
## m2.e_corr -> -0.035545651647 Ha  (plain RMP2 = e_corr_os + e_corr_ss)
## m2.e_tot  -> -74.998568790110 Ha
```

### Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138463 | 1e-7 Ha |
| \(E_{\mathrm{corr}}\) (plain RMP2) | **−0.035545651647** | **1e-7 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) | −74.998568790110 | 1e-6 Ha |

Cross-check: \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) recovers
the pin to machine precision.

### Atomiverse reproduction (PASS today, pre-rewrite scalar `eri_dense`)

Captured 2026-07-24 (GCC 14.3, `-DATV_BUILD_HF=ON`, double precision):

| Quantity | Atomiverse (Ha) | \(\Delta\) vs PySCF pin |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138463072 | ~7e-14 |
| \(E_{\mathrm{corr}}\) | −0.035545651648477 | ~1.5e-12 |
| \(E_{\mathrm{tot}}\) | −74.998568790111548 | ~1.5e-12 |

Both the current scalar `eri_dense` and the block-assembled rewrite must land
inside the pins above (the ERI equivalence judge caps block-vs-scalar drift at
≤1e-12 per element — see `tests/cpu/scf/test_two_electron.cpp`).

### Notes

- Product regression `reg_posthf-eri-block-mp2.cpp` asserts the hard-coded RHF /
  RMP2 pins — no live PySCF, no reading this file at ctest time. Public path:
  ```cpp
  cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
  rhf.tol = 1e-12;
  auto m = rhf.mp2(frame);   // m.e_corr plain RMP2 through the dense ERI tensor
  ```
  Members, not a Config bag (iron law).
- Møller, C.; Plesset, M. S. *Phys. Rev.* **46**, 618 (1934). Values match
  PySCF 2.14.0 `pyscf.mp.MP2`.
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`scs-mp2-h2o-sto3g`, `frozen-core-mp2-h2o`); slug = queue id
  `posthf-eri-block-mp2`.


## H₂ MP3/STO-3G — e_corr pin

Closed-shell third-order Møller–Plesset (RMP3) on H₂ / STO-3G. Product gate
is the total correlation energy \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\)
(observable `e_corr`, tolerance **1e-6 Ha**).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py mp3`.

PySCF 2.14 does **not** ship `mp.MP3`. The pin rebuilds the Szabo–Ostlund /
Psi4NumPy closed-shell MP3 diagram sum on **PySCF RHF orbitals + MO ERIs**
(`ao2mo`), which is the same formula Atomiverse implements
(`RHF::mp3` → spin-orbital residual form of the same diagrams). Cross-check:
\(E^{(2)}\) matches `mp.MP2` to machine precision; \(E_{\mathrm{corr}}\) lies
between MP2 and FCI/CISD on this 2-electron system.

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — pin_scf `h2` preset |
| Basis | STO-3G (`sto-3g`) |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | Canonical RMP3 (no frozen core) |
| AO / MO count | 2 / 2 |
| Occupied spatial MOs | 1 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py mp3 \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

Equivalent explicit script (oracle formula):

```python
import numpy as np
from pyscf import gto, scf, ao2mo, mp

mol = gto.M(
    atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""",
    basis="sto-3g",
    cart=True,
    unit="Angstrom",
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()                 # -1.116759307396 Ha
print("MP2 check", mp.MP2(mf).run().e_corr)

C, eps = mf.mo_coeff, mf.mo_energy
nmo, ndocc = C.shape[1], mol.nelectron // 2
eri_chem = ao2mo.restore(1, ao2mo.full(mol, C), nmo)
## Physicist <pq|rs> = (pr|qs)
MO = np.einsum("prqs->pqrs", eri_chem)
o, v = slice(0, ndocc), slice(ndocc, nmo)
eocc, evirt = eps[:ndocc], eps[ndocc:]
Dinv = 1.0 / (
    eocc.reshape(-1, 1, 1, 1)
    + eocc.reshape(-1, 1, 1)
    - evirt.reshape(-1, 1)
    - evirt
)
## Szabo:1996 / Psi4NumPy MP3.py — 12 spatial diagrams (E3 only) + MP2
## ... (see tools/pyscf_oracle/pin_scf.py method "mp3")
```

### Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396 | 1e-7 Ha (reference only) |
| \(E^{(2)}\) (RMP2) | −0.013138073590 | 1e-6 Ha (diagnostic; matches `mp.MP2`) |
| \(E^{(3)}\) | −0.004836072637 | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\) | **−0.017974146227** | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}^{\mathrm{MP3}}\) | −1.134733453623 | 1e-6 Ha |

Cross-check: FCI/CISD correlation on the same geometry is −0.020524527092 Ha
(PySCF `fci.FCI` / `ci.CISD`); MP3 recovers ~87.6% of the correlation and
sits between MP2 and FCI as expected.

### Performance

Fair wall-time vs the offline oracle (same geom/basis/method class; median of
≥3 runs). Protocol: `tools/pyscf_oracle/fair_bench.py --method mp3`.

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method mp3 --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_mp3-h2-sto3g
```

Recorded 2026-07-23 (login node, `OMP_NUM_THREADS=1`, median of 5):

| Side | Median wall (s) | Notes |
|---|---|---|
| PySCF oracle (RHF + MP3 diagrams) | 0.01602 | `fair_bench.py --method mp3` |
| Atomiverse `reg_mp3-h2-sto3g` | 0.01241 | includes RHF + MP3 + asserts |
| **ratio** \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.774** | gate ≤ 10.0 — **within_10x=true** |

### Notes

- Product regression `reg_mp3-h2-sto3g.cpp` asserts the hard-coded
  \(E_{\mathrm{corr}}\) (and \(E_{\mathrm{tot}}\), \(E^{(2)}\), \(E^{(3)}\))
  pins — no live PySCF, no reading this file at ctest time. Public path:
  ```cpp
  cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
  rhf.tol = 1e-12;
  auto r = rhf.mp3(frame);   // r.e_corr = E2+E3; r.e_corr_mp2 / e_corr_mp3
  ```
  Members / ctor args only — no Config bag (iron law).
- Literature: Szabo & Ostlund, *Modern Quantum Chemistry*, Table 6.2 /
  pp. 364–368 (third-order diagrams). Psi4NumPy
  `Moller-Plesset/MP3.py` is the spatial reduction used by the pin.
- Domain path under `regressions/scf/`; slug = queue id `mp3-h2-sto3g`.


## H₂ MP4(SDTQ)/STO-3G — e_corr pin

Closed-shell fourth-order Møller–Plesset MP4(SDTQ) on H₂ / STO-3G. Product gate
is the total correlation energy
\(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\)
(observable `e_corr`, tolerance **1e-6 Ha**).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py mp4`.

PySCF 2.14 does **not** ship `mp.MP4`. The pin rebuilds intermediate-normalization
Rayleigh–Schrödinger PT through order 4 on the full spin-orbital determinant
basis from **PySCF RHF orbitals + MO ERIs** (`ao2mo`). Cross-checks:
\(E^{(2)}\) matches `mp.MP2` to machine precision; \(E^{(2)}+E^{(3)}\) matches
the offline MP3 pin; \(E_{\mathrm{corr}}\) lies between MP3 and FCI/CISD on
this 2-electron system. Atomiverse implements the equivalent spin-orbital
SDTQ residual form (`RHF::mp4`).

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — pin_scf `h2` preset |
| Basis | STO-3G (`sto-3g`) |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | Canonical RMP4(SDTQ) (no frozen core) |
| AO / MO count | 2 / 2 |
| Occupied spatial MOs | 1 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py mp4 \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

### Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396 | 1e-7 |
| \(E^{(2)}\) | −0.013138073590 | 1e-6 |
| \(E^{(3)}\) | −0.004836072637 | 1e-6 |
| \(E^{(4)}\) | −0.001711078792 | 1e-6 |
| \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\) | **−0.019685225019** | **1e-6** (product gate) |
| \(E_{\mathrm{tot}}=E_{\mathrm{RHF}}+E_{\mathrm{corr}}\) | −1.136444532416 | 1e-6 |

Notes for this 2e system: triples and true quadruple excitations vanish
(\(n_{\mathrm{occ}}^{\mathrm{SO}}=2\)); \(E^{(4)}\) is carried by the doubles
residual channel plus renormalization \(E_Q^{(4)}=-E^{(2)}\|\Psi^{(1)}\|^2\).
FCI / CCSD correlation on the same geometry is −0.020524527092 Ha.

### Fair wall twin

```bash
python tools/pyscf_oracle/fair_bench.py --method mp4 --geom h2 --basis sto-3g \
    --repeat 3 ./build/regressions/reg_mp4-h2-sto3g
```

| Field | Value |
|---|---|
| \(T_{\mathrm{pyscf}}\) median (s) | 0.747 (det-space RSPT-4 twin) |
| \(T_{\mathrm{atv}}\) median (s) | 0.023 |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.031** |
| within_10x | **true** |
| perf_exempt | false |
| Fair-bench date | 2026-07-23 |
