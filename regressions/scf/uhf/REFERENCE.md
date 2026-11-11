# scf/uhf

One case; every section below is run by `atv_regressions --case scf/uhf` and keeps its own oracle record.


## UHF / STO-3G — H₂⁺, H₂, H₂O⁺

Out-of-library consumer of `cpu::scf::UHF` and, when CUDA is on,
`cuda::scf::UHF`. Same geometry and the same PySCF numbers on both.

### Oracle

PySCF 2.14.0, offline:

```bash
python tools/pyscf_oracle/pin_scf.py uhf --geom h2 --basis sto-3g --charge 1 --spin 1 --grad --conv-tol 1e-12
python tools/pyscf_oracle/pin_scf.py uhf --geom h2 --basis sto-3g --charge 0 --spin 0 --grad --conv-tol 1e-12
python tools/pyscf_oracle/pin_scf.py uhf --geom water --basis sto-3g --charge 1 --spin 1 --grad --conv-tol 1e-12
```

`oracle: pyscf`. Cartesian STO-3G. Energies are `e_tot_ha`. Gradients are
`grad_ha_bohr` from `nuc_grad_method().kernel()` (∂E/∂R).

| System | Electrons (α, β) | Charge | E [Ha] |
| --- | --- | --- | --- |
| H₂⁺ at 0.74 Å | (1, 0) | +1 | −0.538205447565 |
| H₂ at 0.74 Å | (1, 1) | 0 | −1.116759307396 |
| H₂O⁺ (oracle water geometry) | (5, 4) | +1 | −74.655898713621 |

H₂⁺ gradient [Ha/Bohr], atoms along z: (0, 0, +0.191355066881) and
(0, 0, −0.191355066881).

H₂ gradient: (0, 0, −0.027679600707) and (0, 0, +0.027679600707).

H₂O⁺ gradient, atoms O, H(+y), H(−y):

| Atom | x | y | z |
| --- | --- | --- | --- |
| O | 0 | 0 | −0.095196667501 |
| H | 0 | −0.084875484950 | +0.047598333750 |
| H | 0 | +0.084875484950 | +0.047598333750 |

Gates: |ΔE| < 1e-8 Ha, element-wise |Δg| < 1e-6 Ha/Bohr, |Σ g| < 1e-8 Ha/Bohr.

The CUDA path is device s/p integrals and `launch_uhf_forces_sp`. It throws
if the basis is outside STO-3G l ≤ 1.


## CUDA UHF / STO-3G — O₂ and OH

Out-of-library consumer of `cpu::scf::UHF` and, when CUDA is on,
`cuda::scf::UHF`. Same geometry and the same PySCF numbers on both.
The device leg must report `last_scf_on_device` and `last_force_on_device`.

### Oracle

PySCF 2.14.0, offline (`oracle: pyscf`):

```bash
python tools/pyscf_oracle/pin_scf.py uhf --geom o2 --basis sto-3g --spin 2 --grad --conv-tol 1e-12
python tools/pyscf_oracle/pin_scf.py uhf --geom oh --basis sto-3g --spin 1 --grad --conv-tol 1e-12
```

Cartesian STO-3G. Energies are `e_tot_ha`. Gradients are `grad_ha_bohr`
from `nuc_grad_method().kernel()` (∂E/∂R). PySCF's default initial guess
is MINAO (`UHF.init_guess_by_minao`: spin-averaged, because `_break_dm_spin_symm`
does not run when `mol.spin != 0`). The core-Hamiltonian guess is a different
O₂ minimum (−147.378559141664 Ha) and is not the pin.

OH uses r(O–H) = 0.96966 Å along z (the experimental equilibrium distance).
O₂ uses the repository geometry r(O–O) = 1.2075 Å.

| System | Electrons (α, β) | Charge | Spin (2S) | E [Ha] |
| --- | --- | --- | --- | --- |
| O₂ triplet | (9, 7) | 0 | 2 | −147.633946785450 |
| OH doublet | (5, 4) | 0 | 1 | −74.362633276842 |

O₂ gradient [Ha/Bohr], atoms along z: (0, 0, +0.024814046022) and
(0, 0, −0.024814046022).

OH gradient [Ha/Bohr]: (0, 0, +0.056145481913) and (0, 0, −0.056145481913).

Gates: |ΔE| < 1e-8 Ha, element-wise |Δg| < 1e-6 Ha/Bohr, |Σ g| < 1e-8 Ha/Bohr.
Same thresholds as the STO-3G UHF gradient cases.

The CPU call sets `init_guess = MINAO`, no damping, and CDIIS from iteration 1
(PySCF `diis_start_cycle = 1`, `damp = 0`). The `UHF` constructor's historical
Hcore guess plus 0.7 damping is the other O₂ basin. The CUDA path packs the
same spin-averaged MINAO density once at setup; the SCF and the analytic
gradient stay on device (`launch_eri_sp`, `launch_uhf_forces_sp`).


## Reference — O₂ UHF/STO-3G analytic nuclear Hessian

### Method

Analytic UHF nuclear Hessian via unrestricted coupled-perturbed Hartree–Fock
(CPHF) linear response, HF/STO-3G on triplet O₂ (charge 0, \(2S=2\)). Public
API: `cpu::scf::UHF{BasisSet::STO3G, 9, 7}.hessian(frame, &guess)` with ROHF density warm-start
(same multi-minima path as `ump2-o2-sto3g` / `o2_rohf_sto3g`). Observable is
the dense Cartesian Hessian in eV/Å² (Frame/vib contract).

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
|---|---|---|---|
| O | 0.0 | 0.0 | 0.0 |
| O | 0.0 | 0.0 | 1.2075 |

20 Å cubic box (gas phase; the box does not enter the molecular SCF).
Same geometry as `o2_rohf_sto3g`, `ump2-o2-sto3g`, `o2_rohf_grad`.

### Pinned PySCF oracle

PySCF **2.14.0** (`~/work/.x86_64/bin/python`), run 2026-07-23, commands
verbatim:

```python
import numpy as np
from pyscf import gto, scf, hessian

mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=2,
            cart=True, unit="Angstrom", verbose=0)
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
e = mf.kernel()                       # -147.633946785450 Ha
h = hessian.UHF(mf).kernel()          # (natm,natm,3,3) Ha/Bohr^2
## Flatten to [3N,3N] and convert to eV/Å²:
##   HA_TO_EV = 27.211386245988
##   BOHR_IN_A = 0.529177210903
##   H_ev = H_ha * HA_TO_EV / BOHR_IN_A**2
```

Offline pin helper (optional extension):

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python - <<'PY'
## (script above)
PY
```

Pinned values at this geometry (eV/Å²). Off-axis elements are numerically
zero (\(\lesssim 10^{-14}\)); only the Cartesian-diagonal / stretch blocks
are product-gated.

| Element (row, col) | atoms/axes | Value [eV/Å²] |
|---|---|---|
| \(H_{xx}(0,0)\) | O1-x / O1-x | −1.0567201218 |
| \(H_{yy}(1,1)\) | O1-y / O1-y | −1.0567201218 |
| \(H_{zz}(2,2)\) | O1-z / O1-z | +135.7176127486 |
| \(H_{xx}(0,3)\) | O1-x / O2-x | +1.0567201218 |
| \(H_{yy}(1,4)\) | O1-y / O2-y | +1.0567201218 |
| \(H_{zz}(2,5)\) | O1-z / O2-z | −135.7176127486 |
| \(H_{xx}(3,3)\) | O2-x / O2-x | −1.0567201219 |
| \(H_{yy}(4,4)\) | O2-y / O2-y | −1.0567201220 |
| \(H_{zz}(5,5)\) | O2-z / O2-z | +135.7176127485 |

Reference energy (not the product gate): \(E_{\mathrm{UHF}} = -147.633946785450\) Ha.

One real stretch frequency from the same Hessian (thermo analysis, not
gated): \(\nu \approx 2147.91\) cm⁻¹.

### Tolerance and its rationale

**1e-5 eV/Å² per gated element.** Analytic UHF CPHF and PySCF `hessian.UHF`
evaluate the same closed-form second derivatives; residual differences are
SCF/CPHF convergence noise. The stretch blocks are \(\sim 10^2\) eV/Å², so
1e-5 is a relative \(\sim 10^{-7}\) band — tight enough to catch a sign or
spin-channel error in the response or skeleton 2e pieces. Off-axis elements
are asserted only to be \(\lvert H_{ij}\rvert < 10^{-3}\) (near zero by
symmetry on this collinear geometry).

### References

- J. A. Pople, R. Krishnan, H. B. Schlegel, J. S. Binkley,
  *Int. J. Quantum Chem. Symp.* **13**, 225 (1979),
  DOI 10.1002/qua.560160825 — CPHF and analytic HF second derivatives.
- Y. Yamaguchi, Y. Osamura, J. D. Goddard, H. F. Schaefer,
  *A New Dimension to Quantum Chemistry: Analytic Derivative Methods in
  Ab Initio Molecular Electronic Structure Theory* (Oxford, 1994) —
  unrestricted Hessian.
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), response-theory chapter.

### What is checked

`cpu::scf::UHF{BasisSet::STO3G, 9, 7}` with ROHF warm-start → `uhf.hessian(frame, &guess)`
returns a 6×6 Hessian in eV/Å². Product gate: the nine Cartesian-diagonal /
stretch elements above within **1e-5 eV/Å²** of the pinned PySCF values;
failure prints got / want / tol and exits non-zero. ≤ 60 s, single node.


## N₂ ΔSCF (IMOM) / STO-3G — vertical α HOMO→LUMO pin

Closed-shell **ΔSCF** electronic excitation for N₂ at the experimental bond
length, with **IMOM** (initial maximum-overlap) occupation locking on the
non-Aufbau UHF re-SCF. Product API: `UHF::delta_scf(frame, from, to,
MomStyle::IMOM)` on a `UHF{basis, n_alpha, n_beta}`.

Literature / theory:

- A. T. B. Gilbert, N. A. Besley, P. M. W. Gill, *Self-Consistent Field
  Calculations of Excited States Using the Maximum Overlap Method (MOM)*,
  J. Phys. Chem. A **112**, 13164 (2008).
  DOI [10.1021/jp801738f](https://doi.org/10.1021/jp801738f)
- G. M. J. Barca, A. T. B. Gilbert, P. M. W. Gill, *Simple Models for
  Difficult Electronic Excitations*, J. Chem. Theory Comput. **14**, 1501
  (2018). DOI [10.1021/acs.jctc.7b00994](https://doi.org/10.1021/acs.jctc.7b00994)
  (IMOM = frozen initial reference)
- Implementation twin: PySCF `scf.addons.mom_occ_(uhf, (C,C), (occ_a,occ_b))`
  on a RHF-seeded α HOMO→LUMO promotion (IMOM; reference orbitals fixed).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | N₂, charge 0, singlet ground |
| Geometry (Å) | N(0,0,0); N(0,0,1.0977) — experimental bond length |
| Basis | STO-3G |
| Angular convention | `cart=True` (no \(d\) shells; engine-neutral) |
| Ground SCF | RHF → UHF warm-start, `conv_tol=1e-12` |
| Excited SCF | UHF, α occupation promote HOMO→LUMO (indices 6→7), β Aufbau; IMOM via `mom_occ_` |
| AO / MO | 10 AO; 7 occupied spatial MOs (14 e) |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py delta-scf \
    --geom n2 --basis sto-3g --conv-tol 1e-12
## optional: --from-mo 6 --to-mo 7  (defaults = HOMO/LUMO)
```

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{ground}}\) (RHF/UHF) | −107.495893307834 Ha | 1e-6 Ha |
| \(E_{\mathrm{excited}}\) (UHF-IMOM) | −107.192823826964 Ha | 1e-6 Ha |
| ΔE = \(E_{\mathrm{ex}}-E_{\mathrm{gs}}\) | 0.303069480870 Ha = **8.246940703328 eV** | **1e-4 eV** (product gate) |
| ⟨S²⟩ (excited, diagnostic) | ≈ 1.017 | (spin-contaminated Ms=0 single) |

CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)
(`units::kHartreeInEv`) — both sides of the eV gate use the same factor.

This is a **textbook minimal** ΔSCF foothold (not a spectroscopic band
assignment): STO-3G N₂ TDA S1 is ~9.13 eV on the same geometry; the ΔSCF
IMOM root is a single-determinant estimate of the HOMO→LUMO promotion.

### Gates (product regression)

- ΔE: \(|\Delta E_{\mathrm{ours}}-\mathrm{pin}| < 10^{-4}\,\mathrm{eV}\).
- \(E_{\mathrm{ground}}\) and \(E_{\mathrm{excited}}\): \(10^{-6}\) Ha.
- \(E_{\mathrm{excited}} > E_{\mathrm{ground}}\).

### Performance

Fair wall-time protocol (`tools/pyscf_oracle/fair_bench.py`): same geom/basis,
median of ≥3 runs, ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\), target ≤ 10.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method delta_scf --geom n2 --basis sto-3g --repeat 3 \
    -- -- ./build/regressions/reg_delta-scf-n2
'
```

| Metric | Value |
|---|---|
| \(T_{\mathrm{pyscf}}\) median (s) | 0.0283 |
| \(T_{\mathrm{atv}}\) median (s) | 0.2217 |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **7.83** |
| within_10x | **true** |
| Notes | `OMP_NUM_THREADS=1`; ATV path = RHF warm-start + ground UHF + IMOM UHF (three SCF loops); PySCF twin = RHF + `mom_occ_` UHF |

### Public API (no Config bags)

```cpp
cpu::scf::UHF uhf(cpu::scf::BasisSet::STO3G, /*n_alpha=*/7, /*n_beta=*/7);
uhf.tol = 1e-12;
auto r = uhf.delta_scf(frame, /*from=*/6, /*to=*/7, cpu::scf::MomStyle::IMOM);
// r.delta_e_ev [eV]; r.e_ground / r.e_excited [Ha]
```

Members only: `MomStyle mom`, `set_mom_reference(det)`,
`SlaterDeterminant::with_occupation` — no `DeltaScfConfig`, no factories.


## Reference — N2 RHF/6-31G* 电离能:Koopmans 与 ΔSCF

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md(外部运行、钉定);
钉定脚本:`tools/pin_charges_oracle.py`(只在外部 venv 里跑,绝不进
CMake/ctest)。

| 字段 | 值 |
|---|---|
| 几何 (Å) | N(0,0,0); N(0,0,1.0977)(实验键长)|
| 基组 | 6-31G* (cartesian-6d) |
| PySCF 版本 | 2.14.0,`conv_tol = 1e-12` |
| E(N₂) RHF | −108.942662347983 Ha |
| **−ε_HOMO** | **0.611756029712 Ha = 16.64672961 eV**(`mf.mo_energy[nocc−1]`)|
| E(N₂⁺) UHF | −108.365971102534 Ha(`charge=1, spin=1`;⟨S²⟩ ≈ 0.766)|
| **ΔSCF IP** | **15.69256822 eV** = (E₊ − E₀)×27.211386245988 |
| 对照容差 | 各 ≤1e-3 eV |

物理背景(Szabo & Ostlund, *Modern Quantum Chemistry*, §3.3):HF 极限的
N₂ Koopmans IP ≈ 17.3 eV,实验 15.58 eV —— 冻结轨道高估;6-31G* 下
−ε_HOMO = 16.65 eV(基组未到极限,数值随基组走,故对照钉在**同基组**的
PySCF 值上,文献带只作宽松 sanity:15.5–17.5 eV)。ΔSCF 让阳离子轨道弛豫,
15.69 eV 低于 Koopmans 且更接近实验 —— 回归同时断言 ΔSCF < Koopmans。

阳离子只经构造参数配置:`cpu::scf::UHF uhf{Pople631Gs, 7, 6, /*charge=*/1}`。

`n2_ionization` 声明 **4 个 CPU 线程**；同一 case 按最高声明线程数运行。
完整 section 包含中性 RHF、轨道观察器和阳离子 UHF，并强制检查 **≤60 秒**。
基组、SCF 收敛控制、PySCF 参考值和各 1e-3 eV 容差保持不变。

引文:
- Koopmans, *Physica* **1**, 104 (1934), DOI 10.1016/S0031-8914(34)90011-2。
- Szabo & Ostlund, *Modern Quantum Chemistry* (Dover 1996), §3.3(N₂ 讨论:
  −ε_HOMO ≈ 17.3 eV vs 实验 15.6 eV)。
- 实验垂直 IP 15.58 eV:NIST WebBook (N₂)。
