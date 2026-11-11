# response/vibrations

One case; every section below is run by `atv_regressions --case response/vibrations` and keeps its own oracle record.


## Reference — H₂O HF/6-31G* harmonic frequencies (analytic CPHF Hessian)

### Method

Analytic RHF nuclear Hessian via coupled-perturbed Hartree-Fock (CPHF)
linear response, HF/6-31G* (`Pople631Gs`, Cartesian d functions — the
Pople convention, PySCF `cart=True`), fed through the existing vib
pipeline (`cpu::vib::MassWeighting` → `cpu::vib::HessianDiagonalizer`) to
harmonic frequencies. Two gates:

1. **First gate (in-house self-consistency):** the analytic frequencies
   against the finite-difference production line — the `cpu::vib::Hessian`
   driver (central differences of the analytic forces, step 10⁻³ Å) with
   the same `cpu::scf::RHF` bound at `vib::Phase::force` — mode-by-mode
   ≤ 1 cm⁻¹. Both producers are in-house; they meet at the same
   `vib::HessianResult` consumer.
2. **External oracle:** the pinned PySCF `hessian.RHF` frequencies below,
   mode-by-mode ≤ 1 cm⁻¹.

### Geometry

The repository's standard water case geometry (identical to
`regressions/scf/rhf`, `h2o_rhf_dipole`, `h2o_charges_pyscf`):
R(O–H) = 0.95 Å, ∠H–O–H = 104.5°; O at the origin, both H symmetric about
+z in the xz-plane (half-angle 52.25°):

| Atom | x [Å] | y [Å] | z [Å] |
|---|---|---|---|
| O | 0.00000000 | 0.0 | 0.00000000 |
| H | +0.75115510 | 0.0 | 0.58160642 |
| H | −0.75115510 | 0.0 | 0.58160642 |

20 Å cubic box (gas phase; the box does not enter the molecular SCF).
Note this is NOT the equilibrium HF/6-31G* geometry, so the frequencies
below are *this geometry's* harmonic frequencies, not literature
equilibrium values — the oracle is the pinned external run at exactly
this geometry.

### Pinned PySCF oracle

PySCF **2.14.0** (`~/work/.x86_64/bin/python`), run 2026-07-21, commands
verbatim:

```python
import numpy as np, math
from pyscf import gto, scf, hessian
from pyscf.hessian import thermo

r_oh = 0.95
half = 0.5 * 104.5 * math.pi / 180.0
hx = r_oh * math.sin(half); hz = r_oh * math.cos(half)
mol = gto.M(atom=f"O 0 0 0; H {hx:.16f} 0 {hz:.16f}; H {-hx:.16f} 0 {hz:.16f}",
            basis="6-31g*", cart=True, unit="Angstrom", verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
h = hessian.RHF(mf).kernel()
res = thermo.harmonic_analysis(mol, h)
print(mf.e_tot)                  # -76.0107068004 Ha
print(res['freq_wavenumber'])    # [1843.7084 4036.1737 4146.3782] cm^-1
```

Pinned values at this geometry:

| Quantity | Value |
|---|---|
| E_RHF/6-31G* (cart) | −76.0107068004 Ha |
| ν₁ (bend, a₁) | 1843.7084 cm⁻¹ |
| ν₂ (sym stretch, a₁) | 4036.1737 cm⁻¹ |
| ν₃ (asym stretch, b₂) | 4146.3782 cm⁻¹ |

### Tolerance and its rationale

**1 cm⁻¹ per mode, both gates.** The analytic Hessian and PySCF's
`hessian.RHF` evaluate the same closed-form CPHF expressions; residual
differences are SCF/CPHF convergence noise and the harmonic-analysis
projection, well below 1 cm⁻¹. The FD gate carries the additional
O(h²) truncation of the central difference at 10⁻³ Å, also ≪ 1 cm⁻¹ for
water. This is a deliberately tight parity band — a sign/factor error in
any CPHF piece (skeleton derivatives, B^x, the response coupling) moves
frequencies by tens to hundreds of cm⁻¹.

### References

- J. A. Pople, R. Krishnan, H. B. Schlegel, J. S. Binkley,
  *Int. J. Quantum Chem. Symp.* **13**, 225 (1979),
  DOI 10.1002/qua.560160825 — CPHF and analytic HF second derivatives.
- P. Pulay, *Mol. Phys.* **17**, 197 (1969) — forces and derivative
  methods.
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), response-theory chapter.
- P. C. Hariharan, J. A. Pople, *Theor. Chim. Acta* **28**, 213 (1973) —
  the 6-31G* basis.

### What is checked

`cpu::scf::Hessian<RHF>{rhf}.run(frame)` →
`vib::MassWeighting` + `vib::HessianDiagonalizer` gives 3 real modes
(`n_imaginary == 0` is NOT asserted — this geometry is slightly off the
HF/6-31G* minimum, but all three vibrational modes are real and positive);
each mode within 1 cm⁻¹ of (a) the `cpu::vib::Hessian` FD production line
with the same RHF, and (b) the pinned PySCF values. Failure prints
mode-by-mode got / want / tol and exits non-zero. ≤ 60 s, single node.


## Reference — H₂O RHF/STO-3G dipole-derivative tensor (atomic polar tensor)

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | STO-3G (Cartesian) |
| 收敛 | `conv_tol = 1e-13`, RHF |
| PySCF 版本 | 2.14.0 |
| oracle | **finite difference** of `mf.dip_moment(unit='au')`, central, h = 1e-3 Bohr |
| 对照容差 | 元素级 1e-4 au(受 FD oracle 截断误差限制,非引擎);平移不变性 1e-6 au |

### 为什么 oracle 是 FD,不是解析

PySCF 的解析偶极导数在 `pyscf.prop.infrared` 模块,该 venv **未安装**。因此
参考值用 PySCF SCF 偶极的有限差分(每个核坐标 ±h,中心差分)。FD 截断误差
O(h²)~4e-7 au 就是 1e-4 容差的下限来源——引擎侧(CPHF 密度响应解析 + 偶极积分
块半数值导数,h=1e-4)比这个 FD oracle 更准,实测最大偏差 4.0e-7 au,即完全
落在 oracle 的 FD 噪声内。

### Pinned ∂μ_a/∂R_{A,j} [au] — [atom A][axis j][dipole a]

```
A0(O) j0: [-0.61244432,  0,           0         ]
A0(O) j1: [ 0,          -0.05805634,  0         ]
A0(O) j2: [ 0,           0,           0.03881088]
A1(H) j0: [ 0.30622222,  0,           0         ]
A1(H) j1: [ 0,           0.02902815,  0.15747119]
A1(H) j2: [ 0,           0.21470465, -0.01940554]
A2(H) j0: [ 0.30622222,  0,           0         ]
A2(H) j1: [ 0,           0.02902815, -0.15747119]
A2(H) j2: [ 0,          -0.21470465, -0.01940554]
```

Σ over atoms per (j,a) ≈ 2e-7 au (translational invariance; the FD floor).

### 物理与实现

∂μ_a/∂R_{A,j} = −Tr[∂P/∂R_{A,j}·r_a] − Tr[P·∂r_a/∂R_{A,j}] + Z_A δ_{aj}。
∂P/∂R 是 CPHF 密度响应(`CphfPack::density_response`,已在 test_cphf.cpp 对
FD 验证);∂⟨μ|r_a|ν⟩/∂R 用偶极积分块中心差分(半数值,同 ECP 梯度模式)。
IR 强度 I_k ∝ Σ_a (Σ_x ∂μ_a/∂R_x · L_{x,k})²(投影到简正模 `normal_modes`);
本回归门控 APT 张量本身——IR 强度的载重物理。逐模强度的 km/mol 投影需
`pyscf.prop` oracle 方能验证,留作后续。

文献:Wilson, Decius, Cross, *Molecular Vibrations* (1955)(APT / IR 强度);
Helgaker, Jørgensen, Olsen §10(CPHF / 分子性质导数);PySCF
`pyscf/prop/infrared`(解析实现参考)。


## Reference — harmonic oscillator driver matrix

### Potential

Isotropic classical harmonic oscillator (engineering units: eV, Å, amu, fs):

```text
U(r) = ½ k |r − r₀|²
F(r) = −k (r − r₀)
```

Cartesian Hessian (single free atom): `H = k I_3` [eV/Å²].

Angular frequency from Newton’s law in Atomiverse metal coupling:

```text
a [Å/fs²] = force_to_accel · F/m
ω² = force_to_accel · k / m
T  = 2π / ω  [fs]
```

with `force_to_accel = 1/103.6427` (`core/units.h`, LAMMPS metal).

### Why this case

One user-defined `Energy|Force` evaluator type is registered into four public
presets (`Evaluate`, `MD`, `Minimize`, `Hessian`) without any private header.
A user-defined `Criterion` stops the MD path by a step budget. No library
code knows these types.

### Tolerances

| Check | Tolerance | Justification |
|---|---|---|
| Evaluate U, F | `1e-12` absolute | analytic, double arithmetic |
| Minimize \|r−r₀\| | `1e-4` Å | ForceConverged ftol / k |
| Hessian eigenvalues | `1e-3` relative to k | central FD step vs exact k |
| NVE period / energy | period within 5%; \|ΔE\|/(k A²) ≤ 0.05 | VV O(dt²) at chosen dt |

### Parameters used by `reg_harmonic_driver_matrix`

`k = 1.0` eV/Å², `m = 12.0` amu, `r₀ = (0,0,0)`, initial displacement
`A = 0.1` Å along x, `dt = 0.5` fs.
