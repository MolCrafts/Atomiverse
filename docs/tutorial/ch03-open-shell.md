# Chapter 3: Unrestricted Hartree-Fock

Restricted Hartree-Fock is a good first model because it explains molecular
orbitals, density matrices, and self-consistency with the least amount of
machinery. It also fails in exactly the situations that matter most for open
shells and bond breaking. This chapter is about that failure and about the
smallest change that repairs it: allow the $\alpha$ and $\beta$ electrons
to occupy different spatial orbitals.

| | |
|---|---|
| **Example** | [`examples/03_uhf.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/03_uhf.cpp) |
| **Results** | [`examples/results/03_uhf_pes.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/03_uhf_pes.json) |
| **Reference limit** | \(2\times E(\mathrm{H})=-0.9332\) Ha |

```bash
cmake --build build --target 03_uhf
./build/examples/03_uhf
```

## Live result: H₂ dissociation (STO-3G)

RHF cannot dissociate to two neutral H atoms; UHF can. The scan below is the
actual table printed by `03_uhf` (columns: \(R\), \(E_\mathrm{RHF}\),
\(E_\mathrm{UHF}\), \(\langle S^2\rangle\)).

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true, tooltip: true, strokeWidth: 2}
data:
  values:
    - {R: 0.50, method: RHF, E: -1.042996}
    - {R: 0.75, method: RHF, E: -1.116151}
    - {R: 1.00, method: RHF, E: -1.066109}
    - {R: 1.25, method: RHF, E: -0.989114}
    - {R: 1.50, method: RHF, E: -0.910874}
    - {R: 1.75, method: RHF, E: -0.841349}
    - {R: 2.00, method: RHF, E: -0.783793}
    - {R: 2.50, method: RHF, E: -0.702944}
    - {R: 3.00, method: RHF, E: -0.656048}
    - {R: 3.50, method: RHF, E: -0.629820}
    - {R: 4.00, method: RHF, E: -0.614870}
    - {R: 0.50, method: UHF, E: -1.042996}
    - {R: 0.75, method: UHF, E: -1.116151}
    - {R: 1.00, method: UHF, E: -1.066109}
    - {R: 1.25, method: UHF, E: -0.994190}
    - {R: 1.50, method: UHF, E: -0.957707}
    - {R: 1.75, method: UHF, E: -0.943080}
    - {R: 2.00, method: UHF, E: -0.937213}
    - {R: 2.50, method: UHF, E: -0.933867}
    - {R: 3.00, method: UHF, E: -0.933285}
    - {R: 3.50, method: UHF, E: -0.933182}
    - {R: 4.00, method: UHF, E: -0.933166}
encoding:
  x: {field: R, type: quantitative, title: "R (Å)", scale: {zero: false}}
  y:
    field: E
    type: quantitative
    title: "E (Hartree)"
    scale: {zero: false, nice: true}
  color: {field: method, type: nominal, title: null}
  tooltip:
    - {field: method, type: nominal}
    - {field: R, type: quantitative, format: ".2f"}
    - {field: E, type: quantitative, format: ".6f"}
```

At \(R=4\) Å, UHF sits at **−0.933166 Ha** — within \(3\times 10^{-5}\) Ha of
the exact one-electron limit \(2\times E(\mathrm{H})=-0.9332\) Ha. RHF is still
**−0.61487 Ha**, more than 0.3 Ha too high. Spin contamination
\(\langle S^2\rangle\) rises from 0 near equilibrium toward 1 as the UHF
solution becomes two separate radicals.

## Why RHF Stops Being Enough

The central RHF assumption is that one spatial orbital is occupied by two
electrons of opposite spin. Near equilibrium, that is often a reasonable
picture. For $\mathrm{H}_2$ at $0.74$ Angstrom, it is exactly the right
place to begin.

The problem appears when the bond is stretched. Far apart, the physically
correct dissociation limit is two separate hydrogen atoms, each carrying one
electron. RHF refuses to describe that situation honestly because it insists
that both electrons remain in the same delocalized spatial orbital. The result
is not a small quantitative error. It is a qualitatively wrong picture of the
electrons.

RHF also has no way to describe a genuine odd-electron system such as the
hydrogen atom. There is simply no second electron available to occupy the same
spatial orbital with opposite spin. That is why the UHF example in this
repository begins with the hydrogen atom: it is the smallest case that forces
you to abandon the restricted ansatz.

## What UHF Changes And What It Keeps

UHF keeps the mean-field Hartree-Fock idea. The electrons still move in an
average field built from the current density, and the calculation is still an
SCF fixed-point problem. What changes is the wavefunction ansatz. Instead of one
orbital coefficient matrix $C$, we now have two:

$$
\varphi_i^\alpha(\mathbf{r}) = \sum_\mu C^\alpha_{\mu i}\chi_\mu(\mathbf{r}),
\qquad
\varphi_j^\beta(\mathbf{r}) = \sum_\mu C^\beta_{\mu j}\chi_\mu(\mathbf{r}).
$$

That one change forces the density and Fock matrices to split by spin:

$$
P^\alpha = C^\alpha_{\mathrm{occ}}\left(C^\alpha_{\mathrm{occ}}\right)^T,
\qquad
P^\beta = C^\beta_{\mathrm{occ}}\left(C^\beta_{\mathrm{occ}}\right)^T,
\qquad
P = P^\alpha + P^\beta.
$$

The Coulomb field depends on the total density because every electron repels
every other electron. Exchange depends only on same-spin density because the
exchange term comes from antisymmetry of the wavefunction. That gives the two
UHF Fock matrices:

$$
F^\alpha = H_{\mathrm{core}} + J(P) - K(P^\alpha),
\qquad
F^\beta = H_{\mathrm{core}} + J(P) - K(P^\beta).
$$

Once those matrices are built, the logic is completely familiar. Diagonalize
each spin channel, occupy the lowest orbitals in that channel, rebuild the two
densities, and iterate until they stop moving.

## Reading The Example

The key change in the code is as small as the key change in the theory:
instead of one occupation pattern, you now choose one for each spin channel.
In the example, that choice appears directly in calls such as

```cpp
cpu::scf::UHF uhf(n_alpha, n_beta);
uhf.basis = cpu::scf::BasisSet::STO3G;
uhf.tol = tol;
uhf.max_iter = max_iter;
uhf.run(frame);                       // eV energy on frame.meta[kEnergy]
double e_ha = uhf.energy();           // total energy [Hartree]
const auto& det = uhf.determinant();  // per-spin orbitals and occupations
```

The input you must choose carefully is the spin occupation. `n_alpha` and
`n_beta` are not bookkeeping details. They define the spin state. Their sum
must equal the total number of electrons, and their difference fixes
$M_S = (N_\alpha - N_\beta)/2$.

This is why the first example in the file is the hydrogen atom. It forces the
issue immediately: one electron means one occupied spin channel and one empty
spin channel. The second and third examples then show what happens when the
same unrestricted machinery is applied to $\mathrm{H}_2$ near equilibrium and
far from equilibrium.

An equilibrium closed-shell system such as $\mathrm{H}_2$ at $0.74$
Angstrom should come back with nearly identical $\alpha$ and $\beta$
channels and a spin contamination near zero. A hydrogen atom should come back
with `n_alpha = 1`, `n_beta = 0`, and a total energy near the STO-3G reference
of $-0.4666$ Hartree. A stretched $\mathrm{H}_2$ scan should show the
$\alpha$ and $\beta$ channels separating as the bond length grows.

## Charge and Multiplicity Conventions

**Concept.** An open-shell calculation is specified by two user-visible
integers: the molecular charge $q$ and the spin multiplicity $M$. Everything
the solver needs — how many $\alpha$ and how many $\beta$ electrons to occupy
— follows deterministically from those two numbers. Getting this translation
wrong is the single most common way to converge cleanly to the wrong state,
so Atomiverse makes it an explicit, validated step rather than a convention
you must remember.

**Math.** The electron count folds the charge in at integral-build time,

$$ N_e = \sum_A Z_A - q, $$

and the multiplicity $M = 2S + 1$ fixes the spin split through
$n_\alpha - n_\beta = M - 1$:

$$ n_\alpha = \frac{N_e + M - 1}{2}, \qquad
   n_\beta = \frac{N_e - M + 1}{2}. $$

These are integers only when $N_e$ and $M - 1$ have the same parity (even
$N_e$ pairs with odd $M$, odd $N_e$ with even $M$), and $n_\beta \ge 0$
requires $M \le N_e + 1$. Any violation means the requested state does not
exist.

**Algorithm.** The resolver is a struct you instantiate directly —
construction computes and validates in one step, and the resolved counts feed
the method constructor:

```cpp
// H2O+ : N_e = Σ Z − q = 9, doublet
cpu::scf::SpinResolution occ(/*n_electrons=*/9, /*multiplicity=*/2);
cpu::scf::UHF uhf(occ.n_alpha, occ.n_beta);  // e.g. (5, 4)
```

Worked table (all rows are validated against PySCF 2.13.1 and
regression-locked — see below):

| System | charge | $M$ | $N_e$ | $n_\alpha$ | $n_\beta$ |
|---|---|---|---|---|---|
| $\mathrm{H_2O}$ | 0 | 1 | 10 | 5 | 5 |
| $\mathrm{OH^-}$ | −1 | 1 | 10 | 5 | 5 |
| $\mathrm{H_2O^+}$ | +1 | 2 | 9 | 5 | 4 |
| O atom | 0 | 3 | 8 | 5 | 3 |

Rejected combinations (the constructor throws `std::invalid_argument`):
$\mathrm{H_2O}(0, 2)$ — even $N_e = 10$ with even $M$ breaks parity;
$\mathrm{H_2O^+}(+1, 1)$ — odd $N_e = 9$ with odd $M$ breaks parity;
$\mathrm{H}(0, 4)$ — $M = 4 > N_e + 1 = 2$ would need $n_\beta = -1$.

One convention contrast worth memorizing: PySCF's `mol.spin` is **not** the
multiplicity — it is $2S = n_\alpha - n_\beta = M - 1$. A PySCF doublet is
`spin=1`; the same state here is `SpinResolution(N_e, /*multiplicity=*/2)`.

**Limitations.** The resolver fixes bookkeeping only. It cannot tell you
*which* multiplicity is the ground state (the O atom is a triplet because of
Hund's rule, not because $M=3$ is the only valid input), and it does not
protect against spin contamination — that diagnostic is the next section.

<!-- Every (charge, M, n_alpha, n_beta) tuple and every rejected combination
     in this section is regression-locked by
     tests/cpu/scf/test_spin_resolution.cpp (test_cpu_scf_spin_resolution):
     WorkedTableMatchesTutorialCh03, OhMinusChargeFoldsIntoElectronCount,
     RejectsInvalidCombinations, FeedsUhfConfigRawCounts,
     FeedsUksConfigRawCounts. Change one side only in lockstep with the
     other. -->

## Spin Contamination: What It Means Here

The price of unrestricted flexibility is that the UHF determinant is usually
not an exact eigenfunction of $\hat S^2$. That is what the spin
contamination number measures. In the usual notation, the contaminated amount is

$$
\Delta\langle \hat S^2 \rangle
= \langle \hat S^2 \rangle_{\mathrm{UHF}} - S(S+1).
$$

Near equilibrium in a closed-shell singlet, this number should be close to
zero. When $\mathrm{H}_2$ is stretched and the unrestricted solution breaks
spin symmetry, the contamination rises. That is not automatically a bug. In the
bond-breaking regime it is part of the mechanism by which UHF escapes the
artificially restricted RHF picture.

The repository example makes this tangible. Near equilibrium, RHF and UHF stay
close because the electrons still behave like a closed-shell pair. As the bond
is stretched, the unrestricted solution lowers the energy by letting the two
spins localize differently. The growth of spin contamination is the visible
trace of that symmetry breaking.

## ΔSCF: Ionization Energies Beyond Koopmans

The previous chapter's Koopmans section (see the Koopmans discussion in
[Chapter 2: Restricted Hartree-Fock](ch02-rhf.md)) estimated the ionization
energy of a closed-shell molecule as $I_K = -\varepsilon_\text{HOMO}$: the
negative of the highest occupied orbital energy. That estimate is free — you
already have the orbital energies once the neutral SCF converges — but it rests
on a physical fiction. It assumes that when you remove an electron, the
remaining electrons do not move. They do. ΔSCF is the honest alternative: solve
the cation self-consistently and take an energy difference,

$$ I_{\Delta\text{SCF}} = E^{+}(N-1) - E(N). $$

Both energies are evaluated at the **same** geometry, basis, and convergence
threshold. That makes this a *vertical* ionization energy — the nuclei are
frozen at the neutral equilibrium while an electron is removed.

### Why ΔSCF Is Always Lower Than Koopmans

The relationship between the two estimates is not empirical; it is a two-line
variational argument. Koopmans' $-\varepsilon_\text{HOMO}$ is exactly the energy
of the $(N-1)$-electron determinant you get by deleting one electron from the
neutral orbitals and **freezing everything else**. The ΔSCF cation energy
$E^{+}$ is the result of variationally optimizing that same $(N-1)$-electron
wavefunction over the same Hilbert space. A variational optimum can only lower
the energy, so $E^{+} \le E_\text{frozen}$, and therefore

$$ I_{\Delta\text{SCF}} \le I_K = -\varepsilon_\text{HOMO}. $$

The gap between them, $I_K - I_{\Delta\text{SCF}} \ge 0$, is the **orbital
relaxation energy**: how much the remaining electrons lower their energy by
rearranging around the newly created hole. For a real molecule it is a fraction
of an eV to a couple of eV, and the ΔSCF value is the better one.

There is one clean teaching case where the two estimates are *exactly* equal.
For He and H₂ in a minimal (STO-3G) basis the cation has a single electron in a
spatial orbital that is fixed by symmetry — there is no second electron to
relax and no other orbital of the right symmetry to mix in. The relaxation
energy is identically zero, and the example prints $I_K = I_{\Delta\text{SCF}}$
to solver precision. Watch that row: it is the boundary condition that makes the
inequality meaningful everywhere else.

### Spin Bookkeeping for the Cation

Removing one electron from a closed-shell neutral turns an $N$-electron singlet
into an $(N-1)$-electron doublet ($M = 2$). Using the charge/multiplicity rules
from the section above, the electron count folds the $+1$ charge in at
integral-build time ($N_e = N - 1$), and the doublet split is

$$ n_\alpha = \frac{N}{2}, \qquad n_\beta = \frac{N}{2} - 1. $$

Concretely: the neutral has $N/2$ doubly-occupied levels; the cation keeps all
$N/2$ $\alpha$ orbitals and drops one $\beta$ electron. For H₂O ($N = 10$) that
is $(n_\alpha, n_\beta) = (5, 4)$; for He and H₂ ($N = 2$) it is $(1, 0)$.

### Warm-Starting the Cation

An open-shell SCF can converge cleanly to the *wrong* state — a hole in a lower
orbital rather than the HOMO, or an excited configuration. The defense is to
warm-start the cation from the neutral density instead of from a bare Hcore
guess. Atomiverse exposes this through `SCFInitialGuess`: seed both spin
channels with half of the converged neutral density,

```cpp
using Spin = cpu::scf::SlaterDeterminant::Spin;

cpu::scf::RHF neutral;
neutral.basis = basis;
neutral.charge = 0;
neutral.tol = tol;
neutral.max_iter = max_iter;
neutral.run(frame);

cpu::scf::SCFInitialGuess guess;                              // scf_guess.h
guess.P_alpha = neutral.determinant().density(Spin::alpha);   // half the total
guess.P_beta  = guess.P_alpha;                                // restricted density
guess.has_alpha = guess.has_beta = true;

cpu::scf::UHF cation(N/2, N/2 - 1);
cation.basis = basis;
cation.charge = 1;
cation.tol = tol;
cation.run(frame, &guess);
```

A bare Hcore cold start converges to the same energy on these small systems —
the example and its tests check that the two agree — but density warm-starting
is the habit to build: on larger or near-degenerate cations it is the
difference between the ground-state ion and a plausible-looking excited
solution.

### Monitoring Spin Contamination

Because the cation is unrestricted, check its $\langle \hat S^2 \rangle$ the
same way the previous section did. A clean doublet has

$$ \langle \hat S^2 \rangle = S(S+1) = \tfrac{1}{2}\cdot\tfrac{3}{2} = 0.75, $$

so the cation's `determinant().spin_contamination()` (which reports
$\langle \hat S^2 \rangle - S(S+1)$) should sit at ≈ 0. The example prints $\langle \hat S^2 \rangle =
\text{spin\_contamination} + 0.75$ for every cation; values noticeably above
0.75 warn that the unrestricted determinant has picked up higher-spin character
and the ΔSCF number should be read with caution.

### Vertical Versus Experiment

These are *vertical* ionization energies at a fixed neutral geometry. A measured
*adiabatic* ionization energy also lets the cation relax its geometry, which
lowers $E^{+}$ further and makes the adiabatic value smaller than the vertical
one. So even a well-converged ΔSCF number is only a semi-quantitative match to
an experimental ionization energy in a small basis — useful for trends and for
the pedagogy of relaxation, not for spectroscopic accuracy. Enlarging the basis
(STO-3G → 6-31G) and, eventually, adding correlation (a later chapter) moves the
numbers toward experiment; relaxing the cation geometry would convert vertical
to adiabatic. The worked table for all five systems in both bases is
[`examples/12_delta_scf.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/12_delta_scf.cpp).

## Running The Example

The build path is straightforward:

```bash
cmake -B build -DATV_BUILD_CUDA=OFF -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON
cmake --build build --target 03_uhf
./build/examples/03_uhf
```

The example is a good companion to this chapter because it places RHF and UHF
side by side instead of treating UHF as a separate world. You can watch the
method collapse back to RHF when the restricted approximation is safe, then
watch it diverge only when that extra freedom becomes physically necessary.

## Where Readers Usually Go Wrong

The most common mistake is giving the wrong spin occupation. If the total
electron count is wrong, everything downstream is wrong. If the spin split is
wrong, the solver may converge cleanly to the wrong state. The second common
mistake is to read nonzero spin contamination as a sign that the calculation
failed. In the bond-breaking example it is the opposite: contamination is the
visible trace of unrestricted symmetry breaking doing the job RHF could not do.

The more subtle mistake is to compare only total energies and ignore the rest of
the result. UHF is a spin-resolved method. If the $\alpha$ and $\beta$
densities, orbital energies, and contamination all look like the restricted
solution, then the unrestricted machinery has not actually changed the physics,
even if the solver was configured for UHF.

## SCF Stability Analysis(SCF 解的稳定性)

前面几节反复出现一个措辞:RHF 在拉伸的 H₂ 上"失败"。现在把这句话说精确。
SCF 收敛只保证你落在能量对轨道旋转的一个**驻点**上——梯度为零,仅此而已。
驻点可以是极小点,也可以是鞍点。判定属于哪一种的工具与几何优化里完全同源:
看二阶导。对轨道旋转参数求二阶导得到**轨道 Hessian**,其占据-虚轨道块由两
个矩阵 A、B 组成(实轨道、闭壳层空间轨道形式,化学家记号):

$$
{}^1A_{ai,bj} = \delta_{ab}\delta_{ij}(\varepsilon_a-\varepsilon_i)
              + 2(ai|jb) - (ij|ab), \qquad
{}^1B_{ai,bj} = 2(ai|bj) - (aj|bi)
$$
$$
{}^3A_{ai,bj} = \delta_{ab}\delta_{ij}(\varepsilon_a-\varepsilon_i)
              - (ij|ab), \qquad\qquad\;\;
{}^3B_{ai,bj} = -(aj|bi)
$$

**本征值的含义**:¹(A+B) 与 ³(A+B) 的最低本征值(单位 Hartree)就是能量面
沿最软一条轨道旋转方向的曲率。它 ≥ 0,驻点在该通道内是极小点;它 < 0,
存在一条严格下山的方向——你收敛到的不是"这个理论里最好的单行列式"。

Seeger 与 Pople(*J. Chem. Phys.* **66**, 3045 (1977))把方向分了类,这正是
**internal / external** 的区别:

- **internal(¹,单重态通道)**:旋转后仍是实的闭壳层 RHF 行列式。
  ¹(A+B) 最低本征值 < 0 意味着存在**另一个更低的 RHF 解**。
- **external(³,三重态通道)**:旋转打破 α/β 空间轨道的等同性,离开 RHF
  流形进入 UHF。³(A+B) 最低本征值 < 0 意味着**存在能量更低的 UHF 解**——
  RHF 在约束下是极小点,放开约束就是鞍点。

两个通道是不同的矩阵块:internal 稳定完全不担保 external 稳定。拉伸的 H₂
恰好就卡在这个陷阱里。

### 上手:拉伸 H₂ 的稳定性判定

`RHF::stability()` 在 `run()` 之后调用,返回两个通道的最低本征值、稳定标志,
以及(external 不稳定时)沿不稳定模式旋转过的系数矩阵——直接当 UHF 初猜用:

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.run(frame);                                   // H2, R = 1.5 Å

auto st = rhf.stability();
// st.internal_eigenvalue = +0.7089 Ha  -> internal 稳定
// st.external_eigenvalue = -0.2092 Ha  -> external 不稳定!
if (!st.external_stable) {
    // 破缺对称初猜:α 密度取旋转后的轨道,β 保持 RHF 轨道
    cpu::scf::SCFInitialGuess guess;
    // P_alpha 由 st.rotated_C 的占据列构成,P_beta 用 rhf.determinant()
    // 的密度 —— 完整代码见 regressions/scf/h2_rhf_stability。
    cpu::scf::UHF uhf{1, 1};
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.run(frame, &guess);
    // uhf.energy() = -0.95771 Ha < rhf.energy() = -0.91087 Ha
}
```

结果正是本章开头那条 RHF vs UHF 的 \(E(R)\) 曲线(见"Live result"一节的
molplot 图)在单点上的解释:R = 1.5 Å 处两条曲线的 0.047 Ha 间隙,就是
external 负本征值指出的那条下山路。沿 R 扫描两个通道的最低本征值,可以
看到 UHF 从 RHF 分岔的精确位置——³(A+B) 的最低本征值在 R ≈ 1.15 Å 处过零
(Coulson-Fischer 点),此后 RHF 一路是鞍点:

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true, tooltip: true, strokeWidth: 2}
data:
  values:
    - {R: 0.50, channel: "internal ¹(A+B)", eig: 1.478867}
    - {R: 0.735, channel: "internal ¹(A+B)", eig: 1.135177}
    - {R: 1.00, channel: "internal ¹(A+B)", eig: 0.910609}
    - {R: 1.20, channel: "internal ¹(A+B)", eig: 0.806038}
    - {R: 1.50, channel: "internal ¹(A+B)", eig: 0.708897}
    - {R: 2.00, channel: "internal ¹(A+B)", eig: 0.636671}
    - {R: 2.50, channel: "internal ¹(A+B)", eig: 0.617935}
    - {R: 3.00, channel: "internal ¹(A+B)", eig: 0.620746}
    - {R: 0.50, channel: "external ³(A+B)", eig: 0.803386}
    - {R: 0.735, channel: "external ³(A+B)", eig: 0.411452}
    - {R: 1.00, channel: "external ³(A+B)", eig: 0.123446}
    - {R: 1.20, channel: "external ³(A+B)", eig: -0.033128}
    - {R: 1.50, channel: "external ³(A+B)", eig: -0.209247}
    - {R: 2.00, channel: "external ³(A+B)", eig: -0.399883}
    - {R: 2.50, channel: "external ³(A+B)", eig: -0.510906}
    - {R: 3.00, channel: "external ³(A+B)", eig: -0.576100}
encoding:
  x: {field: R, type: quantitative, title: "R (Å)", scale: {zero: false}}
  y:
    field: eig
    type: quantitative
    title: "λ_min (Hartree)"
    scale: {nice: true}
  color: {field: channel, type: nominal, title: null}
  tooltip:
    - {field: channel, type: nominal}
    - {field: R, type: quantitative, format: ".2f"}
    - {field: eig, type: quantitative, format: ".6f"}
```

工程上还有一点值得记下:internal / external 只是同一个轨道 Hessian 核
(σ = (A±B)v 的 matrix-free 形式)的两个通道。这个核同时是二阶 SCF、CPHF
响应方程与 TDHF 激发能的底座——稳定性分析是它最小的消费者,却把它的数学
一次性验证到位。

### 章末对照:PySCF 最低本征值

外部 oracle(PySCF 2.14.0,`cart=True`,`conv_tol=1e-12`;命令与构建方式
逐字记录在 `regressions/scf/h2_rhf_stability/REFERENCE.md`)与 Atomiverse
`rhf.stability()` 的逐项对照,单位 Hartree:

| R (Å) | 通道 | Atomiverse | PySCF 2.14.0 | 差 |
|---|---|---|---|---|
| 0.735 | internal ¹(A+B) | +1.1351770407 | +1.1351770407 | < 1e-9 |
| 0.735 | external ³(A+B) | +0.4114522416 | +0.4114522416 | < 1e-9 |
| 1.5 | internal ¹(A+B) | +0.7088965814 | +0.7088965814 | < 1e-9 |
| 1.5 | external ³(A+B) | −0.2092471629 | −0.2092471629 | < 1e-9 |
| 1.5 | E_UHF(破缺) − E_RHF | −0.0468332388 | −0.0468332388 | < 1e-9 |

PySCF 自身的 `mf.stability(external=True)` 在 R = 1.5 Å 给出
"internal stable / external **unstable**" 的同一判定——教科书陷阱
(internal 稳定 ≠ 稳定)在两套实现里读数一致。

## ROHF:自旋纯的开壳层替代(Restricted Open-shell HF)

回顾一下本章反复出现的交换条件:UHF 给 α 与 β 各自独立的空间轨道,换来了
变分自由度,也换来了自旋污染——⟨S²⟩ 不再等于 S(S+1),波函数混入了更高
自旋态的成分。对拉伸 H₂ 那样的静态关联问题,污染是症状同时也是解药;但
对一个平衡几何附近、性质良好的开壳层分子(基态 O₂、甲基自由基),你常常
既想要开壳层占据,又想要一个**干净的自旋本征态**。这就是 ROHF 的位置:
**一套**空间轨道,分成闭壳(双占)、开壳(单占 α)、虚轨道三个空间,
β 占据严格是 α 占据的子集。单 MO 集这一个约束就解析地保证
⟨S²⟩ = S(S+1)——不是数值上碰巧小,而是恒等式(Roothaan, *Rev. Mod.
Phys.* **32**, 179 (1960))。

代价在哪里?能量泛函本身与 UHF 相同(Pople–Nesbet 双密度形式,对
O₂ 是 E = ½Tr[P_α(H+F_α)] + ½Tr[P_β(H+F_β)] + V_nn),但"一套轨道"
意味着 F_α 与 F_β 不能各自对角化——必须把两个自旋 Fock 耦合成**一个**
有效 Fock 再做一次限制式对角化。Roothaan 的分块办法是:在闭/开/虚三空间
的非对角块上放物理约定俗成的组合(co = F_β、ov = F_α、cv = F_c),对角块
放 F_c ≡ (F_α+F_β)/2。Atomiverse 复刻的是 PySCF canonical ROHF 的装配
(Guest–Saunders 系数族,`get_roothaan_fock` 的密度投影子代数),逐项一致。

### 诚实地讲清楚:耦合系数是约定,不是物理

对角块系数(教科书里的 A、B"耦合系数")有好几个著名选择:Roothaan 原始、
Guest–Saunders、McWeeny–Diercksen……它们**收敛到同一个总能量**——能量
泛函里根本没有这些系数——但给出**不同的轨道能**。换句话说:ROHF 的
E_total 是可移植的观测量,ε 是约定绑定量。拿 Atomiverse 的 ROHF 轨道能与
文献表格对照时,先确认对方用的是哪个系数族;本引擎只承诺与 PySCF canonical
(Guest–Saunders)约定逐位一致,验收也只在这个约定下比较 ε。把别的约定的
ε 差异当成 bug 来"修",是这个方法最经典的陷阱。

### 上手:O₂ 三重态与甲基自由基

与 UHF 完全同形——具体方法类、参数即成员、Frame 进 Frame 出:

```cpp
// O2 三重态, R = 1.2075 Å, (n_alpha, n_beta) = (9, 7), STO-3G
cpu::scf::ROHF rohf(9, 7);          // 要求 n_alpha >= n_beta,违反即抛
rohf.basis = cpu::scf::BasisSet::STO3G;
rohf.tol = 1e-12;
rohf.run(frame);
// rohf.energy()                          = -147.632166990682 Ha
// rohf.determinant().spin_contamination() = 0(解析恒等式,≤ 1e-10)

// 平面 D3h 甲基自由基, R(C-H) = 1.079 Å, (5, 4)
cpu::scf::ROHF ch3(5, 4);
ch3.basis = cpu::scf::BasisSet::STO3G;
ch3.run(frame_ch3);                 // -39.072132718153 Ha, <S2> = 0.75
```

尚未支持的路径都是显式拒绝而非静默走错分支:`forces()`(解析梯度需要含
耦合项的 Lagrangian,后续链节)、`mp2()`/`fci()`、`soscf`、`level_shift`
对 ROHF 一律抛出。

### 何时选谁

- **要自旋纯的参考态**(后续做以 ROHF 为参考的相关方法、算自旋密度相关
  性质、或对照要求 ⟨S²⟩ 干净的谱学量):ROHF。
- **要变分最优的单行列式、或处理静态关联/键断裂**:UHF——O₂ 上释放自旋
  约束能再降 1.8 mHa(见下表),污染 3.4e-3 就是这份自由度的收据;
  Tsuchimochi–Scuseria(*J. Chem. Phys.* **133**, 141102 (2010))把 ROHF
  重新表述为"约束 UHF",正是这幅图像。
- 平衡几何、污染本来就小:两者能量差别微小,选哪个更多取决于下游方法对
  参考态的要求。

### 章末对照:PySCF ROHF/UHF 标量

外部 oracle(PySCF 2.14.0,`scf.ROHF`/`scf.UHF`,STO-3G,`cart=True`,
`conv_tol=1e-12`;几何与命令逐字记录在
`regressions/scf/o2_rohf_sto3g/REFERENCE.md` 与
`tests/cpu/scf/test_rohf_solver.cpp` 文件头)与 Atomiverse 的逐项对照:

| 体系 | 量 | Atomiverse | PySCF 2.14.0 | 差 |
|---|---|---|---|---|
| O₂ (9,7) | E_total(ROHF) [Ha] | −147.632166990682 | −147.632166990682 | < 1e-12 |
| O₂ (9,7) | ⟨S²⟩(ROHF) | 2.000000000000 | 2.000000000000 | < 1e-14 |
| O₂ (9,7) | E_total(UHF) [Ha] | −147.633946785450 | −147.633946785450 | < 1e-12 |
| O₂ (9,7) | ⟨S²⟩(UHF) | 2.003410857681 | 2.003410857681 | < 1e-9 |
| CH₃ (5,4) | E_total(ROHF) [Ha] | −39.072132718153 | −39.072132718153 | < 1e-12 |
| CH₃ (5,4) | ⟨S²⟩(ROHF) | 0.750000000000 | 0.750000000000 | < 1e-14 |
| H₂ (1,1) | E(ROHF) − E(RHF) [Ha] | < 1e-8 | 0(精确退化) | — |

ROHF 的 ⟨S²⟩ 行不依赖任何外部引擎——它是单 MO 集的解析恒等式;表里放它
是为了与 UHF 行的 2.0034 形成对照:同一几何、同一基组,一行是约定保证的
2,一行是变分换来的污染。

## GHF 与自旋对称破缺

把本章走过的路摆成一排,你会发现它其实是一架**约束移除的梯子**:

| 方法 | 空间轨道 | 自旋结构 | 被移除的约束 |
|---|---|---|---|
| RHF | 一套,α/β 共享 | 共线,成对 | — |
| ROHF | 一套,分裂占据 | 共线,自旋纯 | "全部双占" |
| UHF | α/β 各一套 | 共线,可污染 | "α/β 同轨道" |
| **GHF** | 二分量旋量 | **非共线** | "自旋沿一根轴量子化" |

RHF、ROHF、UHF 的每个轨道都还是"纯 α"或"纯 β"——整个分子共享一根
自旋量子化轴,区别只在占据方式。GHF(generalized Hartree-Fock)拆掉
最后这根轴:每个分子轨道是一个**二分量旋量**
φ_i = (φ_i^α; φ_i^β),两个分量各自是完整的 AO 展开。密度矩阵随之变成
2n×2n 的 2×2 自旋分块 [[P^αα, P^αβ], [P^βα, P^ββ]],多出来的
**自旋非对角块 P^αβ** 正是两条独立自旋通道(UHF 的数据结构)原则上
无法表示的东西——自旋方向可以逐点变化,这就是非共线磁性的语言。
Fock 矩阵的结构很能说明物理:库仑项只看电荷密度
(F 的对角块含 J[P^αα+P^ββ]),交换作用于全部四块
(F^{σσ'} = δ_{σσ'}(h + J) − K[P^{σσ'}])——把自旋块耦合起来的是交换,
不是静电。

诚实地交代分类学:Fukutome 早在 1981 年就把 HF 平均场按时间反演与自旋
对称性的破缺方式完整分成**八类**(*Int. J. Quantum Chem.* **20**, 955
(1981), DOI 10.1002/qua.560200502)——RHF(他叫 TICS)、UHF(ASDW)、
本节的实代数 GHF 落在其中的 torsional spin density wave(TSDW)类;
完整的八类还需要复系数(电流承载态)与自旋轨道耦合才全部现身。本引擎
这一环**只做实代数 GHF**:磁化强度被限制在自旋空间的一张平面里
(实代数中 σ_x、σ_z 是实矩阵而 σ_y 不是),共面的非共线自旋恰好够用;
复 GHF 与自旋轨道耦合是后续链节。一般自旋轨道的系统表述见
Helgaker–Jørgensen–Olsen《Molecular Electronic-Structure Theory》§10。

### 塌缩定理:多出来的自由度什么时候真的有用?

没有自旋轨道项时,**每一个 UHF 解都是 GHF 的驻点**——把 UHF 密度嵌成
块对角的旋量密度,SCF 映射保持它不动。更强的是:对自旋**无阻挫**的
体系,GHF 的极小值就是共线的,E_GHF = E_UHF 严格成立。这是 GHF 实现
最重要的正确性哨兵——在 H₂ 上跑 GHF,能量必须塌缩回 UHF 到 1e-8 Ha,
差一点都说明旋量 Fock 装配有 bug。

自由度真正兑现的地方是**自旋阻挫**:正三角形 H₃,三个自旋两两都想
反平行,共线世界里三角形凑不齐(两个反平行了,第三个对谁都不对)。
GHF 的出路是把三个磁矩摆成共面 120° 的 Néel 扇形——每一对夹角 120°,
"尽可能反平行"被三方平摊。代价函数上这是实打实的能量:

```molvis {format="xyz" representation="ball-and-stick" controls="view" height="260px"}
3
name=H3_equilateral Connct="[0,1],[1,2],[2,0]"
H  0.0000  0.0000  0.0000
H  1.0000  0.0000  0.0000
H  0.5000  0.8660254  0.0000
```

收敛后的每原子 Mulliken 自旋矢量(实代数:磁化位于 x–z 自旋平面,
分量 m_x = diag[(P^αβ+P^βα)S]、m_z = diag[(P^αα−P^ββ)S] 按原子求和)
就是教科书图像本身——molvis 的自旋矢量渲染把它画在结构上,数值上:

| 原子 | m_x | m_z | \|m\| | 方向角 |
|---|---|---|---|---|
| H(0) | −0.000000 | +0.622166 | 0.622166 | 0° |
| H(1) | +0.538812 | −0.311083 | 0.622166 | +120° |
| H(2) | −0.538812 | −0.311083 | 0.622166 | −120° |

三个磁矩等长、两两严格 120°——不是初猜的残影,是变分收敛的结果
(初猜角度并不精确等于 120° 也收敛到这里)。

### 上手:三角 H₃

与 UHF/ROHF 同形:具体方法类、参数即成员、Frame 进 Frame 出。两个
GHF 特有的成员:`n_electrons`(旋量占据需要总电子数,不设即抛)与
`guess_spin_rotation`(初猜磁矩方向角步长,弧度)。

```cpp
// 正三角 H3, 边长 1.0 Å, STO-3G —— 与 PySCF oracle 逐字同一几何
cpu::scf::GHF ghf(3);               // N = 3 个电子进 2n 个旋量的 aufbau
ghf.basis = cpu::scf::BasisSet::STO3G;
ghf.tol = 1e-12;
ghf.guess_spin_rotation = 2.0 * M_PI / 3.0;   // 三角形的几何适配 Néel 角
ghf.run(frame);
// ghf.energy()  = -1.340440343541 Ha —— 低于最优共线 UHF 4.46 mHa
// ghf.determinant().generalized()   = true
// ghf.determinant().spin_block(Spin::alpha, Spin::beta)  // P^αβ ≠ 0
```

**初猜必须显式破坏共线性,这是本方法第一陷阱。** 共线初猜被 SCF 映射
严格保持(共线密度的 Fock 是块对角的,迭代永远出不了共线子空间),
DIIS 会安安稳稳收敛到共线驻点——能量正是 UHF 的 −1.335980 Ha,收敛
判据全绿,你却停在鞍点上。`guess_spin_rotation = 0` 的这个行为被测试
钉死(`test_cpu_scf_ghf` 的 CollinearGuessStaysOnUhfSaddle)。初猜的
构造是非共线磁性的标准做法(平面波世界的 MAGMOM/SAXIS):电荷密度
½P_c⊗1₂ 不旋转,每个 AO 叠加一个满强度局域磁矩,方向角
θ_μ = `guess_spin_rotation`·μ;角度太小(≤0.3 rad)会滑回共线吸引盆,
三角 H₃ 要用几何适配的 2π/3。

还有一个值得诚实交代的细节:同一几何上共线 UHF 本身就有**两个**驻点
——对称自旋纯二重态(−1.328176 Ha,⟨S²⟩ = 0.75)与更低的对称破缺解
(−1.335980 Ha,⟨S²⟩ = 0.838)。本引擎的冷启动 UHF 落在前者,PySCF 的
MINAO 初猜落在后者;GHF 的非共线极小值在**两者之下**,并且
PySCF `mf.stability()` 证认它是真极小(内部稳定),而对共线 UHF 做
`stability(external=True)` 报告 UHF→GHF 不稳定——阻挫的谱学签名。
完整的证认流程逐字记录在 `regressions/scf/h3_ghf_sto3g/REFERENCE.md`。

本环节未支持的路径一律显式拒绝(不静默走错分支):解析梯度 `forces()`、
`mp2()`/`fci()`、`soscf`、`level_shift`、EDIIS/ADIIS、非 Hcore 初猜,
对 GHF 全部抛出;旋量交换需要稠密 ERI(`compute_eri`,构造时默认打开)。

### 章末对照:PySCF GHF/UHF 标量

外部 oracle(PySCF 2.14.0,`scf.GHF`/`scf.UHF`,STO-3G,`cart=True`,
`conv_tol ≤ 1e-12`;几何、命令与非共线极小值的证认流程逐字记录在
`regressions/scf/h3_ghf_sto3g/REFERENCE.md`):

| 体系 | 量 | Atomiverse | PySCF 2.14.0 | 差 |
|---|---|---|---|---|
| H₃ 正三角 | E_total(GHF, 非共线) [Ha] | −1.340440343541 | −1.340440343541 | < 1e-12 |
| H₃ 正三角 | E_total(UHF, 对称破缺共线) [Ha] | −1.335980054018¹ | −1.335980054018 | < 1e-10 |
| H₃ 正三角 | E_total(UHF, 对称二重态) [Ha] | −1.328175755381 | −1.328175755381 | < 1e-10 |
| H₃ 正三角 | ‖P^αβ‖_F(GHF) | 0.9780588 | 0.9780588 | < 1e-6 |
| H₃ 正三角 | E_GHF − E_UHF(最优共线) [Ha] | −0.004460 | −0.004460 | — |
| H₂ (R=0.74) | E_GHF − E_UHF [Ha](塌缩) | < 1e-15 | 0(定理) | — |

¹ 引擎的 GHF 在 `guess_spin_rotation = 0` 时收敛到的共线驻点能量——
即"共线陷阱"行;引擎冷启动 UHF 本身落在对称二重态一行。

The next chapter keeps the spin-resolved structure of UHF but changes the mean
field itself. Instead of explicit Hartree-Fock exchange, it uses an
exchange-correlation functional evaluated on a real-space grid:
[Chapter 5: Unrestricted Kohn-Sham DFT (UKS)](ch05-kohn-sham-dft.md).
