# Tutorials

Scientific methods end-to-end: theory, runnable examples, and literature
comparisons. Install, build, and the Driver / Component vocabulary are in the
[User Guide](../user-guide/) — start there if you have not built the project
yet.

Every chapter shares the same `cpu::Frame` data model, the same
Driver/Component pattern, and the same units (eV, Å, fs, amu): **one loop,
many physics**.

## How to use a chapter

1. **Build the example** named at the top (`-DATV_BUILD_EXAMPLES=ON`).
2. **Run it** and compare printed numbers to
   [`examples/results/`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/).
3. **Read the theory** with the code beside you.
4. **Check the literature column** — docs never invent Atomiverse numbers.

Interactive figures use **MolPlot** (line / scatter trends) and **MolVis**
(structures). Sparse two-point comparisons stay in tables.

Quantum-chemistry path: **Chapter 2**. Classical MD path: **Chapter 10**.
Architecture orientation: [Chapter 0](ch00-architecture.md) or
[User Guide → Core concepts](../user-guide/concepts.md).

## I. Foundations

| Ch | Title | Example | Results |
|----|-------|---------|---------|
| [0](ch00-architecture.md) | Tour of the CPU backend | — | — |
| [1](ch01-foundations.md) | Systems, units, precision, validation | — | FD gradient checks |

## II. Localized-basis electronic structure

| Ch | Title | Example | Results |
|----|-------|---------|---------|
| [2](ch02-rhf.md) | Atomic orbitals → single-point RHF | `01_scf`, `02_rhf`, `16_density_cube` | [`02_rhf.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/02_rhf.json), cubes |
| [3](ch03-open-shell.md) | Unrestricted Hartree–Fock | `03_uhf` | [`03_uhf_pes.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/03_uhf_pes.json) |
| [4](ch04-basis-sets.md) | Gaussian basis sets and convergence | `11_koopmans` | [`11_koopmans.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/11_koopmans.json) |

## III. Density-functional theory

| Ch | Title | Example | Results |
|----|-------|---------|---------|
| [5](ch05-kohn-sham-dft.md) | Unrestricted Kohn–Sham DFT | `04_uks` | [`04_uks.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/04_uks.json) |
| [6](ch06-hybrid-dft.md) | Hybrids and analytic gradients | `05_geom_opt` | energy / force convergence |

## IV. Correlation methods

| Ch | Title | Example | Results |
|----|-------|---------|---------|
| [7](ch07-correlation.md) | Beyond mean field (MP2, FCI) | `13_mp2`, `14_fci` | [`13_mp2.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/13_mp2.json), [`14_fci.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/14_fci.json) |

## V. Periodic / plane-wave electronic structure

| Ch | Title | Example | Flag |
|----|-------|---------|------|
| [8](ch08-plane-wave.md) | Plane-wave DFT | `08_pw` | [`08_pw.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/08_pw.json) (`-DATV_BUILD_PW=ON`) |
| [9](ch09-gpw.md) | Gaussian and plane waves | `09_gpw` | [`09_gpw.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/09_gpw.json) (`-DATV_BUILD_GPW=ON`) |

## VI. Classical simulation

| Ch | Title | Example | Results |
|----|-------|---------|---------|
| [10](ch10-classical-md.md) | Classical MD on the CPU | `07_md_lj` | [`07_md_lj.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/07_md_lj.json) |
| [11](ch11-ensembles.md) | Thermostats and barostats | `07_md_lj` + tests | NVT/NPT conservation |
| [12](ch12-geometry-optimization.md) | L-BFGS / CG minimization | `05_geom_opt`, `15_am1` | [`05_geom_opt.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/05_geom_opt.json), [`15_am1.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/15_am1.json) |

## VII. Analysis & response

| Ch | Title | Example |
|----|-------|---------|
| [13](ch13-observables.md) | Observables and trajectories | RDF / MSD observers |
| [14](ch14-vibrations.md) | Vibrations and thermochemistry | `cpu::vib::Hessian` |
| [15](ch15-cross-method.md) | Swap the evaluator | `06_aimd` |

## VIII–IX. Beyond pairs & semiempirical

| Ch | Title | Example | Results |
|----|-------|---------|---------|
| [16](ch16-many-body-potentials.md) | EAM and Stillinger–Weber | potentials tests | FD oracles |
| [17](ch17-semiempirical-am1.md) | AM1 | `15_am1` | [`15_am1.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/15_am1.json) |

## X. The Hartree–Fock arc (Jensen-style)

读完 ch02–ch04 + ch18–ch20,应相当于读完 Frank Jensen《Introduction to
Computational Chemistry》的 Hartree–Fock 理论部分:方法为主干,算法、技巧与
技术织入其中。每章以"与 PySCF 对照"表收尾(标量一律表格,禁止柱状图;方法学
见附录)。ch02(HF 理论)、ch03(开壳方法)、ch04(基组近似)随
hf-capability-gaps 链扩写;下列三章为该链新增:

| Ch | Title | 填充环 |
|----|-------|--------|
| [18](ch18-scf-techniques.md) | SCF 技巧:初猜、混合与二阶方法 | 07, 09 |
| [19](ch19-hf-large-systems.md) | 大体系 HF:Schwarz → RI → Ewald/PME → FMM/CFMM | 12, 19–21 |
| [20](ch20-hf-properties-environment.md) | HF 性质与环境:偶极到相对论 | 03–04, 13–16, 18 |
| [附录](appendix-pyscf-oracle.md) | PySCF 对照方法学(oracle 协议) | 01 |

---

Acceptance policy (external-engine parity) lives in
[`benchmarks/README.md`](https://github.com/MolCrafts/Atomiverse/blob/dev/benchmarks/README.md).
For a one-page SCF map see the
[SCF Learning Path](../user-guide/scf-roadmap.md).
