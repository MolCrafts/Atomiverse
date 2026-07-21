# Switching Backends

This chapter explains a practical rule that appears all over Atomiverse: the
problem description stays on the CPU side, while the execution backend changes.
If you understand that one rule, moving between the tutorial examples, the CUDA
drivers, and the LibTorch backend stops feeling like a rewrite and starts
feeling like a change of execution model.

## The Stable Part: `cpu::Frame`

In user-facing code, the system is described in `cpu::Frame`. Positions,
masses, atom types, and optional topology live there first. That is true even
when the eventual calculation runs on CUDA. The backend decision changes how
the calculation is executed, not how the physical system is described.

This is easiest to see in the RHF pair of examples. The CPU reference path is
[`examples/02_rhf.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/02_rhf.cpp). The CUDA
translation of the same problem is
[`examples/02_rhf.cu`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/02_rhf.cu). Both begin
by constructing the same molecular geometry in `cpu::Frame`. The difference
comes only after the integrals and workflow are chosen.

## CPU Solver Versus CUDA Driver

The CPU path exposes the solver lifecycle directly:

```cpp
auto integrals = /* IntegralPack is private — use RHF */;
auto config = /* use cpu::scf::RHF; rhf.tol = tol */;
config.max_iter = 100;
cpu::scf::RHF rhf /* was SCFSolver */;

const auto& result = rhf.run(frame);
```

This is the best path for learning, debugging, and validating the mathematics.
The code is compact and the linear algebra stays close to the physics.

The CUDA path keeps the same `cpu::Frame` and the same integral data, but hands
the iterative work to a device-resident evaluator driven by `cuda::Evaluate`.
The inner SCF cycle (Fock build, diagonalize, density, convergence) lives
inside `cuda::scf::SCFSolver`, wrapped by the `cuda::scf::SCF`
component:

```cpp
auto integrals = /* IntegralPack is private — use RHF */;

cuda::scf::SCF scf{integrals, tol, /*max_iter=*/100};

cuda::Evaluate eval;
eval.on(ATV::evaluate::Phase::evaluate).add(std::move(scf));
eval.build();
eval.run(frame);              // frame.meta.get(kEnergy) populated
```

That is the main backend switch in Atomiverse. The physical problem is the
same. The execution style changes from a direct CPU solver call to a staged
CUDA evaluator.

## A Good Progression

When you are building a new workflow or debugging a result, start on the CPU
side first. The tutorial sequence is arranged that way on purpose:
`examples/01_scf.cpp`, `examples/02_rhf.cpp`,
`examples/03_uhf.cpp`, and `examples/04_uks.cpp` are all
CPU-first examples. Once the result looks correct, move one representative case
to CUDA. For RHF, that representative translation is
`examples/02_rhf.cu`.

This progression keeps two concerns separate. First, you confirm that the
method and the inputs are right. Second, you change the backend and check that
the execution model still reproduces the same physics.

## Where LibTorch Fits

The LibTorch backend is not another version of the chemistry tutorial track. It
is a tensor-backed execution layer for differentiable operators. In that path
you still start from CPU-side problem description when needed, but the next step
is a transfer into `ATV::torch::Frame` through functions such as `from_cpu` or
`to_torch`, depending on whether you are starting from host data or an existing
CUDA frame.

So the three backend modes serve different purposes:

- CPU solvers are the clearest reference path.
- CUDA drivers are the staged high-performance execution path.
- LibTorch is the differentiable tensor-backed path for optimization and
  interop.

The important thing is that you do not need three mental models of the
underlying system. You need one model of the physical problem and three ways to
execute it.
