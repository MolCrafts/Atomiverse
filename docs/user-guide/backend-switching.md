# Switching Backends

This chapter explains a practical rule that appears all over Atomiverse: the
problem description stays on the CPU side, while the execution backend changes.
If you understand that one rule, moving between the tutorial examples and the
CUDA drivers stops feeling like a rewrite and starts feeling like a change of
execution model.

## The Stable Part: `cpu::Frame`

In user-facing code, the system is described in `cpu::Frame`. Positions,
masses, atom types, and optional topology live there first. That is true even
when the eventual calculation runs on CUDA. The backend decision changes how
the calculation is executed, not how the physical system is described.

This is easiest to see for RHF. The [tutorial Practice block](../tutorial/ch02-rhf.md)
builds the molecule in `cpu::Frame` and calls `cpu::scf::RHF`. The CUDA path
starts from the same `cpu::Frame`; only the evaluator and solver backend change.

## Same Shape On Both Devices

The switch is one type name. Parameters are constructor arguments or members
on the concrete method, and `Frame` is the only token that crosses:

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-10};
rhf.run(frame);                       // frame.meta.get(kEnergy) [eV]
```

```cpp
cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-10};
rhf.run(frame);                       // same Frame, same key, same units
```

Integrals, the SCF iteration, and the density are private workspace inside the
method on either device — there is no integral bag to hand around and no
`*Config` to fill in. Forces are `rhf.forces(frame)` on both.

The CPU path is the best one for learning, debugging, and validating the
mathematics: the code is compact and the linear algebra stays close to the
physics. The CUDA path keeps the calculation device-resident; for the shells
its kernels cover, no scientific data returns to the host mid-SCF.

Either method also binds as an evaluator inside an outer driver, again with the
same shape on both devices:

```cpp
cuda::Evaluate eval;
eval.set_evaluator(cuda::scf::RHF{cpu::scf::BasisSet::STO3G});
eval.build();
eval.run(frame);
```

That is the main backend switch in Atomiverse. The physical problem is
described once; only the device that executes it changes.

## A Good Progression

When you are building a new workflow or debugging a result, start on the CPU
side first. Work through the tutorial Practice snippets for RHF, UHF, and UKS
on CPU, then move one representative case to CUDA with the same `cpu::Frame`.

This progression keeps two concerns separate. First, you confirm that the
method and the inputs are right. Second, you change the backend and check that
the execution model still reproduces the same physics.

So the two backend modes serve different purposes:

- CPU solvers are the clearest reference path.
- CUDA drivers are the staged high-performance execution path.

The important thing is that you do not need two mental models of the
underlying system. You need one model of the physical problem and two ways to
execute it.
