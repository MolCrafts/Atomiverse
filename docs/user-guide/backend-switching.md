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

Open shell is the same shape. `n_alpha` and `n_beta` are constructor
arguments, and `run` / `forces` still take one `cpu::Frame`:

```cpp
cuda::scf::UHF uhf{cpu::scf::BasisSet::STO3G, /*n_alpha=*/1,
                   /*n_beta=*/0, /*charge=*/1};
uhf.run(frame);                       // frame.meta.get(kEnergy) [eV]
cpu::scf::GradientResult g = uhf.forces(frame); // kAtom_fx/fy/fz [eV/Å]
```

Integrals, the SCF iteration, and the density are private workspace inside the
method on either device — there is no integral bag to hand around and no
`*Config` to fill in. Forces are `forces(frame)` on either device.

The CPU path is the best one for learning, debugging, and validating the
mathematics: the code is compact and the linear algebra stays close to the
physics. The CUDA path keeps the calculation device-resident; for the shells
its kernels cover, no scientific data returns to the host mid-SCF.

!!! warning "The CUDA SCF covers far less than the CPU one"

    The shape is the same; the coverage is not. Device Hartree–Fock is
    `cuda::scf::RHF` and `cuda::scf::UHF` — there is no `RKS`, `UKS`,
    `ROHF`, or `GHF` on device. Device-native integrals and analytic forces
    cover Cartesian RHF shells with l ≤ 3 (`launch_*_sp` / `launch_*_lgt1`,
    spherical def2 / cc via CartToSph) and UHF on STO-3G s/p.
    `cuda::scf::UHF` **throws** outside that envelope
    (`launch_eri_sp`, `launch_uhf_forces_sp`). `cuda::scf::RHF` throws
    `launch_eri_g` (l ≥ 4) or
    `launch_rhf_forces_lgt1` (forces outside STO-3G l ≤ 1) and does not fill
    ERIs on the host. The plane-wave
    backend is similarly narrower on device
    than on host (LDA, spin-unpolarized). The
    [roadmap](../roadmap/index.md) is the per-method statement of what each
    backend actually does.

Either method also binds as an evaluator inside an outer driver, again with the
same call on both devices:

```cpp
cuda::Evaluate eval;
eval.set_evaluator(cuda::scf::RHF{cpu::scf::BasisSet::STO3G});
eval.build();
eval.run(frame);
```

`cuda::scf::RHF` and `cuda::scf::UHF` are each `cuda::Driver<scf::Phase>` with the stages
`fock → mix → diagonalize → density → check`. It is not itself a Component.
`set_evaluator` mounts it in one Plan slot. That slot's single call runs the
SCF loop. The outer Evaluate or MD plan still sees one component.

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
