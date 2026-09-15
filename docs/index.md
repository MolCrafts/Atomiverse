---
title: Atomiverse
hide:
  - navigation
  - toc
hero:
  kicker: Manual
  title: Atomiverse
  description: >-
    Unified molecular simulation engine. Different methods live in one engine
    and share the same way of building and running a calculation.
  actions:
    - { label: Get started, href: user-guide/getting-started/, style: primary }
    - { label: Tutorial book, href: tutorial/ }
---

<h1 class="molcrafts-sr-only">Atomiverse</h1>

<div class="molcrafts-manual-home" markdown>

<section class="molcrafts-manual-section molcrafts-manual-section--stack" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Why Atomiverse</span>

## One engine. Every method. Every backend.

</div>

<div class="molcrafts-manual-grid molcrafts-manual-grid--cols-3">
  <div>
    <strong>Unified</strong>
    <em>Not a stack of separate codes — MD, SCF, DFT, and sampling share one model.</em>
  </div>
  <div>
    <strong>Multi-backend</strong>
    <em>Run the same calculation on CPU or GPU without rewriting the workflow.</em>
  </div>
  <div>
    <strong>Composable physics</strong>
    <em>Classical force fields or ab initio — change the energy source, keep the run.</em>
  </div>
</div>

```cpp
cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.5});     // fs

sim.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 6.8});  // classical MD
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});  // cutoff, skin
// or, for ab-initio MD, swap the energy source:
// sim.set_evaluator(cpu::scf::RHF{cpu::scf::BasisSet::STO3G});

sim.build();
sim.run(frame, 1000);
```

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--stack" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Capabilities</span>

## What it runs

</div>

<div class="molcrafts-manual-grid molcrafts-manual-grid--cols-3">
  <a href="tutorial/ch10-classical-md/">
    <strong>Classical MD</strong>
    <em>LJ, bonded, many-body, NVE/NVT/NPT/NPH, PME.</em>
  </a>
  <a href="tutorial/ch12-geometry-optimization/">
    <strong>Optimization</strong>
    <em>L-BFGS / CG / FIRE, cell relaxation.</em>
  </a>
  <a href="user-guide/methods/#monte-carlo">
    <strong>Monte Carlo</strong>
    <em>NVT / NPT / μVT, same force pipeline.</em>
  </a>
  <a href="tutorial/ch02-rhf/">
    <strong>Gaussian SCF</strong>
    <em>HF, KS-DFT, post-HF, excited states.</em>
  </a>
  <a href="tutorial/ch08-plane-wave/">
    <strong>Plane-wave DFT</strong>
    <em>Periodic KS, k-points, stress, phonons.</em>
  </a>
  <a href="tutorial/ch09-gpw/">
    <strong>GPW DFT</strong>
    <em>GTH + Gaussians, DFT+U, dispersion, OT.</em>
  </a>
  <a href="tutorial/ch14-vibrations/">
    <strong>Vibrations</strong>
    <em>FD Hessian, thermochemistry.</em>
  </a>
  <a href="tutorial/ch17-semiempirical-am1/">
    <strong>Semiempirics</strong>
    <em>AM1/RM1, DFTB0/2/3, GFN-xTB.</em>
  </a>
  <a href="tutorial/ch13-observables/">
    <strong>Observables</strong>
    <em>RDF, MSD, stress, charges, cubes.</em>
  </a>
</div>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Docs</span>

## Where to go

</div>

<div class="molcrafts-manual-list">
  <a href="user-guide/">
    <strong>User Guide</strong>
    <em>Install, build, and how to run a calculation.</em>
  </a>
  <a href="tutorial/">
    <strong>Tutorial</strong>
    <em>Methods end-to-end — examples, numbers, literature.</em>
  </a>
  <a href="developer/">
    <strong>Developer</strong>
    <em>Internals and how to extend the engine.</em>
  </a>
  <a href="roadmap/">
    <strong>Roadmap</strong>
    <em>What ships today, on which backend, and what is next.</em>
  </a>
</div>

</section>

</div>
