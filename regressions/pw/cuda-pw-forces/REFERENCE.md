# Reference — CUDA plane-wave analytic ionic forces

## Method

Device twin of `cpu::pw::pw_forces_analytic`:

- local Hellmann–Feynman from ρ(G) × v(G) (device kernel)
- Kleinman–Bylander nonlocal from β, ψ (device kernels)
- Ewald ion–ion (shared host classical sum)

Public API: `cuda::pw::pw_forces_analytic(frame, method)` [eV/Å].

## System

2-atom Si FCC primitive cell, lattice \(a = 5.43\) Å, second atom stretched
by \(+0.08\) Å along \(x\) so forces are nonzero and exercise KB projectors.

Settings (both backends):

| Member | Value |
| --- | --- |
| `gth_name` | `GTH-PBE-q4` |
| `ecut_ry` | 8.0 |
| `k_mesh` | 1 (Γ) |
| `tol` | 1e-6 |
| `max_iter` | 40 |
| `xc` | `XCType::LSDA` |

## Oracle

Same-settings CPU `cpu::pw::pw_forces_analytic` (not CP2K; force twin is the
landed CPU analytic path). Optional class: CPU FD gate in
`tests/cpu/planewave/test_pw_forces_fd.cpp` already covers analytic↔FD.

## Pin / gate

\[
\max_{i,\alpha}\lvert F^{\mathrm{cuda}}_{i\alpha} - F^{\mathrm{cpu}}_{i\alpha}\rvert
  \le 5\times 10^{-3}\ \mathrm{eV/Å}
\]

and \(\max |F^{\mathrm{cpu}}| > 0.05\) eV/Å (non-vacuous).

**GH200 job 1050105:** measured \(\max|\Delta F| = 2.86\times 10^{-13}\) eV/Å
(CPU = CUDA \(\max|F| = 2.517979\) eV/Å; force-path SCF energy both
\(-13.85556302\) Ry). Bound kept at design-time \(5\times 10^{-3}\).

**Root cause of earlier \(\sim 2\times 10^{-2}\) residual:** CPU force path
called `XCBuilder::step(ws)` without `method.xc`, so the NL band re-solve
rebuilt \(V_{\mathrm{eff}}\) with the default **PBE** while SCF (and CUDA)
used **LSDA**. Fixed to `xc.step(ws, method.xc)`.

Not bit-exact by design (independent SCF + FFT/eigensolve paths) but
observed residual is at floating-point noise for this cell.

## Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU force parity as above |
| Wall | **perf_exempt** — correctness foothold on small cell |

## Non-goals

Stress product, collinear spin, metallic smear, energy-path changes.
