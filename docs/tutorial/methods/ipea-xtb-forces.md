# IPEA-xTB analytic nuclear forces — off-eq CH4

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ipea-xtb-forces` |
| Regression | `regressions/semiempirical/ipea-xtb-forces/` |
| Public API | `cpu::semiempirical::IPEA` (`with_forces = true`); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; IPEA is a GFN-xTB sibling) |
| Queue | TB method loop — wave 11 |
| Roadmap | `docs/roadmap/semiempirical.md` §3 IPEA-xTB analytic gradient |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Closed-form nuclear gradient of production **IPEA-xTB** (Ásgeirsson, Bauer,
Grimme, *Chem. Sci.* **8**, 4879 (2017)) on off-equilibrium methane, using
Atomiverse’s static H/C/N/O tables transcribed from tblite `ipea1.f90` /
xtb `param_ipea-xtb.txt`. After one shell-resolved Klopman–Ohno SCC
(\(g_{\exp}=1\)), forces are \(\mathbf{F}=-\nabla E\) at that density —
Hellmann–Feynman + Pulay on \(\partial S\), CN/shell-polynomial chains,
\(\partial\gamma/\partial R\), pair repulsion, and D3(BJ) — written to
`frame.span(kAtom_f*)` in eV/Å. This is the **xtb IPEA `--grad` module
boundary** (not GFN1, not a \(6N\) energy finite-difference). Public
surface: concrete method `cpu::semiempirical::IPEA` with members
(`charge`, `scc_tol`, `with_forces`); no public `*Config`, no `make_*` /
`build_*`. Product gate: `reg_ipea-xtb-forces`. Energy identity already
ships on methane (`reg_ipea-xtb-small`; 8.90 meV residual is the same
kernel hole). Off-eq water FD identity lives in `test_cpu_xtb_ipea`
(polar ~0.5 eV hole vs xtb is not a gradient defect).
