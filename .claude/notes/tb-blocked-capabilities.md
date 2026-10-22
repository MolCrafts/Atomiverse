# TB / SE blocked capabilities

Rows that are **not ordinary pending**. A row leaves this list only when its
gate is actually removed.

## Permanent rules (never capabilities)

| Id | Rule |
| -- | ---- |
| `mopac-as-dftb-oracle` | Forbidden: pin DFTB to OpenMOPAC |
| `gfn-ff` | Scope-out: classical GFN-FF belongs to the MD/FF surface |
| `om3-nddo` | No family-locked pin: OM3 is outside the OpenMOPAC surface; adding Thiel OMx is an oracle-set decision, not a missing feature |

## Blocked (science, multi-wave)

| Slug | Gate | Unblock |
| ---- | ---- | ------- |
| `noncollinear-spin-dftb` | Spinor Hamiltonian (2-component density), not a parameter on collinear | Spec + multi-wave after collinear solid |
| `spin-orbit-dftb` | SOC/SK design | After noncollinear |

## Not blocked (ordinary pending)

- Shell-resolved collinear spin (`ShellResolvedSpin = Yes`) and shell-resolved
  \(U_\ell\).
- `stda-xtb` waits on AES-in-Fock / production GFN2 ground state; staff Phase C
  under the queue.
- `gfn2-camm-r`: GFN2 matches xtb 6.7.1 / tblite 0.4.0 term by term; the
  `F_H1,y` pin is tblite 0.4.0 at the identical input (xtb 6.7.1 prints an
  orientation artifact there). See its `REFERENCE.md`.

Related: [dftb-plus-u.md](dftb-plus-u.md), [dftb-vs-dftbplus-cleanup.md](dftb-vs-dftbplus-cleanup.md).
