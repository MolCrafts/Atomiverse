# TB / SE blocked capabilities — why they are not pending (2026-07-29)

Durable harness for rows that are **not ordinary pending**. A row leaves a hold
list only when its gate is *actually* removed.

## Permanent rules (never capabilities to implement)

| Id | Rule |
| -- | ---- |
| `mopac-as-dftb-oracle` | Forbidden: pin DFTB to OpenMOPAC |
| `gfn-ff` | Scope-out: classical GFN-FF → MD/FF surface |
| `om3-nddo` | No family-locked pin: OM3 ∉ OpenMOPAC primary surface. Expanding to Thiel OMx is an oracle-set decision, not a missing ATV feature gate. |

## Closed from blocked (2026-07-29)

| Slug | Resolution |
| ---- | ---------- |
| `se-analytic-hessian` | Shipped `reg_se-analytic-hessian` (VIB+DFTB2 H₂) |
| `dftb-plus-u` | Design note + energy-only Dudarev on `DFTB2::u_eff` + `reg_dftb-plus-u` |
| `dftb-gamma-units-fix` | Fixed γ units (was pending head, not blocked) |
| `gfn2-camm-r` | GFN2 model fixed term by term; `F_H1,y` re-pinned to tblite 0.4.0 at the same input (xtb 6.7.1 printed an orientation artifact). See its REFERENCE. |

## Still blocked (science multi-wave)

| Slug | Gate | Unblock |
| ---- | ---- | ------- |
| `noncollinear-spin-dftb` | Spinor Hamiltonian (2-component density), not a parameter on collinear | Spec + multi-wave after collinear solid |
| `spin-orbit-dftb` | SOC/SK design | After noncollinear |



Shell-resolved **collinear** spin (`ShellResolvedSpin = Yes`) and
shell-resolved \(U_\ell\) are ordinary pending, not these rows.

## Phase C dependents (pending, not blocked)

`stda-xtb` waits on AES-in-Fock / production GFN2 GS. Family Hessian
(`gfn-xtb-analytic-hessian`) **shipped**. Staff remaining Phase C under the
queue; do not re-list as blocked.

Related: [[dftb-plus-u]], [[dftb-gamma-units-defect]].
