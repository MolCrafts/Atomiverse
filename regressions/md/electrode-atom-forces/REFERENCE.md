# electrode-atom-forces

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** (re-checked 2026-07-29)
oracle_kind: real binary for Q; forces via MW `forces.out` on mobile ion
(not electrode_forces keyword — unsupported)

Twin: `tools/metalwalls_oracle/inputs/metalwalls/electrode-atom-forces/mw_twin/`
— Q_E1 = −5.048583047766454e-1 e, Q_E2 = −4.951416952233546e-1 e;
ion Fz = 1.769232616266544e-2 Ha/a0 ≈ 0.910 eV/Å.

ATV foothold: non-zero electrode Q and electrode-atom forces under
`write_electrode_forces` (truncated image sum; absolute F not tightly MW-matched).
