# gpw-xdm-kpoint-stress

## Goal

Dedicated dual-gate pin for **multi-k GPW analytic cell stress** with
**free-atom XDM** (PBE, Becke–Johnson combination rules, QE NC-PBE damping).

The k-summed fixed-\(P_k\) Lagrangian already rebuilds geometry-only
\(E_{\mathrm{disp}}^{\mathrm{XDM}}\) at every strained cell
(`energy_fixed_P_k` / `gpw_stress` multi-k branch). This case is the
**write-first dedicated pin** (queue: deps `gpw-xdm-stress` ✓).

## Product surface

- `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `xdm = true`,
  `compute_stress = true`, `stress_use_fd = false`
- Frame `kStress` [eV/Å³]; internal `last_stress_ha_bohr3()` [Ha/Bohr³]
- Geometry-only kernel: `cpu::gpw::xdm_pbe` (rcut 50 Å)

## Oracle

| Role | Source |
| ---- | ------ |
| \(E_{\mathrm{disp}}\) + strain-FD \(\sigma^{\mathrm{XDM}}\) | offline dual pure-Python free-atom XDM lattice sum (`tools/cp2k_oracle/inputs/gpw-xdm-kpoint-stress/pin_xdm_strain.py`) — geometry-only, **bit-identical** to the `gpw-xdm-stress` pin set |
| \(\sigma\) dual gate | multi-SCF strain-FD on the same 1×1×2 mesh (`gpw_stress_fd_impl`) |

**No CP2K twin:** stock CP2K 2026.1 `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no `TYPE XDM`. Literature/pure-Python escape
(`oracle: literature`); **perf_exempt** (no fair wall twin). Never fake
`oracle: cp2k`.

## Acceptance

See `acceptance.md`.
