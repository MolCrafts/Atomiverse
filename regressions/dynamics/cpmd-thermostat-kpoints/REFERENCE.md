# Reference — multi-k Car–Parrinello MD with electronic Nosé–Hoover thermostat (H₂ NVE, GPW)

## Method

**Car–Parrinello molecular dynamics (CPMD)** on **multi-k closed-shell RKS**
Gaussian-and-plane-waves (GPW / Quickstep) forces (Bloch \(C_k\)), with the
**Nosé–Hoover electronic thermostat engaged on every k channel** — the
capability this case pins is the lift of the old build-time refusal
"NVE electrons only on the k path" (`cpmd-thermostat-kpoints`):

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (nuclei NVE; only the
  fictitious electronic DOF are thermostatted)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  wrapped by `cpu::electronic::CarParrinello{gpw, μ,
  ElectronicThermostat{K₀, τ_e}}`; the electronic dt is pushed by the MD
  integrator (`TimestepProvider` / `TimestepConsumer`), never set here
- One `cpu::electronic::CpmdCplx` per sampled k, **each carrying its own NH
  variable targeting the full \(K_0\)** (the Γ-UKS convention); production
  steps are fixed-density multi-k KS evals (`iters == 0`)
- Published totals are k-weighted:
  \(T_e = \sum_k w_k T_e^{(k)}\),
  \(E_{\mathrm{therm}} = \sum_k w_k E_{\mathrm{therm}}^{(k)}\)
- Conserved diagnostic (Marx & Hutter extended energy):
  \(E_{\mathrm{ext}} = T_n + T_e + E_{\mathrm{KS}} + E_{\mathrm{therm}}\)

**Not BOMD** (production `iters == 0`). **Not nuclear NVT.**
**Not XL-BOMD / SGCP.**

## Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Blöchl & Parrinello, *Phys. Rev. B* **45**, 9413 (1992) — electronic
  Nosé–Hoover for adiabaticity control
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — extended conserved
  energy with electronic NH
- Twin family: Γ thermostat product `cpmd-electronic-thermostat`; multi-k
  NVE products `cpmd-kpoints` / `cpmd-uks-kpoints`
- Γ-limit **solver identity** (thermostatted `CpmdCplx` ≡ real `Cpmd` on
  real inputs, 1e-12) is unit-pinned in
  `tests/cpu/electronic/test_cpmd.cpp` (`ThermostatGammaLimitMatchesRealCpmd`)

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | closed-shell RKS, \(n_\mathrm{occ}=1\) |
| XC / basis / PP | PADE / DZV-GTH-PADE / GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF seed | `tol = 1e-5`, `max_iter = 60`, DIIS |
| CPMD | \(\mu = 400\) a.u.; production fixed-density |
| Thermostat | \(K_0 = 10^{-5}\) Ha, \(\tau_e = 50\) fs, per k channel |
| `dt` | 0.1 fs (nuclear = electronic) |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 cold SCF seed + 1 CP baseline + 4 production |

## Offline oracle (CP2K)

**Family lock (2026-07-31, same policy as `cpmd-electronic-thermostat` /
`cpmd-kpoints` / `cpmd-uks-kpoints`):** CP2K 2026.1 is present on the worker
host but has **no fictitious-mass Car–Parrinello propagator** — the CP
extended conserved quantity (and its electronic NH bookkeeping term) is not a
number CP2K can print, so no live or offline CP2K twin of this observable
exists. The BO surface underneath is CP2K-anchored by `gpw-kpoints-si` /
`gpw-kpoint-forces` / `gpw-bomd-kpoints`. An **unexecutable sketch** of the
intended twin lives in
`tools/cp2k_oracle/inputs/cpmd-thermostat-kpoints/h2_cpmd_thermostat_kpoints_nve.inp`.
The pin is therefore the **literature extended-energy / \(T_e\) control
envelope** below, never a golden snapshot of Atomiverse output.

## Observable and tolerance

Let \(E_{\mathrm{ext}}(t) = E_{\mathrm{KS}} + T_n + T_e + E_{\mathrm{therm}}\)
[eV], with \(E_0\) after the first CP production step (thermostat engaged).

\[
\delta = \max_t \frac{|E_{\mathrm{ext}}(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Production SCF **iters == 0** (fixed-density multi-k eval)
- `n_kpts >= 2` after seed
- \(T_e\) ceiling over production ≤ **0.05 eV** (thermostatted control —
  the fictitious electrons must not heat)
- \(E_{\mathrm{therm}}\) finite on every production step
- PE finite and bound (\(< -5\) eV after warm-up); KE finite, \(\ge 0\)
- All coordinates finite; soft H–H window \(0.4 < r_{\mathrm{HH}} < 1.5\) Å

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | Oracle family lock: CP2K 2026.1 has no fictitious-mass CP propagator — no fair wall twin of thermostatted multi-k CP AIMD exists (2026-07-31 inventory) |
| `ratio_atv_over_cp2k` | n/a |
