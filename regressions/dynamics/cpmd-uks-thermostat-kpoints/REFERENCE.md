# Reference — multi-k UKS Car–Parrinello MD with electronic Nosé–Hoover (H₂ NVE, GPW)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on **multi-k collinear UKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces, with the **Nosé–Hoover electronic thermostat
engaged on every \((k, \sigma)\) channel** — the `cpmd-uks-kpoints` ×
`cpmd-thermostat-kpoints` combo pin:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (nuclei NVE; only the
  fictitious electronic DOF are thermostatted)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 1`**, **`n_beta = 1`** (closed-shell fillings on the UKS
  path — both spin channels propagate and H₂ stays bound; triplet H₂ is
  repulsive and cannot pin NVE — the `xl-bomd-uks-kpoints` /
  `cpmd-uks-kpoints` precedent), **`k_mesh = {1,1,2}`**,
  **`k_time_reversal = false`** (two sampled k), wrapped by
  `cpu::electronic::CarParrinello{gpw, μ, ElectronicThermostat{K₀, τ_e}}`;
  the electronic dt is pushed by the MD integrator (`TimestepProvider` /
  `TimestepConsumer`), never set here
- One `cpu::electronic::CpmdCplx` per **(k, spin)**, **each carrying its
  own NH variable targeting the full \(K_0\)** (the Γ-UKS / multi-k
  thermostat convention); production steps are dual-spin fixed-density
  multi-k KS evals (`iters == 0`, `H_ks_k` + `H_ks_k_beta`), spin-diagonal
  EOM occupancy 1
- Published totals are k-weighted dual-spin sums:
  \(T_e = \sum_k w_k \sum_\sigma T_e^{(k,\sigma)}\),
  \(E_{\mathrm{therm}} = \sum_k w_k \sum_\sigma E_{\mathrm{therm}}^{(k,\sigma)}\)
- Conserved diagnostic (Marx & Hutter extended energy):
  \(E_{\mathrm{ext}} = T_n + T_e + E_{\mathrm{KS}} + E_{\mathrm{therm}}\)

**Not BOMD** (production `iters == 0`). **Not nuclear NVT.**
**Not XL-BOMD / SGCP.** **Not the NVE-only multi-k UKS pin**
(`cpmd-uks-kpoints`) or the RKS multi-k thermostat pin
(`cpmd-thermostat-kpoints`) alone.

## Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Blöchl & Parrinello, *Phys. Rev. B* **45**, 9413 (1992) — electronic
  Nosé–Hoover for adiabaticity control
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — electronic RATTLE
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — extended conserved
  energy with electronic NH
- Twin family: Γ thermostat product `cpmd-electronic-thermostat`; multi-k
  RKS thermostat product `cpmd-thermostat-kpoints`; multi-k UKS NVE
  product `cpmd-uks-kpoints`
- Multi-k UKS BO surface pins: `gpw-kpoints-uks`, `gpw-kpoint-uks-forces`,
  `gpw-bomd-uks-kpoints` (CP2K-anchored)

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | collinear UKS, \(n_\alpha = n_\beta = 1\) (closed-shell fillings) |
| XC | PADE (LSD) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF seed | `tol = 1e-5`, `max_iter = 80`, DIIS |
| CPMD | \(\mu = 400\) a.u.; production fixed-density dual-spin |
| Thermostat | \(K_0 = 10^{-5}\) Ha, \(\tau_e = 50\) fs, per (k, σ) channel |
| Ensemble | NVE nuclei; electronic NH only |
| `dt` | 0.1 fs (nuclear = electronic) |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 cold SCF seed + 1 CP baseline + 4 production |

## Offline oracle (CP2K)

**Family lock (same policy as `cpmd-electronic-thermostat` /
`cpmd-thermostat-kpoints` / `cpmd-uks-kpoints`):** CP2K 2026.1 builds and
runs on this host, but has **no fictitious-mass Car–Parrinello
propagator** — its `RUN_TYPE MD` is Born–Oppenheimer and its CP-like method
is second-generation CPMD (ASPC + corrector). The CP extended conserved
quantity (and its electronic NH bookkeeping term) is not a number CP2K can
print for any input.

The **BO surface underneath is CP2K-anchored** by the shipped multi-k UKS
pins (`gpw-kpoints-uks` energies, `gpw-kpoint-uks-forces` forces,
`gpw-bomd-uks-kpoints` AIMD), and this regression additionally asserts the
closed-shell identity against the multi-k **RKS** path that
`gpw-bomd-kpoints` pinned against CP2K 2026.1 directly.

An **unexecutable sketch** of the intended twin lives in
`tools/cp2k_oracle/inputs/cpmd-uks-thermostat-kpoints/h2_cpmd_uks_thermostat_kpoints_nve.inp`.
The pin is the **literature extended-energy / \(T_e\) control envelope**
below, never a golden snapshot of Atomiverse output.

## Observable and tolerance

Let
\(E_{\mathrm{ext}}(t) = E_{\mathrm{KS}}(t) + T_n(t) + T_e(t) + E_{\mathrm{therm}}(t)\)
[eV], with \(E_0\) after the first CP production step (thermostat engaged).

\[
\delta = \max_t \frac{|E_{\mathrm{ext}}(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- **UKS-vs-RKS closed-shell consistency:** dedicated multi-k UKS
  (`n_alpha = n_beta = 1`) and multi-k RKS (`n_occ = 1`) single points
  (`cpu::single_point`, plain `GPWEvaluator`, same geometry/mesh/cutoff,
  `tol = 1e-7`) agree within **3×10⁻⁴ eV** ≈ 1.1×10⁻⁵ Ha
- Production SCF **iters == 0** (dual-spin fixed-density multi-k eval)
- `n_kpts >= 2` after seed
- \(T_e\) ceiling over production ≤ **0.05 eV** (thermostatted control —
  the fictitious electrons must not heat)
- \(E_{\mathrm{therm}}\) finite on every production step
- PE finite and bound (\(< -5\) eV after warm-up); KE finite, \(\ge 0\),
  non-trivial after warm-up
- All coordinates finite
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | Oracle family lock: CP2K 2026.1 has no fictitious-mass CP propagator — no fair wall twin of thermostatted multi-k UKS CP AIMD exists |
| `ratio_atv_over_oracle` | n/a |
