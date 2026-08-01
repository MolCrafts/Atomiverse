# Acceptance — gpw-d3-atm-kpoint-forces

| Gate | Criterion |
| ---- | --------- |
| Oracle | Offline CP2K 2026.1 `cp2k.psmp` git:5e54ba2; dual ENERGY_FORCE twins C9 T/F + k-mesh; never live in ctest |
| E_disp | \|E_disp(C9) − pin\| ≤ 1e−9 Ha; two-body companion ≤ 1e−9 Ha |
| ATM force pin | max\|ΔF_ATV − ΔF_CP2K\| ≤ 1e−8 Ha/Bohr on 6 atoms; ATM signal ≥ 1e−7 |
| Product wiring | multi-k `gpw_forces_from_scf` d3_atm on−off equals standalone ATM ΔF ≤ 1e−7 |
| Multi-k payload | `n_kpts ≥ 1`, `C_occ_k` present after SCF with `k_mesh={1,1,2}` |
| Unit dual | analytic multi-k product ATM fold vs geometry-only identity (wiring / FD unit) |
| Wall budget | regression ≤ 60 s |
| Fair perf | \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) ≤ 10 on matched water-dimer multi-k C9 ENERGY_FORCE twin (1 MPI × 8 OMP); report product multi-k SCF+force wall (not H₂ wiring / multi-gate reg wall) |
| Public API | Frame I/O; ctor / public members only; no `*Config` / `make_*` |

`acceptance_met = true` only when dual gate green and fair ratio recorded.
