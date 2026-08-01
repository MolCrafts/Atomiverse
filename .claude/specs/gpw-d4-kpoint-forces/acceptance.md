# Acceptance — `gpw-d4-kpoint-forces`

| # | Gate | Tol / criterion |
| - | ---- | --------------- |
| 1 | \|E_disp − dftd4 3.7.0 pin\| | ≤ 1×10⁻⁸ Ha |
| 2 | max\|F_d4 + g_dftd4\| (standalone) | ≤ 1×10⁻⁷ Ha/Bohr |
| 3 | Multi-k SCF converges; `C_occ_k` present | bool |
| 4 | Product `e_disp` equals dftd4 pin | ≤ 1×10⁻⁸ Ha |
| 5 | max\|(F_d4 − F_plain) − F_standalone\| multi-k wiring | ≤ 1×10⁻⁷ Ha/Bohr |
| 6 | max\|ΔF_CP2K − F_standalone\| (FORCE D4−plain) | ≤ 5×10⁻⁷ Ha/Bohr (cutoff parity) |
| 7 | Product wall | ≤ 60 s |
| 8 | Fair wall ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10; real CP2K run |

`acceptance_met=true` only when dual gate + fair wall green and
`REFERENCE.md` records binary+version+input+output line.
