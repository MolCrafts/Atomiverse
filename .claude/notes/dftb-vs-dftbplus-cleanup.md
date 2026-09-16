# DFTB 对照 DFTB+ 清理台账（2026-07-29 wrap-up）

**Family oracle（绑定）：所有 DFTB 族数值以 DFTB+ 为准**。

**Wave wrap-up status:** product-path DFTB fidelity on mio/3ob closed for
H₂ / H₂O / CH₄; Elstner γ; multi-k open shell; TD oscillator strengths.
No live TB specs. Phase A queue row `dftb3-periodic-forces` → **done**.

---

## Closed this wave

| 事项 | 状态 |
| ---- | ---- |
| Full H/C/N/O mio-1-1 static SK | **done** — `tools/gen_dftb_mio_tables.py` → `dftb_mio_sk_tables.h` |
| Hetero SKF poly-header parse | **done** |
| DFTB+ Elstner γ (not Ohno–Klopman) | **done** — `internal/dftb_gamma.h` |
| H₂O DFTB2 vs DFTB+ 24.1 | **done** — \|ΔE\| ≈ 0.03 meV |
| CH₄ 3ob vs DFTB+ | **done** — \|ΔE\| ≈ 0.4 µeV |
| DFTB3 equal-U ∂γ/∂U force bug | **done** — ½ ∂γ_same/∂U |
| DFTB3 Γ-PBC analytic F | **done** — `reg_dftb3-periodic-forces` |
| multi-k collinear open shell | **done** — dual α/β H(k) + dual-channel F |
| TD-DFTB oscillator strengths | **done** — length-form \(f\); triplets dark |
| Doc scrub Ohno → Elstner | **done** |

## Still open (not this wave’s active pending)

| 项 | 台账位置 |
| -- | -------- |
| ThreeOb multi-k | roadmap deepen only（无 `[ ]` slug） |
| TD excited-state gradients / LC-TD | roadmap deepen |
| DFTB3 k-mesh / open shell / stress / D3 | roadmap deepen |
| noncollinear / SOC | **blocked** — `tb-blocked-capabilities.md` |
| Phase B NDDO / Phase C GFN | queue pending（不同 oracle 族） |

## Pin host

| 项 | 路径 |
| -- | ---- |
| DFTB+ 24.1 | `…/software/dftbplus-24.1/bin/dftb+` |
| mio / 3ob | `…/software/dftb-params/{mio-1-1,3ob-3-1}` |
| Env | `source tools/tb_oracle/setup_dftbplus_env.sh` |

Related: [[dftb-gamma-units-defect]]、[[dftb-plus-u]]、`docs/developer/tb-method-queue.md`。
