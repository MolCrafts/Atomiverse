# Specs index

- [scf-driver-refactor-06-criterion-unify](scf-driver-refactor-06-criterion-unify.md) — 终止即组件：三内置判据迁 Criterion、删 LoopHandler/add_handler、统一 verdict 轮询 [approved]


Completed work lives in the git log (campaigns: frame-block, cpu-breadth,
pw-breadth, se-breadth / RM1, python-on-demand-install).

- [hf-capability-gaps-01-env](hf-capability-gaps-01-env.md) — PySCF 对照环境 + oracle 协议 + gen_basis_data.py 恢复 + 教程弧骨架(ch18-20/附录) [approved]
- [hf-capability-gaps-02-cart2sph](hf-capability-gaps-02-cart2sph.md) — 球谐 AO(5d/7f):real_harmonics 上提 core/ + 壳层块 c2s + 逐基组 cart/sph 约定门 [approved]
- [hf-capability-gaps-03-multipole](hf-capability-gaps-03-multipole.md) — 多极一电子积分内核 + 偶极/四极观测器(consume SlaterDeterminant/expect) [approved]
- [hf-capability-gaps-04-charges](hf-capability-gaps-04-charges.md) — Löwdin/ESP(MK+CHELPG)电荷 + Koopmans/ΔSCF 工作流 [approved]
- [hf-capability-gaps-05-basis](hf-capability-gaps-05-basis.md) — 基组扩容:第三周期 + 6-31+G* + aug-cc-pVDZ 静态表 [approved]
- [hf-capability-gaps-06-ecp](hf-capability-gaps-06-ecp.md) — def2-ECP 静态表 + McMurchie-Davidson type-1/2 积分 + ECP↔x2c 互斥门 [approved]
- [hf-capability-gaps-07-guess](hf-capability-gaps-07-guess.md) — Hückel/SAP 初猜 + EDIIS/ADIIS 远域混合器(→CDIIS 升级切换) [approved]
- [hf-capability-gaps-08-stability](hf-capability-gaps-08-stability.md) — 轨道 Hessian A/B 单一内核(σ-vector)+ Seeger-Pople 稳定性分析 [approved]
- [hf-capability-gaps-09-soscf](hf-capability-gaps-09-soscf.md) — SOSCF/CIAH 二阶收敛器(复用 08 内核,信赖域 + exp(κ)) [approved]
- [hf-capability-gaps-10-rohf](hf-capability-gaps-10-rohf.md) — ROHF 具体方法(PySCF/Guest-Saunders 正则化约定绑定) [approved]
- [hf-capability-gaps-11-ghf](hf-capability-gaps-11-ghf.md) — GHF 实二分量旋量(SlaterDeterminant 旋量推广,与 slater-determinant 链协调) [approved]
- [hf-capability-gaps-12-df](hf-capability-gaps-12-df.md) — RI-J/RI-JK 密度拟合:def2-JKFIT 表 + l=4 扩展 + 成员驱动 J/K 接缝 [approved]
- [hf-capability-gaps-13-cphf](hf-capability-gaps-13-cphf.md) — CPHF 线性响应 + 解析 Hessian(08 内核第三配置,喂 vib::HessianResult) [approved]
- [hf-capability-gaps-14-tdhf](hf-capability-gaps-14-tdhf.md) — TDHF/RPA + TDA 激发态(08 内核第四配置,稳定性前置检查) [approved]
- [hf-capability-gaps-15-embed](hf-capability-gaps-15-embed.md) — 点电荷嵌入 + 均匀外场(有限场偶极 == 解析偶极交叉验证) [approved]
- [hf-capability-gaps-16-solvent](hf-capability-gaps-16-solvent.md) — C-PCM 隐式溶剂(SCF 内自洽反应场;ddCOSMO 出局为后续路线) [approved]
- [hf-capability-gaps-17-bsse](hf-capability-gaps-17-bsse.md) — ghost 基组 + Boys-Bernardi counterpoise 5-SCF 编排 [approved]
- [hf-capability-gaps-18-x2c](hf-capability-gaps-18-x2c.md) — 标量相对论 x2c-1e(pVp 积分 + 一步退耦;picture-change 警告) [approved]
- [hf-capability-gaps-19-longrange](hf-capability-gaps-19-longrange.md) — 行为保持重构:顶层 longrange/ facade + cpu/kspace→cpu/longrange 更名 [approved]
- [hf-capability-gaps-20-fmm](hf-capability-gaps-20-fmm.md) — Greengard-Rokhlin FMM 算子核(点电荷开边界,M2M/M2L/L2L + 八叉树) [approved]
- [hf-capability-gaps-21-cfmm](hf-capability-gaps-21-cfmm.md) — CFMM 高斯分布 J-builder 接入 HF(硬门:逐元素复现 direct J 先于速度论断) [approved]

