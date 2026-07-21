---
slug: hf-capability-gaps-20-fmm
criteria:
  - id: ac-001
    summary: longrange/ FMM 三头文件 backend/Component/Frame/MPI 无关,OOP 构造无工厂
    type: code
    pass_when: |
      include/atomiverse/longrange/{multipole_expansion,fmm_operators,fmm_tree}.h 存在,
      grep 无 "cpu/"、"cuda/"、"Frame"、"Component"、"mpi" include/符号,仅依赖
      core/(real_harmonics.h、vec3.h、precision.h);构造函数校验参数(order<0、
      max_per_leaf<1 抛异常),无 make_*/build_* 自由函数;公共符号有含单位说明
      (无量纲 q/r 核,eV 换算属链环 21)的 Doxygen。
    status: pending
  - id: ac-002
    summary: 展开单元测试通过(P2M/M2P/L2P 逐阶收敛 + 单极精确)
    type: runtime
    pass_when: |
      test_multipole_expansion 在 ctest 通过且秒级:单点电荷 P2M+M2P 误差随 p
      几何衰减;l=0 单极对 q/r 精确(1e-12);LocalExpansion L2P 收敛;
      非法 order 构造抛异常。
    status: pending
  - id: ac-003
    summary: 平移算子单元测试通过(M2M/L2L 阶内精确,M2L 几何衰减)
    type: runtime
    pass_when: |
      test_fmm_operators 在 ctest 通过:M2M 平移结果与对父中心直接 P2M 的多极
      系数/求值一致(<1e-12);L2L 同式(<1e-12);固定间隔比 c 下 M2L+L2P 误差
      逐阶几何衰减,衰减因子与 (1/c) 量级一致。
    status: pending
  - id: ac-004
    summary: FmmTree 单元测试通过(vs 直和、交互列表判据、退化例)
    type: runtime
    pass_when: |
      test_fmm_tree 在 ctest 通过且秒级(N≤512):随机点集 energy/potentials 与
      O(N²) 直和相对误差 <1e-6;交互列表每一对满足 ≥1 胞宽 well-separated 判据;
      叶容量 ≤ max_per_leaf;N=1、N=2、单叶退化例正确。
    status: pending
  - id: ac-005
    summary: regression:逐阶误差表几何衰减至文档 p 处 ≤1e-10(随机+晶格 vs 直和)
    type: scientific
    pass_when: |
      regressions/longrange/fmm_point_charges(库外消费者,regressions 目标,≤60 s)
      对随机与晶格点电荷集输出逐阶 p 误差表:相邻阶误差比 <1 且近似恒定(几何衰减),
      在 REFERENCE.md 文档标注的 p 处相对误差 ≤1e-10;REFERENCE.md 引
      DOI 10.1016/0021-9991(87)90140-9、DOI 10.1063/1.468354 及误差界公式。
    status: pending
  - id: ac-006
    summary: 标度烟检:N vs 4N 墙钟比宽松钉住近线性
    type: performance
    pass_when: |
      同一 regression 内 4N 与 N 两次求值(相同 order/叶容量)的墙钟比 < 8
      (明显低于 O(N²) 的 16),断言写入 regression 输出;总时长仍 ≤60 s;
      tests/ 内无任何计时断言。
    status: pending
  - id: ac-007
    summary: ch19「长程库仑 II:快速多极方法」节:叙事 + 误差表格 + molplot 标度曲线
    type: docs
    pass_when: |
      docs/tutorial/ch19-hf-large-systems.md 含「长程库仑 II:快速多极方法」一节:
      多极压缩远场与 M2M/M2L/L2L 游走叙事、与 PME 的开边界对偶说明、误差-阶数
      收敛表格(标量表格,无条形图)、molplot 标度曲线图(曲线允许)。
    status: pending
  - id: ac-008
    summary: 全套构建与测试门通过,tests/ 无新增 e2e
    type: runtime
    pass_when: |
      带 -DATV_BUILD_TESTS=ON 的完整构建通过;ctest 全绿且新增测试为秒级单测;
      no_factory_scrub 与 test_stage_typing_scrub 通过;src/longrange 为无条件
      编译(不挂任何 ATV_BUILD_* 门)。
    status: pending
---

# Acceptance criteria

数学正确性由阶内精确恒等式(M2M/L2L,ac-003)与逐阶几何衰减(ac-002/ac-003)钉住;
物理正确性由 O(N²) 直和 oracle 的 ≤1e-10 收敛表钉住(ac-005);近线性标度只在
regression 宽松钉住(ac-006);架构约束(facade 纯净、无工厂、无条件编译)由
ac-001/ac-008 钉住;教程为 ac-007。
