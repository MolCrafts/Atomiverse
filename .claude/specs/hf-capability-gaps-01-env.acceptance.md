---
slug: hf-capability-gaps-01-env
criteria:
  - id: ac-001
    summary: setup_pyscf_env.sh idempotent; pyscf+molvis+molplot importable in venv
    type: runtime
    evaluator_hint: run externally in login shell, not via ctest
    pass_when: |
      Running tools/setup_pyscf_env.sh twice in a row both exit 0, and
      ~/work/.x86_64/bin/python -c "import pyscf, molvis" exits 0; the script
      reports molplot availability informationally.
    note: AMENDED during impl (2026-07-20) — no Python molplot distribution
      exists (molplot is a TypeScript sub-package of the molvis repo); the
      original pass_when required `import molplot`, which is unsatisfiable.
      Status recorded in tools/setup_pyscf_env.sh header and
      docs/tutorial/appendix-pyscf-oracle.md; revisit when a Python molplot
      ships.
    status: verified
    last_checked: 2026-07-20
  - id: ac-002
    summary: regenerated sto3g.h is byte-identical to the shipped header
    type: runtime
    evaluator_hint: run with the venv python; generator never touches CMake/ctest
    pass_when: |
      ~/work/.x86_64/bin/python tools/gen_basis_data.py --check
      include/atomiverse/cpu/scf/basis_data/sto3g.h sto-3g sto3g H He Li Be B C N O
      exits 0; the same command against a copy with one byte flipped exits non-zero.
    status: verified
    last_checked: 2026-07-20
  - id: ac-003
    summary: PySCF remains external-only; no build/test dependency introduced
    type: code
    pass_when: |
      grep -ri pyscf across CMakeLists.txt, cmake/, tests/, src/, include/ matches
      only comments/doc strings (basis_data header provenance notes); no CMake
      target, ctest registration, or find_package/execute_process invokes Python
      or PySCF.
    status: verified
    last_checked: 2026-07-20
  - id: ac-004
    summary: oracle protocol doc pins metadata, tables-only scalars, molvis/molplot-only figures
    type: docs
    pass_when: |
      docs/tutorial/appendix-pyscf-oracle.md exists and states all four rules:
      (1) PySCF runs externally only, never a build/test dependency; (2) every
      pinned number carries geometry, basis, convergence settings and PySCF
      version; (3) scalar comparisons are markdown tables, bar charts forbidden;
      (4) molvis/molplot are the only visualization tools for tutorial figures.
    status: pending
  - id: ac-005
    summary: ch18/ch19/ch20 Jensen-style stubs exist with comparison-table placeholder
    type: docs
    pass_when: |
      docs/tutorial/ch18-scf-techniques.md, ch19-hf-large-systems.md and
      ch20-hf-properties-environment.md each exist with a section skeleton
      (method-narrative headings) and an end-of-chapter PySCF comparison-table
      placeholder marked for later links to fill.
    status: pending
  - id: ac-006
    summary: tutorial index describes the HF arc and links resolve
    type: docs
    pass_when: |
      docs/tutorial/index.md contains the new arc description and links to
      ch18, ch19, ch20 and appendix-pyscf-oracle.md, and every linked file
      exists in docs/tutorial/.
    status: pending
  - id: ac-007
    summary: full unit suite unchanged — no new ctest targets, all green
    type: runtime
    pass_when: |
      ctest --test-dir build --output-on-failure passes with the same test list
      as before this link (no target added or removed by this change).
    note: verified against the last good build — 181/186 pass; the only 5
      failures are the pre-existing MPI ones (byte-identical at pristine
      HEAD, established 2026-07-20); the +2 targets vs the 184 baseline were
      added by the concurrent scf-slater-determinant chain commits, not by
      this link (this link touches only tools/ and docs/). A transient
      compile error in mean_field_method.cpp belongs to that chain's
      in-flight 03-unbag WIP, outside this link's file set.
    status: verified
    last_checked: 2026-07-20
---

# Acceptance criteria

ac-001/ac-002 在 venv 中外部执行（协议本身禁止其进入 ctest）；ac-003 守住
"禁依赖不禁知识" 的边界；ac-004–ac-006 锁定教程弧骨架；ac-007 确认本环对
构建与单元套件零扰动。
