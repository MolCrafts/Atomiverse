# dynamics/second-gen-cpmd: one case, one section per pinned system or quantity.

atv_regression(SECTION second-gen-cpmd-water
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION second-gen-cpmd-uks
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION second-gen-cpmd-ot
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION second-gen-cpmd-uks-ot
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION second-gen-cpmd-kpoints
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION second-gen-cpmd-ot-kpoints
  REQUIRES GPW MD
  THREADS 8)
