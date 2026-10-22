# dynamics/car-parrinello: one case, one section per pinned system or quantity.

atv_regression(SECTION cpmd-water
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION cpmd-rhf-h2
  REQUIRES HF MD)

atv_regression(SECTION cpmd-rks-water
  REQUIRES HF MD)

atv_regression(SECTION cpmd-uhf-o2
  REQUIRES HF MD)

atv_regression(SECTION cpmd-uks
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION cpmd-kpoints
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION cpmd-uks-kpoints
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION cpmd-thermostat-kpoints
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION cpmd-electronic-thermostat
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION cpmd-rattle-water
  REQUIRES GPW MD
  THREADS 8)
