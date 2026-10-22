# gpw/hirshfeld: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-hirshfeld-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-hirshfeld-uks
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-hirshfeld-kpoints
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-hirshfeld-uks-kpoints
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-hirshfeld-kpoint-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-hirshfeld-kpoint-stress
  REQUIRES GPW
  THREADS 8)
