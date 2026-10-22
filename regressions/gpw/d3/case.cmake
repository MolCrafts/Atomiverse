# gpw/d3: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-d3-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d3-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d3-atm
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d3-atm-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d3-atm-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d3bj-forces
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-d3bj-stress
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-pbe-d3bj-ice
  REQUIRES GPW
  THREADS 8)
