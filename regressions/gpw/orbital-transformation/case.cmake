# gpw/orbital-transformation: one case, one section per pinned system or quantity.

atv_regression(SECTION ot-minimizer-water
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION ot-uks-o2
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION ot-dftu
  REQUIRES GPW
  THREADS 8)
