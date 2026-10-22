# gpw/bands-dos: one case, one section per pinned system or quantity.

atv_regression(SECTION gpw-bands-si
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-bands-metallic
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-bands-uks
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-dos-si
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-dos-metallic
  REQUIRES GPW
  THREADS 8)

atv_regression(SECTION gpw-dos-uks
  REQUIRES GPW
  THREADS 8)
