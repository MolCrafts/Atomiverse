# response/phonons: one case, one section per pinned system or quantity.

atv_regression(SECTION phonon-dispersion-si
  REQUIRES VIB MD)

atv_regression(SECTION fd-phonon-supercell
  REQUIRES VIB MD)

atv_regression(SECTION gpw-fd-phonon-si
  REQUIRES GPW VIB
  THREADS 8)
