# dynamics/xl-bomd: one case, one section per pinned system or quantity.

atv_regression(SECTION xl-bomd-water
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION xl-bomd-rhf-water
  REQUIRES HF MD)

atv_regression(SECTION xl-bomd-uks
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION xl-bomd-kpoints
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION xl-bomd-linearized-forces
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION xl-bomd-linearized-stress
  REQUIRES GPW MD
  THREADS 8)
