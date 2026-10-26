# dynamics/bomd: one case, one section per pinned system or quantity.

atv_regression(SECTION h2_rhf_sto3g_aimd
  REQUIRES HF MD
  DEVICES both)

atv_regression(SECTION bomd-d3bj-ar2-nve
  REQUIRES HF MD)

atv_regression(SECTION gpw-bomd-water
  REQUIRES GPW MD
  THREADS 8)

atv_regression(SECTION pw-bomd-h
  REQUIRES PW MD)
