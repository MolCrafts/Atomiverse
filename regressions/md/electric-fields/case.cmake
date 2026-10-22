# md/electric-fields: one case, one section per pinned system or quantity.

atv_regression(SECTION external-efield-md
  REQUIRES MD)

atv_regression(SECTION oscillating-efield-md
  REQUIRES MD)

atv_regression(SECTION finite-field-single-electrode
  REQUIRES MD)

atv_regression(SECTION displacement-dfield-md
  REQUIRES KSPACE MD)

atv_regression(SECTION constant-d-polarization-feedback
  REQUIRES KSPACE MD)
