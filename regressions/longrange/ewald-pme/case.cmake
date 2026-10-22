# longrange/ewald-pme: one case, one section per pinned system or quantity.

atv_regression(SECTION cuda-ewald-nacl
  REQUIRES MD POTENTIALS NEIGHBOR
  DEVICES cuda)

atv_regression(SECTION cuda-pme-nacl
  REQUIRES MD POTENTIALS NEIGHBOR KSPACE
  DEVICES cuda)

atv_regression(SECTION cuda-pme-half-spectrum
  REQUIRES MD POTENTIALS NEIGHBOR KSPACE EVALUATE
  DEVICES cuda)

atv_regression(SECTION slab-2d-ewald-foothold
  REQUIRES KSPACE MD)
