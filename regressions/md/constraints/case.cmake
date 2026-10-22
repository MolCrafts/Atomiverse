# md/constraints: one case, one section per pinned system or quantity.

atv_regression(SECTION rigid_water
  REQUIRES MD)

atv_regression(SECTION settle_tip3p
  REQUIRES MD)

atv_regression(SECTION virtual-site-multi-topology
  REQUIRES MD)

atv_regression(SECTION four-site-tip4p-foothold
  REQUIRES MD)

atv_regression(SECTION tip3p_two_water
  REQUIRES POTENTIALS)
