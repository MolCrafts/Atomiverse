# forcefield/gaff: one case, one section per pinned system or quantity.

atv_regression(SECTION gaff_peo_fragment
  REQUIRES POTENTIALS)

atv_regression(SECTION gaff-fourier-cvff-lammps
  REQUIRES POTENTIALS)

atv_regression(SECTION cuda-gaff-fragment
  REQUIRES MD POTENTIALS NEIGHBOR KSPACE EVALUATE
  DEVICES cuda)

atv_regression(SECTION dihedral_charmm
  REQUIRES MD)

atv_regression(SECTION improper_harmonic
  REQUIRES MD)
