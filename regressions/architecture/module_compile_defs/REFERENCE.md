# module_compile_defs

Literature: N/A (build/packaging contract).

Expected: under default monorepo configure with classical product modules ON,
`reg_module_compile_defs` compiles and exits 0. The TU fails to compile if any
of `ATV_HAS_{MD,POTENTIALS,NEIGHBOR,MINIMIZER,MC,PATH,VIB,SEMIEMPIRICAL,EVALUATE}`
is missing.
