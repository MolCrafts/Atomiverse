/// @file reg_module_compile_defs.cpp
/// @brief Public compile-definition contract for full-classical product modules.
///
/// Built under default monorepo classical product gates ON. Fails to compile
/// if any required ATV_HAS_* macro is missing.

#ifndef ATV_HAS_MD
#error "ATV_HAS_MD required for full-classical regression"
#endif
#ifndef ATV_HAS_POTENTIALS
#error "ATV_HAS_POTENTIALS required for full-classical regression"
#endif
#ifndef ATV_HAS_NEIGHBOR
#error "ATV_HAS_NEIGHBOR required for full-classical regression"
#endif
#ifndef ATV_HAS_MINIMIZER
#error "ATV_HAS_MINIMIZER required for full-classical regression"
#endif
#ifndef ATV_HAS_MC
#error "ATV_HAS_MC required for full-classical regression"
#endif
#ifndef ATV_HAS_PATH
#error "ATV_HAS_PATH required for full-classical regression"
#endif
#ifndef ATV_HAS_VIB
#error "ATV_HAS_VIB required for full-classical regression"
#endif
#ifndef ATV_HAS_SEMIEMPIRICAL
#error "ATV_HAS_SEMIEMPIRICAL required for full-classical regression"
#endif
#ifndef ATV_HAS_EVALUATE
#error "ATV_HAS_EVALUATE required for full-classical regression"
#endif

int main() { return 0; }
