/// @file python/bind.cpp
/// @brief Top-level nanobind module definition for _atomiverse.
///
/// Atomiverse binds the driver / component API only — never a Frame data
/// type. Simulation state crosses the boundary as a `molrs.Frame` (from the
/// separate `molrs` extension); each `run()` borrows it via a PyCapsule.
///
/// Per-domain init functions are compiled only when the matching
/// ATV_BUILD_* product module is ON (see cmake/AtvProductModules.cmake).

#include <nanobind/nanobind.h>

namespace nb = nanobind;

#ifdef ATV_HAS_MD
void bind_md(nb::module_& m);
#endif
#ifdef ATV_HAS_MC
void bind_mc(nb::module_& m);
#endif
#ifdef ATV_HAS_MINIMIZER
void bind_minimizer(nb::module_& m);
#endif
#ifdef ATV_HAS_PW
void bind_pw(nb::module_& m);
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
void bind_semiempirical(nb::module_& m);
#endif
#ifdef ATV_HAS_HF
void bind_scf(nb::module_& m);
#endif

NB_MODULE(_atomiverse, m) {
    m.doc() = "Atomiverse: molecular simulation compute backend";

#ifdef ATV_HAS_MD
    bind_md(m);
#endif
#ifdef ATV_HAS_MC
    bind_mc(m);
#endif
#ifdef ATV_HAS_MINIMIZER
    bind_minimizer(m);
#endif
#ifdef ATV_HAS_PW
    bind_pw(m);
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
    bind_semiempirical(m);
#endif
#ifdef ATV_HAS_HF
    bind_scf(m);
#endif
}
