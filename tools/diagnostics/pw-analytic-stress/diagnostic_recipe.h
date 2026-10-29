#pragma once
#include "atomiverse/electronic/smearing.h"
namespace pw_analytic_diagnostic {
// Opt-in fixture parameters only; no solver or functional policy is introduced.
enum class Geometry { standard, stretched_h2_triplet, primitive_si2 };
struct Recipe {
    ATV::electronic::Smearing smearing = ATV::electronic::Smearing::fermi_dirac;
    double width_ry = .02;
    int kmesh = 2;
    bool isolated_h3 = true;
    // Trailing diagnostic-only fields preserve existing four-field recipes.
    Geometry geometry = Geometry::standard;
    bool rotate_z90 = false;
    bool empty_beta() const {
        return geometry == Geometry::stretched_h2_triplet;
    }
    bool mp() const {
        return smearing == ATV::electronic::Smearing::methfessel_paxton;
    }
    bool fixed() const { return smearing == ATV::electronic::Smearing::none; }
    bool gaussian() const {
        return smearing == ATV::electronic::Smearing::gaussian;
    }
    const char* pass_prefix() const {
        return mp()           ? "PW_MP"
               : empty_beta() ? "PW_EMPTY_BETA"
               : fixed()      ? "PW_FIXED_K"
               : gaussian()   ? "PW_GAUSSIAN"
                              : "PW_MULTI_K_FD";
    }
    const char* force_prefix() const {
        return mp()           ? "MP_FORCE"
               : empty_beta() ? "EMPTY_BETA_FORCE"
               : fixed()      ? "FIXED_K_FORCE"
               : gaussian()   ? "GAUSSIAN_FORCE"
                              : "MULTIK_FD_FORCE";
    }
    const char* stress_prefix() const {
        return mp()           ? "MP_STRESS"
               : empty_beta() ? "EMPTY_BETA_STRESS"
               : fixed()      ? "FIXED_K_STRESS"
               : gaussian()   ? "GAUSSIAN_STRESS"
                              : "MULTIK_FD_STRESS";
    }
};
} // namespace pw_analytic_diagnostic
