/// @file cpu/scf/roks.cpp
/// @brief ROKS constructor defaults + occupation resolution. The shared
///        driver loop lives in mean_field_method.cpp (explicitly instantiated
///        there); the Roothaan effective-Fock coupling is the same
///        stages/rohf_couple_stage.h used by ROHF, driven by
///        SpinType::RestrictedOpen + MeanFieldType::KS.

#include "atomiverse/cpu/scf/roks.h"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {

ROKS::ROKS() {
    spin_ = SpinType::RestrictedOpen;
    method_ = MeanFieldType::KS;
    xc_type_ = XCType::LSDA;
    xc = XCType::LSDA;
    init_guess = InitGuess::MINAO;
    damping_factor = 0.5;
    grid_level = GridLevel::MatchPySCF;
    diis_max_vectors = 6;
    diis_start_iter = 1;
    tol = 1e-8;
}
ROKS::ROKS(int na, int nb, XCType xc_in) : ROKS() {
    n_alpha = na;
    n_beta = nb;
    xc = xc_in;
    xc_type_ = xc_in;
}
void ROKS::resolve_occupation(int) {
    if (n_alpha < n_beta)
        throw std::invalid_argument(
            "ROKS::run: n_alpha (" + std::to_string(n_alpha) + ") < n_beta ("
            + std::to_string(n_beta)
            + ") — ROKS defines the open shell as excess alpha electrons; "
              "swap the counts (no fallback).");
    xc_type_ = xc;
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
