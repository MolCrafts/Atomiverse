/// @file cpu/scf/rohf.cpp
/// @brief ROHF constructor defaults + occupation resolution. The shared
///        driver loop lives in mean_field_method.cpp (explicitly instantiated
///        there); the Roothaan effective-Fock coupling lives in
///        stages/rohf_couple_stage.h.

#include "atomiverse/cpu/scf/rohf.h"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {

ROHF::ROHF() {
    spin_ = SpinType::RestrictedOpen;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.7;
    diis_max_vectors = 6;
    diis_start_iter = 5;
    tol = 1e-8;
}
ROHF::ROHF(int na, int nb) : ROHF() {
    n_alpha = na;
    n_beta = nb;
}
void ROHF::resolve_occupation(int) {
    if (n_alpha < n_beta)
        throw std::invalid_argument(
            "ROHF::run: n_alpha (" + std::to_string(n_alpha)
            + ") < n_beta (" + std::to_string(n_beta)
            + ") — ROHF defines the open shell as excess alpha electrons; "
              "swap the counts (no fallback).");
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
