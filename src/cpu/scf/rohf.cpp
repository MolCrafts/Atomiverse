/// @file cpu/scf/rohf.cpp
/// @brief ROHF constructor defaults and occupation resolution. The nuclear
///        Hessian is `Hessian<ROHF>` (FD of analytic forces; PySCF has no
///        `hessian.ROHF`). The shared driver loop lives in
///        mean_field_method.cpp.

#include "atomiverse/cpu/scf/rohf.h"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {

ROHF::ROHF(BasisSet basis_in, int na, int nb, int charge_in)
    : MeanFieldMethod<ROHF>(basis_in, charge_in), n_alpha(na), n_beta(nb) {
    spin_ = SpinType::RestrictedOpen;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.7;
    diis_max_vectors = 6;
    diis_start_iter = 5;
    tol = 1e-8;
}
void ROHF::resolve_occupation(int) {
    if (n_alpha < n_beta)
        throw std::invalid_argument(
            "ROHF::run: n_alpha (" + std::to_string(n_alpha) + ") < n_beta ("
            + std::to_string(n_beta)
            + ") — ROHF defines the open shell as excess alpha electrons; "
              "swap the counts (no fallback).");
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
