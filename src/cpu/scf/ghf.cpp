/// @file cpu/scf/ghf.cpp
/// @brief GHF constructor defaults + occupation resolution. The shared driver
///        loop lives in mean_field_method.cpp (explicitly instantiated there).

#include "atomiverse/cpu/scf/ghf.h"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {

GHF::GHF() {
    spin_ = SpinType::Generalized;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.7;
    diis_max_vectors = 6;
    diis_start_iter = 5;
    tol = 1e-8;
    // The spinor exchange K[P^ab] of the non-symmetric spin-off-diagonal
    // block needs the stored dense ERI tensor (the direct 8-fold-symmetry
    // builders assume symmetric densities).
    compute_eri = true;
    // No GHF gradient path in this link: keep nested force-slot runs from
    // silently requesting forces (forces() itself still throws).
    compute_force = false;
}
GHF::GHF(int n) : GHF() { n_electrons = n; }

void GHF::resolve_occupation(int) {
    if (n_electrons <= 0)
        throw std::invalid_argument(
            "GHF::run: n_electrons must be set (> 0) before run — got "
            + std::to_string(n_electrons)
            + ". The spinor aufbau needs the total electron count.");
    // The single spinor channel carries all N electrons; the β channel is
    // unused for the generalized form.
    n_alpha_ = n_electrons;
    n_beta_ = 0;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
