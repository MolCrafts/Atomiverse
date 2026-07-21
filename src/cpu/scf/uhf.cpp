/// @file cpu/scf/uhf.cpp
/// @brief UHF constructor defaults + occupation resolution. The shared driver
///        loop lives in mean_field_method.cpp (explicitly instantiated there).

#include "atomiverse/cpu/scf/uhf.h"

namespace ATV {
namespace cpu {
namespace scf {

UHF::UHF() {
    spin_ = SpinType::Unrestricted;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::Hcore;
    damping_factor = 0.7;
    diis_max_vectors = 6;
    diis_start_iter = 5;
    tol = 1e-8;
}
UHF::UHF(int na, int nb) : UHF() {
    n_alpha = na;
    n_beta = nb;
}
void UHF::resolve_occupation(int) {
    n_alpha_ = n_alpha;
    n_beta_ = n_beta;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
