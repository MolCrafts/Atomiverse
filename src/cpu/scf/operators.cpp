/// @file cpu/scf/operators.cpp
/// @brief Stored-ERI two-electron operator adapter
///        (internal::IntegralView::coulomb_exchange_fused) over ExactJK.

#include "atomiverse/cpu/scf/internal/exact_jk.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"

#include <xtensor-blas/xlinalg.hpp>

namespace ATV {
namespace cpu {
namespace scf {

// Dense-input adapter for internal post-HF/operator consumers. All exact
// contractions use the same s8 engine; SCF retains its engine across iterations.

void internal::IntegralView::coulomb_exchange_fused(const RealMatrix& P_total,
                                                    RealMatrix& G) const {
    internal::ExactJK engine;
    engine.prepare(*this);
    RealMatrix J, K;
    engine.compute(P_total, J, K);
    G = J - double(0.5) * K;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
