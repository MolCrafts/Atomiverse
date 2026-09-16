/// @file cpu/planewave/internal/kpoint.cpp
/// @brief Monkhorst-Pack k-point mesh implementation.

#include "atomiverse/cpu/planewave/internal/kpoint.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

KPointGrid generate_mp_mesh(int nk1, int nk2, int nk3, AccReal shift1,
                            AccReal shift2, AccReal shift3) {
    KPointGrid kg;
    int nk = nk1 * nk2 * nk3;
    kg.kpoints.reserve(nk);
    kg.weights.resize(nk, AccReal(1) / AccReal(nk));

    AccReal dn1 = AccReal(1) / AccReal(nk1), dn2 = AccReal(1) / AccReal(nk2),
            dn3 = AccReal(1) / AccReal(nk3);
    for (int i = 0; i < nk1; ++i)
        for (int j = 0; j < nk2; ++j)
            for (int k = 0; k < nk3; ++k) {
                kg.kpoints.push_back({(AccReal(i) + shift1) * dn1,
                                      (AccReal(j) + shift2) * dn2,
                                      (AccReal(k) + shift3) * dn3});
            }
    return kg;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
