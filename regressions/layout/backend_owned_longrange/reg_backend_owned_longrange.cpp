/// @file reg_backend_owned_longrange.cpp
/// @brief backend-mirror-02-absorb layout regression — reciprocal-space
///        operator math is BACKEND-OWNED: an out-of-library consumer reaches it
///        only through the backend-owned `cpu/longrange/` header path, never a
///        top-level shared module (the old top-level include root is gone).
///
/// Layout guarantee, not a physics value: it hard-codes "the cpu/longrange
/// include path resolves and links", which is the contract phase 02 pins.
#include "atomiverse/cpu/longrange/ewald_sum.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {
    // recommend_alpha is a pure static on the absorbed Ewald operator — usable
    // from outside the library via the backend-owned header path only.
    const double alpha = ATV::longrange::EwaldSum::recommend_alpha(10.0, 1e-5);
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        std::fprintf(stderr, "backend_owned_longrange: bad alpha %g\n", alpha);
        return EXIT_FAILURE;
    }
    std::puts("backend_owned_longrange: cpu/longrange/ewald_sum.h resolves and "
              "links");
    return EXIT_SUCCESS;
}
