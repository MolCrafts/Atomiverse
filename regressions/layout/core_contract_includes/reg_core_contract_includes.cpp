/// @file reg_core_contract_includes.cpp
/// @brief backend-mirror-01-core layout regression — the device-independent
///        contract headers resolve under `atomiverse/core/` (never `facade/`),
///        and an out-of-library consumer can include AND link against them.
///
/// This is a LAYOUT guarantee, not a physics value: it pins the public include
/// root, so it hard-codes "compiles + links + the contract symbols are usable"
/// rather than a cited literature number. It includes ONLY the two core
/// contract headers named by the spec.
#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"

#include <cstdio>
#include <cstdlib>

int main() {
    // slot_of is the one sanctioned Phase->slot conversion (core/stage.h),
    // slot_count the matching sentinel count — both usable from outside the lib.
    const int slot = ATV::slot_of(ATV::evaluate::Phase::evaluate);
    const int n = ATV::slot_count<ATV::evaluate::Phase>();

    // plan.h being on the include root is proven by the #include above: the
    // template ATV::Plan<Component> is a backend-parameterized type, so there is
    // nothing to instantiate here without a backend — resolution is the point.

    if (slot < 0 || slot >= n) {
        std::fprintf(stderr,
                     "core_contract_includes: slot %d out of range [0,%d)\n",
                     slot, n);
        return EXIT_FAILURE;
    }
    std::puts("core_contract_includes: core/{plan,stage}.h resolve and link");
    return EXIT_SUCCESS;
}
