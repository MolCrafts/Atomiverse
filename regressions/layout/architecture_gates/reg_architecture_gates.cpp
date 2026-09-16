/// @file reg_architecture_gates.cpp
/// @brief backend-mirror-06-gates layout regression — living documentation of
///        the mirrored / dissolved backend layout that the architecture scrubs
///        (tests/architecture/*.sh) enforce. It includes one header from each
///        invariant the gates protect, so compilation is the out-of-library
///        witness that the locked-down shape actually resolves and links:
///          - contracts under core/, not a facade layer        [no_facade]
///          - full-word backend directory names                [path_words]
///          - reciprocal-space math is backend-owned, not top-level [no_toplevel]
///          - the dissolved runtime: a thin coordinator over the
///            communication / device / asynchronous sub-runtimes [no_backend]
///        The .sh scrubs are the enforcement; this is the consumer-side witness.
#include "atomiverse/core/plan.h"                         // core/ contracts
#include "atomiverse/cpu/asynchronous/analysis_runtime.h" // async analysis
#include "atomiverse/cpu/communication/domain_runtime.h"  // DD lifecycle
#include "atomiverse/cpu/device/device_runtime.h"         // threads + timing
#include "atomiverse/cpu/driver/runtime.h"                // the coordinator
#include "atomiverse/cpu/dynamics/md.h"                   // full-word domain
#include "atomiverse/cpu/longrange/ewald_sum.h"           // backend-owned math

#include <cstdio>

int main() {
    std::puts("architecture_gates: mirrored full-word layout with a dissolved "
              "runtime resolves and links");
    return 0;
}
