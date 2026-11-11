#include "../nddo_nve_cuda.cuh"
#include "atomiverse/cuda/semiempirical/pm3_solver.cuh"
#include "harness/registry.h"
#include "nddo-nve.h"
#include <cstring>

namespace atv_reg_semiempirical_pm3_nve {
int run(int argc, char** argv) {
    std::string device = "both";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--device=", 9) == 0)
            device = argv[i] + 9;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device = argv[++i];
    }
    if (device != "cpu" && device != "cuda" && device != "both")
        return 1;
    try {
        if (device == "cpu")
            return run_cpu();
        return atv_reg_nddo_nve_cuda::grade<ATV::cpu::semiempirical::PM3,
                                            ATV::cuda::semiempirical::PM3>(
            water_pin(), "PM3", device == "both");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL PM3 NVE: %s\n", error.what());
        return 1;
    }
}
} // namespace atv_reg_semiempirical_pm3_nve
ATV_REGRESSION("semiempirical/pm3", "nddo-nve",
               atv_reg_semiempirical_pm3_nve::run);
