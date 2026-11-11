#include "../nddo_mopac_cuda.cuh"
#include "atomiverse/cpu/semiempirical/mndo_solver.h"
#include "atomiverse/cuda/semiempirical/mndo_solver.cuh"
#include "harness/registry.h"
#include "mndo-mopac.h"

namespace atv_reg_semiempirical_mndo_mopac {
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
        if (device != "cuda" && run_cpu() != 0)
            return 1;
        if (device != "cpu")
            atv_reg_nddo_mopac_cuda::grade<ATV::cpu::semiempirical::MNDO,
                                           ATV::cuda::semiempirical::MNDO>(
                atv_reg_nddo_mopac::kMndoPins, "MNDO");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CUDA MNDO: %s\n", error.what());
        return 1;
    }
    return 0;
}
} // namespace atv_reg_semiempirical_mndo_mopac
ATV_REGRESSION("semiempirical/mndo", "mndo-mopac",
               atv_reg_semiempirical_mndo_mopac::run);
