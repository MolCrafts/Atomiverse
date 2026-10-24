/// @brief CPU/CUDA LJ dimer minimization against the same analytic minimum.
#include <cstring>
#include <limits>
#include <string>

#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "harness/registry.h"
#include "lj_dimer.h"

namespace atv_reg_minimizer_lj_dimer {
int run(int argc, char** argv) {
    std::string device = "both";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--device=", 9) == 0)
            device = argv[i] + 9;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device = argv[++i];
    }
    if (device != "cpu" && device != "cuda" && device != "both") {
        std::printf("FAIL LJ dimer device got=%s want=cpu|cuda|both\n",
                    device.c_str());
        return 1;
    }
    int rc = 0;
    if (device == "cpu" || device == "both")
        rc |= lj_dimer_case::run_cpu();
    if (device == "cuda" || device == "both") {
        using namespace ATV;
        cuda::minimizer::Minimize opt;
        // CUDA uses a fixed step, while CPU uses StrongWolfe. Alpha=1 is the
        // full quasi-Newton step, still subject to the optimizer's 0.1 A cap.
        // Both paths must reach the same analytic minimum, not the same trajectory.
        opt.set_optimizer(cuda::minimizer::LBFGS{10, 1.0, 20});
        opt.add_criterion(cuda::ForceConverged{0.01, 0.005, 0.05, 0.02, 1e-6});
        opt.set_evaluator(cuda::PairLJCut{lj_dimer_case::epsilon,
                                          lj_dimer_case::sigma,
                                          lj_dimer_case::cutoff});
        opt.set_neighbor(cuda::NeighborList{lj_dimer_case::cutoff, 0.3});
        opt.build();
        std::printf("[cuda] LBFGS: memory=10 alpha0=1 fixed max_iter=200\n");
        rc |= lj_dimer_case::check_minimum(
            opt, "cuda", std::numeric_limits<cuda::force_t>::epsilon());
    }
    return rc;
}
} // namespace atv_reg_minimizer_lj_dimer

ATV_REGRESSION("minimizer/geometry", "lj_dimer",
               atv_reg_minimizer_lj_dimer::run);
