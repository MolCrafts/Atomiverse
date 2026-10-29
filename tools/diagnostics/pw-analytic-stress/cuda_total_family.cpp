#include "cuda_total_common.h"
#include "qe-gamma-pbe-fixed-2026-10-28/pins.h"
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(
            stderr, "usage: cuda-total-family pbe|pbesol|revpbe|rpbe --mode\n");
        return 2;
    }
    for (const auto& ref : pw_cuda_pbe_qe::pins)
        if (std::string(argv[1]) == ref.model)
            return run_cuda_total(argc - 1, argv + 1, ref);
    std::fprintf(stderr, "unknown PBE-family model\n");
    return 2;
}
