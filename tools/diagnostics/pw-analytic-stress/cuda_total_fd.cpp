#include "cuda_total_common.h"
#include "qe-gamma-rks-fd-2026-10-28/pins.h"
int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    for (const auto& r : fd_qe::pins)
        if (std::string(argv[1]) == r.model)
            return run_cuda_total(argc - 1, argv + 1, r);
    return 2;
}
