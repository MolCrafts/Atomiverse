/// @file regressions/scf/cuda-uhf/reg_cuda-uhf.cu
/// @brief O2 and OH STO-3G UHF on CPU and CUDA against the same PySCF pins.
///
/// `--device=cpu|cuda|both` (default both). The CUDA leg must report
/// last_scf_on_device and last_force_on_device. See REFERENCE.md.

#include <cstring>
#include <string>

#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/scf/uhf.cuh"
#include "cuda-uhf.h"
#include "harness/registry.h"

namespace atv_reg_scf_cuda_uhf {

namespace {

std::string parse_device(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0)
            want = a + 9;
        else if (std::strcmp(a, "--device") == 0 && i + 1 < argc)
            want = argv[++i];
        else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                 || std::strcmp(a, "both") == 0)
            want = a;
    }
    return want;
}

cuda_uhf_case::Out run_cpu(ATV::cpu::Frame& frame, int na, int nb, int charge) {
    ATV::cpu::scf::UHF uhf{
        ATV::cpu::scf::BasisSet::STO3G, na, nb, charge, 1e-12, 100};
    uhf.init_guess = ATV::cpu::scf::InitGuess::MINAO;
    uhf.damping_factor = 0.0;
    uhf.diis_start_iter = 1;
    uhf.diis_max_vectors = 8;
    const auto g = uhf.forces(frame);
    cuda_uhf_case::Out out;
    out.energy_ha = uhf.energy();
    out.converged = uhf.converged();
    out.forces = g.forces;
    return out;
}

cuda_uhf_case::Out run_cuda(ATV::cpu::Frame& frame, int na, int nb,
                            int charge) {
    ATV::cuda::scf::UHF uhf{
        ATV::cpu::scf::BasisSet::STO3G, na, nb, charge, 1e-12, 100};
    const auto g = uhf.forces(frame);
    if (!uhf.last_scf_on_device() || !uhf.last_force_on_device()) {
        std::printf("FAIL: CUDA UHF left the device path "
                    "(scf=%d force=%d)\n",
                    uhf.last_scf_on_device() ? 1 : 0,
                    uhf.last_force_on_device() ? 1 : 0);
        cuda_uhf_case::Out bad;
        bad.converged = false;
        return bad;
    }
    cuda_uhf_case::Out out;
    out.energy_ha = uhf.energy();
    out.converged = uhf.converged();
    out.forces = g.forces;
    return out;
}

} // namespace

int run(int argc, char** argv) {
    const std::string want = parse_device(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::printf("FAIL: unknown --device=%s\n", want.c_str());
        return 1;
    }
    int rc = 0;
    if (want == "cpu" || want == "both")
        rc |= cuda_uhf_case::run_all(run_cpu);
    if (want == "cuda" || want == "both")
        rc |= cuda_uhf_case::run_all(run_cuda);
    return rc;
}

} // namespace atv_reg_scf_cuda_uhf

ATV_REGRESSION("scf/uhf", "cuda-uhf", atv_reg_scf_cuda_uhf::run);
