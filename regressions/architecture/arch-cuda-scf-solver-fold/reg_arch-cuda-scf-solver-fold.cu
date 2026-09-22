/// @file reg_arch-cuda-scf-solver-fold.cu
/// @brief Architecture gate: cuda::scf::RHF public run after SCFSolver fold.
///
/// Literature H2/STO-3G energy via Frame-only public API. Built only with
/// ATV_BUILD_CUDA + ATV_BUILD_HF.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cuda/scf/rhf.cuh"

namespace {

ATV::cpu::Frame make_h2() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = 0.74;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_mass)[1] = 1.008;
    frame.box = cpu::Box::cubic(100.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;
    constexpr double kEnergyRefHa = -1.1167;
    constexpr double kTolHa = 0.01;

    cpu::Frame frame = make_h2();
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    rhf.with_gradient(false);
    rhf.run(frame);

    const double e_ha =
        static_cast<double>(frame.meta.get(kEnergy)) / units::kHartreeInEv;
    const double err = std::abs(e_ha - kEnergyRefHa);

    std::printf("arch-cuda-scf-solver-fold: E=%.8f Ha  ref=%.4f  |dE|=%.3e  "
                "converged=%d  device_scf=%d\n",
                e_ha, kEnergyRefHa, err, rhf.converged() ? 1 : 0,
                rhf.last_scf_on_device() ? 1 : 0);

    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: SCF did not converge\n");
        return 1;
    }
    if (!rhf.last_scf_on_device()) {
        std::fprintf(stderr,
                     "FAIL: expected device s/p SCF path for H2/STO-3G\n");
        return 1;
    }
    if (!(err <= kTolHa)) {
        std::fprintf(stderr, "FAIL: |E-ref|=%.3e > tol %.3e\n", err, kTolHa);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
