/// @file regressions/scf/h2o_rhf_sto3g/reg_h2o_rhf_sto3g.cpp
/// @brief H2O / STO-3G RHF energy vs LibreTexts pin — **CPU device**.
///
/// When `ATV_BUILD_CUDA=ON`, CMake builds the **same case** from
/// `reg_h2o_rhf_sto3g.cu` instead, which accepts `--device=cpu|cuda|both`
/// (PyTorch-style single regression, multi-device). This `.cpp` is the
/// CUDA-off path only.
///
/// Public API only: Frame + `cpu::scf::RHF`. No SCFSolver /
/// *Config / user Context. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    constexpr double kPi = 3.14159265358979323846;
    const double r_oh = 0.95;
    const double half = 0.5 * 104.5 * kPi / 180.0;
    const double hx = r_oh * std::sin(half);
    const double hz = r_oh * std::cos(half);

    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = hx;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = hz;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = -hx;
    frame.span(kAtom_y)[2] = 0.0;
    frame.span(kAtom_z)[2] = hz;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    rhf.run(frame);

    constexpr double kEnergyRef = -74.961754;
    constexpr double kTol = 1.0e-3;
    const double got = frame.meta.get(kEnergy) / units::kHartreeInEv;

    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL [cpu]: RHF/STO-3G(H2O) did not converge\n");
        return 1;
    }
    if (!std::isfinite(got) || std::abs(got - kEnergyRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cpu]: E_RHF/STO-3G(H2O) got=%.6f want=%.6f "
                     "tol=%.1e Ha\n",
                     got, kEnergyRef, kTol);
        return 1;
    }

    std::printf("PASS [cpu]: E_RHF/STO-3G(H2O) = %.6f Ha (want %.6f +/- %.1e) "
                "[LibreTexts §28.2; PySCF cross-check]\n",
                got, kEnergyRef, kTol);
    return 0;
}
