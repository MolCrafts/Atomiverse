/// @file regressions/scf/cam-b3lyp-h2-sto3g/reg_cam-b3lyp-h2-sto3g.cpp
/// @brief H₂ / STO-3G closed-shell CAM-B3LYP energy (RSH foothold).
///
/// Out-of-library consumer: public `cpu/scf/methods.h` + `cpu/frame/frame.h`
/// only. Path: `RKS{XCType::CAMB3LYP}` + `run(frame)`. No Config bag, no
/// factory, no live PySCF.
///
/// v1 CAM-B3LYP reuses the B3LYP semi-local mix and adds
/// \(\alpha K[1/r] + \beta K[\mathrm{erf}(\mu r)/r]\) with Yanai
/// \((\alpha,\beta,\mu)=(0.19,0.46,0.33)\). Golden is a composed PySCF
/// 2.14.0 pin that uses the same mix (see REFERENCE.md).

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/xc_functional.h"

int main() {
    using namespace ATV;

    const auto xc = cpu::scf::XCType::CAMB3LYP;
    const auto rsh = cpu::scf::rsh_exchange_params(xc);
    if (std::fabs(rsh.alpha - 0.19) > 1e-15
        || std::fabs(rsh.beta - 0.46) > 1e-15
        || std::fabs(rsh.mu - 0.33) > 1e-15) {
        std::fprintf(
            stderr,
            "FAIL: CAM-B3LYP rsh_exchange_params != (0.19,0.46,0.33)\n");
        return 1;
    }

    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RKS rks(xc);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-10;
    rks.max_iter = 100;
    rks.run(frame);

    if (!rks.converged()) {
        std::fprintf(stderr, "FAIL: RKS/CAM-B3LYP H2 did not converge\n");
        return 1;
    }

    // PySCF 2.14.0 composed pin — see REFERENCE.md.
    constexpr double kEpinHa = -1.3067931936345114;
    constexpr double kTolHa = 1.0e-6;

    const double e_ha = rks.energy();
    const double diff = std::fabs(e_ha - kEpinHa);
    std::printf("E_RKS(CAM-B3LYP/STO-3G H2) got=%.16f want=%.16f "
                "|diff|=%.3e tol=%.0e\n",
                e_ha, kEpinHa, diff, kTolHa);
    if (!std::isfinite(e_ha) || diff > kTolHa) {
        std::fprintf(stderr, "FAIL: CAM-B3LYP H2 energy off PySCF pin\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
