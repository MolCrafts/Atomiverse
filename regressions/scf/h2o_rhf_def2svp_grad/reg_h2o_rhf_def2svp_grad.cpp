/// @file regressions/scf/h2o_rhf_def2svp_grad/reg_h2o_rhf_def2svp_grad.cpp
/// @brief H2O / def2-SVP (spherical 5d) RHF analytic nuclear gradient vs the
///        externally pinned PySCF `nuc_grad_method().kernel()` value — the
///        spherical-basis analytic-forces chain link's e2e.
///
/// Out-of-library consumer: public headers + link atomiverse only. def2-SVP is
/// DEFINED with real spherical harmonics; this run proves the derivative-
/// integral c2s sandwich (1e path) + density back-transform (2e path) end to
/// end against a third-party analytic gradient — NOT a self-snapshot.
///
/// Checks (REFERENCE.md, PySCF 2.14.0, conv_tol=1e-12):
///   1. element-wise |g_analytic − g_PySCF| ≤ 1e-6 Ha/Bohr
///   2. translational invariance |Σ_A g_A| ≤ 1e-8 Ha/Bohr
///
/// CUDA-off path (cpu only). With `ATV_BUILD_CUDA=ON`, CMake builds the
/// multi-device `reg_h2o_rhf_def2svp_grad.cu` (`--device=cpu|cuda|both`)
/// instead — one case, not a `*_cuda` sibling.
///
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // Pinned geometry (identical to REFERENCE.md and the energy sibling):
    // O(0,0,0.1173), H(0,±0.7572,−0.4692) Å.
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::Def2SVP, 0, 1e-12, 200};
    const cpu::scf::GradientResult res = rhf.forces(frame);

    // Atomiverse reports FORCE (−dE/dR) in eV/Å; convert back to the gradient
    // dE/dR in Ha/Bohr to compare with the PySCF analytic gradient.
    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 analytic gradient [Ha/Bohr] (REFERENCE.md). x components are
    // numerically zero (molecule in the yz-plane).
    const double want[3][3] = {
        {0.0, 0.0, 1.804450607022e-02},                 // O
        {0.0, 1.107031890619e-02, -9.022253035110e-03}, // H
        {0.0, -1.107031890619e-02, -9.022253035110e-03} // H
    };
    const double tol = 1e-6;

    double max_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < 3; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(A, d);
            net[d] += g;
            const double err = std::fabs(g - want[A][d]);
            if (err > max_err)
                max_err = err;
            std::printf("g[%d][%d] got=% .12e want=% .12e |err|=%.2e\n", A, d,
                        g, want[A][d], err);
        }
    }
    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});
    std::printf("max_elementwise_err=%.3e tol=%.0e  |Σ_A g_A|_max=%.3e\n",
                max_err, tol, net_max);

    if (!(max_err < tol)) {
        std::printf("FAIL: spherical def2-SVP RHF gradient off PySCF oracle\n");
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: gradient not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
