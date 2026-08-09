/// @file regressions/scf/h2o_rhf_sto3g_grad/reg_h2o_rhf_sto3g_grad.cpp
/// @brief H2O / STO-3G RHF analytic nuclear gradient vs the externally pinned
///        PySCF `nuc_grad_method().kernel()` value — the s/p analytic-forces
///        chain link's e2e (Hellmann-Feynman + Pulay, minimal basis).
///
/// Out-of-library consumer: public headers + link atomiverse only. STO-3G on
/// H2O carries s and p shells only, so Cartesian and spherical conventions
/// coincide — this pins the s/p derivative-integral path in isolation from
/// the d-shell c2s machinery (that lives in the def2-SVP sibling case).
///
/// Checks (REFERENCE.md, PySCF 2.14.0, conv_tol=1e-12):
///   1. |E_RHF − E_PySCF| ≤ 1e-8 Ha
///   2. element-wise |g_analytic − g_PySCF| ≤ 1e-6 Ha/Bohr
///   3. translational invariance |Σ_A g_A| ≤ 1e-8 Ha/Bohr
///
/// CUDA-off path (cpu only). With `ATV_BUILD_CUDA=ON`, CMake builds the
/// multi-device `reg_h2o_rhf_sto3g_grad.cu` (`--device=cpu|cuda|both`)
/// instead — one case, not a `*_cuda` sibling. The CUDA leg there is the
/// fail-closed DEVICE s/p gradient path (`last_force_on_device()` asserted).
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

    // Pinned geometry (identical to REFERENCE.md and the def2-SVP sibling):
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

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
    const cpu::scf::GradientResult res = rhf.forces(frame);
    if (!rhf.converged()) {
        std::printf("FAIL: RHF forces did not converge\n");
        return 1;
    }

    // PySCF 2.14.0 pin (REFERENCE.md): E_RHF and analytic gradient.
    const double e_ha =
        units::convert(res.energy, units::Energy::eV, units::Energy::Hartree);
    const double e_want = -74.963023138463; // Ha
    std::printf("E got=%.12f want=%.12f |err|=%.2e Ha\n", e_ha, e_want,
                std::fabs(e_ha - e_want));
    if (!(std::fabs(e_ha - e_want) < 1e-8)) {
        std::printf("FAIL: STO-3G RHF energy off PySCF oracle\n");
        return 1;
    }

    // Atomiverse reports FORCE (−dE/dR) in eV/Å; convert back to the gradient
    // dE/dR in Ha/Bohr to compare with the PySCF analytic gradient.
    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 analytic gradient [Ha/Bohr] (REFERENCE.md). x components
    // are numerically zero (molecule in the yz-plane).
    const double want[3][3] = {
        {0.0, 0.0, -6.142775911808e-02},                // O
        {0.0, -2.364133873480e-02, 3.071387955904e-02}, // H (+y)
        {0.0, 2.364133873480e-02, 3.071387955904e-02}   // H (−y)
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
        std::printf("FAIL: STO-3G RHF gradient off PySCF oracle\n");
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: gradient not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
