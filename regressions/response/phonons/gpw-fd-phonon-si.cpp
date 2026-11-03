/// @file regressions/response/phonons/gpw-fd-phonon-si.cpp
/// @brief GPW Si2 phonons at the stationary CP2K vibration input geometry.
/// External energy/force and stretch pins: CP2K 2026.2; see REFERENCE.md.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/vibration/nist_constants.h"
#include "atomiverse/cpu/vibration/phonon.h"
#include "harness/registry.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace atv_reg_gpw_gpw_fd_phonon_si {
using namespace ATV;
namespace {
constexpr double kL = 8.0;
constexpr double kBond =
    2.1450905162; // Actual CORE_PPL GRID VIB input, Angstrom.
constexpr double kMassSi = 28.0855;
constexpr double kZeroModeTol = 80.0; // Original near-zero bound, cm^-1.
constexpr double kOmegaTol = 100.0;   // Original curvature bound, cm^-1.
constexpr double kExternalStretch = 573.65790135; // CP2K CORE_PPL GRID VIB.
constexpr double kWallBudget = 60.0; // Whole section, repository limit.
cpu::Frame si2(double bond) {
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_atomic_number)[i] = 14;
        frame.span(kAtom_mass)[i] = kMassSi;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = 4 + (i == 0 ? -0.5 : 0.5) * bond;
        frame.span(kAtom_y)[i] = frame.span(kAtom_z)[i] = 4;
    }
    frame.box = cpu::Box::cubic(kL);
    return frame;
}
} // namespace
int run() {
    const auto start = std::chrono::steady_clock::now();
    const cpu::gpw::Kind Si{
        molrs::Element::Si,
        cpu::gpw::GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::Si, "GTH-PBE-q4")};
    cpu::gpw::GpwDft method{{Si}, 150, cpu::xc::XCType::PADE};
    method.max_iter = 200;
    method.tol = 1e-10;
    method.residual_tol = 1e-8;
    method.eps_default = 1e-10;
    method.nonperiodic = true;
    method.n_alpha = 5;
    method.n_beta = 3;

    // Nonstationary input deliberately retains the former test bond. Its
    // external full force vector catches clipped charge tails and wrong kernels.
    auto displaced = si2(2.246);
    method.build(displaced);
    const auto& off = method.forces(displaced);
    constexpr double reference_energy = -7.600561394706659;
    constexpr double reference_forces[2][3] = {
        {0.0272740080, 6.32154382e-9, 3.51118623e-9},
        {-0.0272740080, -6.32099739e-9, -3.51163041e-9}};
    constexpr double energy_tol = 1e-5, force_tol = 1e-4;
    if (!std::isfinite(off.e_total)
        || std::abs(off.e_total - reference_energy) > energy_tol) {
        std::printf("FAIL E got=%.14f want=%.14f tol=%g Ha\n", off.e_total,
                    reference_energy, energy_tol);
        return 1;
    }
    for (int atom = 0; atom < 2; ++atom) {
        const double force[3] = {displaced.span(kAtom_fx)[atom],
                                 displaced.span(kAtom_fy)[atom],
                                 displaced.span(kAtom_fz)[atom]};
        for (int axis = 0; axis < 3; ++axis) {
            const double got =
                force[axis] * units::kEvPerAngstromInHartreePerBohr;
            const double want = reference_forces[atom][axis];
            if (!std::isfinite(got) || std::abs(got - want) > force_tol) {
                std::printf(
                    "FAIL F(%d,%d) got=%.12g want=%.12g tol=%g Ha/Bohr\n", atom,
                    axis, got, want, force_tol);
                return 1;
            }
        }
    }

    // Five raw Cartesian translation/rotation modes require a stationary
    // molecule. Enforce this precondition instead of projecting away errors.
    auto frame = si2(kBond);
    double E0, Ep, Em;
    constexpr double h = 5e-3;
    {
        cpu::Evaluate evaluate;
        evaluate.on(evaluate::Phase::evaluate).add_ref(method);
        evaluate.enable_timing(false);
        evaluate.build();
        evaluate.run(frame);
        for (int atom = 0; atom < 2; ++atom)
            for (double force :
                 {frame.span(kAtom_fx)[atom], frame.span(kAtom_fy)[atom],
                  frame.span(kAtom_fz)[atom]}) {
                const double got =
                    force * units::kEvPerAngstromInHartreePerBohr;
                if (!std::isfinite(got) || std::abs(got) > 1e-5) {
                    std::printf("FAIL equilibrium force got=%.12g want=0 "
                                "tol=1e-5 Ha/Bohr\n",
                                got);
                    return 1;
                }
            }
        E0 = method.last_result().e_total;
        if (std::abs(E0 - (-7.603345177586478)) > energy_tol) {
            std::printf("FAIL equilibrium E got=%.14f want=-7.603345177586478 "
                        "tol=%g Ha\n",
                        E0, energy_tol);
            return 1;
        }
        auto plus = si2(kBond + h), minus = si2(kBond - h);
        evaluate.run(plus);
        Ep = method.last_result().e_total;
        evaluate.run(minus);
        Em = method.last_result().e_total;
    }
    const double curvature = (Ep + Em - 2 * E0) / (h * h) * cpu::vib::nist::BOHR
                             * cpu::vib::nist::BOHR;
    if (!(curvature > 0) || !std::isfinite(curvature)) {
        std::printf("FAIL stretch curvature got=%g want=>0 Ha/Bohr^2\n",
                    curvature);
        return 1;
    }
    const double one_dimensional = std::sqrt(curvature / (0.5 * kMassSi))
                                   * cpu::vib::nist::au2wavenumber();
    cpu::vib::Phonon phonon(1, 1, 1);
    phonon.on(vib::Phase::force).add_ref(method);
    phonon.enable_timing(false);
    phonon.build();
    const auto spectrum = phonon.run(frame);
    if (spectrum.n_atoms != 2 || spectrum.frequencies.size() != 6) {
        std::printf("FAIL spectrum atoms=%d modes=%zu want=2,6\n",
                    spectrum.n_atoms, spectrum.frequencies.size());
        return 1;
    }
    std::vector<double> modes(spectrum.frequencies.begin(),
                              spectrum.frequencies.end());
    for (double w : modes)
        if (!std::isfinite(w))
            return 1;
    std::sort(modes.begin(), modes.end(),
              [](double a, double b) { return std::abs(a) < std::abs(b); });
    for (int i = 0; i < 5; ++i)
        if (std::abs(modes[i]) >= kZeroModeTol) {
            std::printf("FAIL zero mode got=%.6f want=0 tol=%.1f cm^-1\n",
                        modes[i], kZeroModeTol);
            return 1;
        }
    const double stretch = modes.back();
    if (std::abs(stretch - kExternalStretch) > kOmegaTol
        || std::abs(stretch - one_dimensional) > kOmegaTol) {
        std::printf("FAIL stretch got=%.8f CP2K=%.8f energy-curvature=%.8f "
                    "tol=%g cm^-1\n",
                    stretch, kExternalStretch, one_dimensional, kOmegaTol);
        return 1;
    }
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (wall > kWallBudget) {
        std::printf("FAIL section wall got=%.3f want<=%.1f seconds\n", wall,
                    kWallBudget);
        return 1;
    }
    std::printf("PASS gpw-fd-phonon-si: 5/5 near-zero, stretch %.6f cm^-1 "
                "(CP2K %.6f), whole section %.3f s\n",
                stretch, kExternalStretch, wall);
    return 0;
}
} // namespace atv_reg_gpw_gpw_fd_phonon_si
ATV_REGRESSION("response/phonons", "gpw-fd-phonon-si",
               atv_reg_gpw_gpw_fd_phonon_si::run);
