/// @file regressions/scf/uks-hessian-o2-sto3g/reg_uks-hessian-o2-sto3g.cpp
/// @brief O₂ UKS-LSDA/STO-3G nuclear Hessian vs offline PySCF 2.14.0
///        FD-of-analytic-gradient pins (eV/Å²). See REFERENCE.md.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// ROKS density warm-start (same multi-minima path as uks-tda-o2-lsda) then
/// `UKS::hessian`. Exit non-zero on any miss. ≤ 60 s, single CPU.

#include <cmath>
#include <cstdio>
#include <utility>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

namespace {

ATV::cpu::Frame o2_frame() {
    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_mass)[i] = 15.999;
        frame.span(ATV::kAtom_x)[i] = 0.0;
        frame.span(ATV::kAtom_y)[i] = 0.0;
    }
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 1.2075;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = o2_frame();

    // Cold UKS often lands on a higher local solution for O2; warm-start
    // from ROKS densities to reach the PySCF UKS minimum used for the pin.
    cpu::scf::ROKS roks(9, 7, cpu::scf::XCType::LSDA);
    roks.basis = cpu::scf::BasisSet::STO3G;
    roks.tol = 1e-12;
    roks.max_iter = 200;
    roks.grid_level = cpu::scf::GridLevel::MatchPySCF;
    roks.run(frame);
    if (!roks.converged()) {
        std::printf("FAIL: ROKS warm-start did not converge\n");
        return 1;
    }

    using Spin = cpu::scf::SlaterDeterminant::Spin;
    cpu::scf::SCFInitialGuess guess;
    guess.P_alpha = roks.determinant().density(Spin::alpha);
    guess.P_beta = roks.determinant().density(Spin::beta);
    guess.has_alpha = true;
    guess.has_beta = true;

    cpu::scf::UKS uks(9, 7, cpu::scf::XCType::LSDA);
    uks.basis = cpu::scf::BasisSet::STO3G;
    uks.tol = 1e-12;
    uks.max_iter = 200;
    uks.grid_level = cpu::scf::GridLevel::MatchPySCF;

    const cpu::vib::HessianResult res = uks.hessian(frame, &guess);

    // PySCF 2.14.0 FD-of-grad pins — see REFERENCE.md (never live PySCF).
    constexpr double want_e = -147.193805922283;
    // Cross-engine residual: ATV MatchPySCF grid vs PySCF level=3 leaves
    // ~1e-3 eV/Å on forces; that propagates through h=1e-4 Bohr FD to
    // ~1e-2 eV/Å² on Hessian elements. Product gate 2e-2 (2× that floor)
    // still catches sign/factor errors on stiff stretch blocks (~10² eV/Å²).
    // Full UCPKS twin of hessian.UKS at 1e-5 is follow-up.
    constexpr double tol = 2e-2;     // eV/Å² product gate
    constexpr double off_tol = 1e-2; // near-zero off-axis elements

    // Gated Cartesian-diagonal / stretch elements (row, col, want).
    const std::pair<int, int> idx[] = {
        {0, 0}, {1, 1}, {2, 2}, {0, 3}, {1, 4}, {2, 5}, {3, 3}, {4, 4}, {5, 5},
    };
    const double want[] = {
        -6.951874310841, -6.951874310747, 128.455930123774,
        6.951874311534,  6.951874311029,  -128.455930112985,
        -6.951874311765, -6.951874311352, 128.455930103923,
    };
    constexpr int n_gate = 9;

    if (res.n_atoms != 2 || res.hessian.shape()[0] != 6u
        || res.hessian.shape()[1] != 6u) {
        std::printf("FAIL: expected 6x6 Hessian for 2 atoms, got n_atoms=%d "
                    "shape=(%zu,%zu)\n",
                    res.n_atoms,
                    static_cast<std::size_t>(res.hessian.shape()[0]),
                    static_cast<std::size_t>(res.hessian.shape()[1]));
        return 1;
    }

    const double e = uks.energy();
    std::printf("UKS-LSDA E = %.12f Ha (pin %.12f, dE = %.3e)\n", e, want_e,
                e - want_e);

    int status = 0;
    if (!(std::fabs(e - want_e) < 1e-6) || !uks.converged()) {
        std::printf("FAIL: UKS energy/convergence mismatch\n");
        status = 1;
    }

    double max_dev = 0.0;
    for (int g = 0; g < n_gate; ++g) {
        const int r = idx[g].first;
        const int c = idx[g].second;
        const double got = res.hessian(r, c);
        const double dev = std::fabs(got - want[g]);
        if (dev > max_dev)
            max_dev = dev;
        if (!(dev <= tol)) {
            std::printf("FAIL: H(%d,%d) got=%.12f want=%.12f tol=%.1e "
                        "eV/Ang^2\n",
                        r, c, got, want[g], tol);
            status = 1;
        } else {
            std::printf("  H(%d,%d) = %.8f (want %.8f, dev %.3e)\n", r, c, got,
                        want[g], dev);
        }
    }

    // Off-axis elements must be near zero on this collinear geometry.
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) {
            bool gated = false;
            for (int g = 0; g < n_gate; ++g)
                if ((idx[g].first == r && idx[g].second == c)
                    || (idx[g].first == c && idx[g].second == r && r != c)) {
                    gated = true;
                    break;
                }
            if ((r == 3 && c == 0) || (r == 4 && c == 1) || (r == 5 && c == 2))
                gated = true;
            if (gated)
                continue;
            const double got = res.hessian(r, c);
            if (!(std::fabs(got) <= off_tol)) {
                std::printf("FAIL: off-axis H(%d,%d)=%.6e exceeds off_tol "
                            "%.1e eV/Ang^2\n",
                            r, c, got, off_tol);
                status = 1;
            }
        }

    // Symmetry of the assembled Hessian.
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < r; ++c) {
            const double d = std::fabs(res.hessian(r, c) - res.hessian(c, r));
            if (!(d <= 1e-8)) {
                std::printf("FAIL: Hessian not symmetric at (%d,%d) d=%.3e\n",
                            r, c, d);
                status = 1;
            }
        }

    if (status != 0)
        return status;

    std::printf("PASS: O2 UKS-LSDA/STO-3G nuclear Hessian "
                "(FD of analytic forces; max gate dev %.3e eV/Ang^2, "
                "tol %.1e)\n"
                "      Hzz(O1)=%.6f  Hzz(O1,O2)=%.6f eV/Ang^2\n"
                "      [offline PySCF 2.14.0 FD of UKS gradients]\n",
                max_dev, tol, res.hessian(2, 2), res.hessian(2, 5));
    return 0;
}
