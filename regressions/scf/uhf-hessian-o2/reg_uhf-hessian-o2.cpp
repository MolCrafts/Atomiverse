/// @file regressions/scf/uhf-hessian-o2/reg_uhf-hessian-o2.cpp
/// @brief O₂ triplet UHF/STO-3G analytic nuclear Hessian vs offline PySCF
///        2.14.0 `hessian.UHF` pins (eV/Å²). See REFERENCE.md.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// ROHF density warm-start (same multi-minima path as ump2-o2-sto3g) then
/// `UHF::hessian`. Exit non-zero on any miss. ≤ 60 s, single CPU.

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

    // Cold UHF often lands on a higher local solution for O2; warm-start
    // from ROHF densities to reach the PySCF UHF minimum used for the pin.
    cpu::scf::ROHF rohf(9, 7);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;
    rohf.run(frame);
    if (!rohf.converged()) {
        std::printf("FAIL: ROHF warm-start did not converge\n");
        return 1;
    }

    using Spin = cpu::scf::SlaterDeterminant::Spin;
    cpu::scf::SCFInitialGuess guess;
    guess.P_alpha = rohf.determinant().density(Spin::alpha);
    guess.P_beta = rohf.determinant().density(Spin::beta);
    guess.has_alpha = true;
    guess.has_beta = true;

    cpu::scf::UHF uhf(9, 7);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;

    const cpu::vib::HessianResult res = uhf.hessian(frame, &guess);

    // PySCF 2.14.0 pins — see REFERENCE.md (never live-import PySCF here).
    constexpr double want_e_hf = -147.633946785450;
    constexpr double tol = 1e-5;     // eV/Å² product gate
    constexpr double off_tol = 1e-3; // near-zero off-axis elements

    // Gated Cartesian-diagonal / stretch elements (row, col, want).
    const std::pair<int, int> idx[] = {
        {0, 0}, {1, 1}, {2, 2}, {0, 3}, {1, 4}, {2, 5}, {3, 3}, {4, 4}, {5, 5},
    };
    const double want[] = {
        -1.0567201218, -1.0567201218, 135.7176127486,
        1.0567201218,  1.0567201218,  -135.7176127486,
        -1.0567201219, -1.0567201220, 135.7176127485,
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

    const double e_hf = uhf.energy();
    std::printf("UHF E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_hf, want_e_hf,
                e_hf - want_e_hf);

    int status = 0;
    if (!(std::fabs(e_hf - want_e_hf) < 1e-8) || !uhf.converged()) {
        std::printf("FAIL: UHF energy/convergence mismatch\n");
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
            // Also treat the symmetric partners of gated off-diagonal
            // stretch blocks as gated (they share the pin value).
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

    std::printf("PASS: O2 UHF/STO-3G analytic Hessian (max gate dev %.3e "
                "eV/Ang^2, tol %.1e)\n"
                "      Hzz(O1)=%.6f  Hzz(O1,O2)=%.6f eV/Ang^2\n"
                "      [Pople-Krishnan-Schlegel-Binkley 1979; Yamaguchi "
                "unrestricted Hessian]\n",
                max_dev, tol, res.hessian(2, 2), res.hessian(2, 5));
    return 0;
}
