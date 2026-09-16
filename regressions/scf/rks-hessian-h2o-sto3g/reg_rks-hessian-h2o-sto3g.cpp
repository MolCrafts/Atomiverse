/// @file regressions/scf/rks-hessian-h2o-sto3g/reg_rks-hessian-h2o-sto3g.cpp
/// @brief H₂O RKS-LSDA/STO-3G analytic nuclear Hessian vs offline PySCF
///        2.14.0 `hessian.RKS` pins (eV/Å²). See REFERENCE.md.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Exit non-zero on any miss. ≤ 60 s, single CPU.

#include <cmath>
#include <cstdio>
#include <utility>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

namespace {

ATV::cpu::Frame water_frame() {
    ATV::cpu::Frame frame(3);
    // PySCF pin_scf.py water preset [Å].
    const double xyz[3][3] = {
        {0.0, 0.0, 0.1173},
        {0.0, 0.7572, -0.4692},
        {0.0, -0.7572, -0.4692},
    };
    const int Z[3] = {8, 1, 1};
    for (int i = 0; i < 3; ++i) {
        frame.span(ATV::kAtom_type)[i] = Z[i];
        frame.span(ATV::kAtom_x)[i] = static_cast<ATV::StorageReal>(xyz[i][0]);
        frame.span(ATV::kAtom_y)[i] = static_cast<ATV::StorageReal>(xyz[i][1]);
        frame.span(ATV::kAtom_z)[i] = static_cast<ATV::StorageReal>(xyz[i][2]);
    }
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = water_frame();

    cpu::scf::RKS rks(cpu::scf::XCType::LSDA);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-12;
    rks.max_iter = 200;
    rks.grid_level = cpu::scf::GridLevel::MatchPySCF;

    const cpu::vib::HessianResult res = rks.hessian(frame);

    // PySCF 2.14.0 FD-of-grad pins — see REFERENCE.md (never live PySCF).
    constexpr double want_e = -74.732038853874;
    // Cross-engine residual: ATV MatchPySCF grid (33632 pts) vs PySCF
    // level=3 (33704 pts) leaves ~2e-3 eV/Å on forces; that propagates
    // through h=1e-4 Bohr FD to ~1e-2 eV/Å² on Hessian elements. 2e-2
    // is 2× that floor and still catches sign/factor errors (stiff blocks
    // are ~10² eV/Å²). Full CPKS twin of hessian.RKS at 1e-5 is follow-up.
    constexpr double tol = 2e-2; // eV/Å² product gate

    // Gated elements (row, col, want) — FD twin of analytic RKS gradients.
    const std::pair<int, int> idx[] = {
        {0, 0}, {1, 1}, {2, 2}, {0, 3}, {0, 6}, {1, 4}, {1, 5},
        {1, 7}, {1, 8}, {2, 5}, {4, 4}, {5, 5}, {7, 7}, {8, 8},
    };
    const double want[] = {
        -9.632120712096,  89.527387528595,  54.829425918251, 4.820104284662,
        4.820104284678,   -44.758907247881, 38.397479554108, -44.758906500277,
        -38.397479301244, -27.417403943596, 47.406667576594, 26.895192808433,
        47.406666805220,  26.895192389626,
    };
    constexpr int n_gate = 14;

    if (res.n_atoms != 3 || res.hessian.shape()[0] != 9u
        || res.hessian.shape()[1] != 9u) {
        std::printf("FAIL: expected 9x9 Hessian for 3 atoms, got n_atoms=%d "
                    "shape=(%zu,%zu)\n",
                    res.n_atoms,
                    static_cast<std::size_t>(res.hessian.shape()[0]),
                    static_cast<std::size_t>(res.hessian.shape()[1]));
        return 1;
    }

    const double e = rks.energy();
    std::printf("RKS-LSDA E = %.12f Ha (pin %.12f, dE = %.3e)\n", e, want_e,
                e - want_e);

    int status = 0;
    if (!(std::fabs(e - want_e) < 1e-6) || !rks.converged()) {
        std::printf("FAIL: RKS energy/convergence mismatch\n");
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

    // Symmetry of the assembled Hessian.
    for (int r = 0; r < 9; ++r)
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

    std::printf("PASS: H2O RKS-LSDA/STO-3G nuclear Hessian "
                "(FD of analytic forces; max gate dev %.3e eV/Ang^2, "
                "tol %.1e)\n"
                "      Hyy(O)=%.6f  Hzz(O)=%.6f eV/Ang^2\n"
                "      [offline PySCF 2.14.0 FD of RKS gradients]\n",
                max_dev, tol, res.hessian(1, 1), res.hessian(2, 2));
    return 0;
}
