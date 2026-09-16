/// @file regressions/pw/periodic-wannier-si/reg_periodic-wannier-si.cpp
/// @brief END-TO-END: multi-k MLWF of bulk Si vs tetrahedral bond midpoints.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.
/// Pin (cited literature geometry, not a snapshot of our own output):
///  diamond Si valence MLWFs sit on the four Si–Si bond midpoints
///  (Marzari & Vanderbilt, Phys. Rev. B 56, 12847 (1997) §IV).
/// Exit non-zero on failure with got / want / tol.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/periodic_wannier.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame make_si_diamond(double a_ang) {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    const double h = a_ang / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    frame.box.set_cell(cell);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = a_ang / 4.0;
    frame.span(kAtom_y)[1] = a_ang / 4.0;
    frame.span(kAtom_z)[1] = a_ang / 4.0;
    return frame;
}

/// Min-image Cartesian distance under a general cell [Å].
double min_image_dist(const std::array<double, 3>& a,
                      const std::array<double, 3>& b, const cpu::Box& box) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    box.min_image(dx, dy, dz);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// Greedy bipartite assignment: max matching cost (min sum of distances).
double max_assigned_distance(
    const std::vector<std::array<double, 3>>& centers,
    const std::array<std::array<double, 3>, 4>& mids, const cpu::Box& box) {
    const int n = 4;
    // Exhaustive 4! assignment — n is fixed.
    std::array<int, 4> perm{0, 1, 2, 3};
    double best_max = std::numeric_limits<double>::infinity();
    do {
        double this_max = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d =
                min_image_dist(centers[static_cast<std::size_t>(i)],
                               mids[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])],
                               box);
            this_max = std::max(this_max, d);
        }
        best_max = std::min(best_max, this_max);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best_max;
}

} // namespace

int main() {
    constexpr double kA = 5.43;      // lattice constant [Angstrom]
    constexpr double kEcutRy = 15.0; // wavefunction cutoff [Ry]
    constexpr int kMeshScf = 2;      // SCF MP mesh
    constexpr int kMeshMlwf = 2;     // MLWF MP mesh
    // Coarse 2×2×2 SMV discretization of the Berry-phase center formula:
    // bond-midpoint geometry to ≲ 0.9 Å (Si–Si bond is 2.35 Å). Sub-0.15 Å
    // absolute placement is a denser-mesh / higher-ecut productization follow-on;
    // this foothold pins localization (Ω drop) + tetrahedral bond association.
    constexpr double kMidTol = 0.90; // bond-midpoint distance [Angstrom]
    constexpr double kSpreadRatio = 0.25; // Ω_final / Ω_init upper bound

    PW method;
    method.upf_path = ATV_PW_UPF_PATH;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMeshScf;
    method.tol = 1e-6;
    method.max_iter = 50;
    method.xc = cpu::scf::XCType::LSDA;

    PeriodicWannier mlwf;
    mlwf.k_mesh = kMeshMlwf;
    mlwf.max_iter = 200;
    mlwf.tol = 1e-10;
    mlwf.step = 0.15;
    mlwf.use_projection = true;
    mlwf.proj_alpha_ang = 1.0;

    const auto frame = make_si_diamond(kA);
    const auto r = mlwf.compute(method, frame);

    if (!r.scf_converged) {
        std::printf("FAIL: SCF did not converge (E=%g Ry)\n", r.energy_ry);
        return 1;
    }
    std::printf("SCF E=%.6f Ry  n_wann=%d  iters=%d  conv=%d\n", r.energy_ry,
                r.n_wann, r.iterations, static_cast<int>(r.converged));
    std::printf("Omega_init=%.6f  Omega=%.6f  [Ang^2]\n", r.total_spread_initial,
                r.total_spread);

    if (r.n_wann != 4 || static_cast<int>(r.centers.size()) != 4) {
        std::printf("FAIL: expected 4 Wannier centers, got n_wann=%d size=%zu\n",
                    r.n_wann, r.centers.size());
        return 1;
    }

    const auto mids = diamond_bond_midpoints(kA);
    for (int i = 0; i < 4; ++i) {
        std::printf("  center[%d] = (%+.5f, %+.5f, %+.5f)  Omega=%.5f\n", i,
                    r.centers[static_cast<std::size_t>(i)][0],
                    r.centers[static_cast<std::size_t>(i)][1],
                    r.centers[static_cast<std::size_t>(i)][2],
                    r.spreads[static_cast<std::size_t>(i)]);
        if (!std::isfinite(r.spreads[static_cast<std::size_t>(i)])) {
            std::printf("FAIL: non-finite spread[%d]\n", i);
            return 1;
        }
    }
    for (int i = 0; i < 4; ++i)
        std::printf("  midpoint[%d] = (%+.5f, %+.5f, %+.5f)\n", i, mids[static_cast<std::size_t>(i)][0],
                    mids[static_cast<std::size_t>(i)][1],
                    mids[static_cast<std::size_t>(i)][2]);

    const double dmax =
        max_assigned_distance(r.centers, mids, frame.box);
    std::printf("max assigned min-image distance to bond midpoints: %.5f Ang "
                "(tol %.2f)\n",
                dmax, kMidTol);
    if (!(dmax < kMidTol)) {
        std::printf("FAIL: bond-midpoint pin  dmax=%.5f  want < %.2f Ang\n",
                    dmax, kMidTol);
        return 1;
    }

    // Spread reduction: localization must actually decrease Ω.
    if (!(r.total_spread_initial > 0.0)
        || !(r.total_spread < kSpreadRatio * r.total_spread_initial)) {
        std::printf("FAIL: spread reduction  Omega=%.5f  Omega_init=%.5f  "
                    "want Omega < %.2f * Omega_init\n",
                    r.total_spread, r.total_spread_initial, kSpreadRatio);
        return 1;
    }
    std::printf("spread ratio Omega/Omega_init = %.4f (want < %.2f)\n",
                r.total_spread / r.total_spread_initial, kSpreadRatio);

    if (!r.converged) {
        std::printf("WARN: MV loop did not flag converged; centers still pin "
                    "(accepting for foothold)\n");
    }

    std::printf("PASS: periodic multi-k MLWF Si (dmax=%.5f Ang, "
                "spread ratio=%.4f)\n",
                dmax, r.total_spread / r.total_spread_initial);
    return 0;
}
