/// @file regressions/gpw/gpw-mbd-kpoint-stress/reg_gpw-mbd-kpoint-stress.cpp
/// @brief END-TO-END: multi-k GPW analytic cell stress folds periodic MBD@rsSCS.
///
/// The capability: the k-summed fixed-\(P_k\) stress Lagrangian rebuilds the
/// geometry-only \(E_{\mathrm{disp}}^{\mathrm{MBD}}\) at every strained cell
/// (previously this path refused with a throw). Gates:
///   A. `E_disp` from the multi-k SCF equals the Γ pymbd pin — MBD is
///      independent of the electronic k mesh (Γ pin cross-check).
///   B. `E_disp` at each strained cell (±h per axis, fixed fractional
///      coordinates) matches fresh offline pymbd pins — the strain
///      derivative being folded is externally validated point by point.
///   C. Analytic fixed-\(P_k\) stress vs the multi-SCF strain-FD oracle
///      (`gpw_stress_fd`, MBD folded through the energy path).
///   D. \(\sigma(\mathrm{mbd}) - \sigma(\mathrm{plain})\) equals the pymbd
///      strain-FD dispersion term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\).
///   E. Orthorhombic off-diagonals exactly zero; multi-k payload present;
///      `kStress` unit conversion.
///
/// Public API only at the product path (GPWEvaluator + Evaluate). Offline
/// pymbd pins in REFERENCE.md; no live oracle at ctest time.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_stress.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

constexpr double kAng2Bohr = 1.0 / 0.52917720859;

// ── Offline pure-Python pymbd pins (libmbd 88069c6; REFERENCE.md) ─────────
// Periodic water dimer, 12 Å cube, free-atom volumes, β=0.83, a=6, MBD
// k-mesh 2×2×2. Unstrained value is the same pin gpw-mbd-periodic /
// gpw-mbd-stress use.
constexpr double kEDispRef = -1.4965720704989138e-03; // Ha
// Strained cells: axis scale 1±h at fixed fractional coordinates, h = 2e-3.
constexpr double kStrainH = 2.0e-3;
constexpr double kEDispStrain[3][2] = {
    // {+h, −h} per axis
    {-1.4925684743305467e-03, -1.5005781658326001e-03}, // x
    {-1.4965139646472281e-03, -1.4966306893944292e-03}, // y
    {-1.4957889518587564e-03, -1.4973531298823062e-03}, // z
};
constexpr double kEDispTol = 1.0e-8; // Ha

// C: analytic fixed-P_k vs multi-SCF strain-FD — same tier as the Γ MBD
// stress regression (reg_gpw-mbd-stress).
constexpr double kAnFdTol = 2.0e-5;     // Ha/Bohr³
constexpr double kOffDiagTol = 1.0e-12; // Ha/Bohr³ (must be exactly 0)
constexpr double kDiagFloor = 1.0e-10;  // Ha/Bohr³ — stress must be nonzero
// D: σ(mbd)−σ(plain) vs the pymbd FD dispersion term. E_disp is additive and
// geometry-only, so the electronic Lagrangian cancels; residual is pin
// mismatch (≤ 2·kEDispTol/(2hΩ) ≈ 4e-10) + SCF noise between the two runs.
constexpr double kWireTol = 2.0e-9;  // Ha/Bohr³
constexpr double kWallBudget = 60.0; // s (product path)

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    constexpr double L = 12.0; // Å, PERIODIC XYZ
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double xs[6] = {0.000000, 0.758602, 0.758602,
                          3.000000, 3.758602, 3.758602};
    const double ys[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zs[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    // ── B: E_disp under strain vs fresh pymbd pins (geometry-only) ────────
    std::vector<int> Z(z, z + 6);
    const int mbd_mesh[3] = {2, 2, 2};
    double sigma_mbd_pin[3] = {0.0, 0.0, 0.0};
    const double vol_bohr3 = std::pow(L * kAng2Bohr, 3);
    {
        double max_derr = 0.0;
        for (int a = 0; a < 3; ++a) {
            double e_pm[2] = {0.0, 0.0};
            for (int s = 0; s < 2; ++s) {
                const double scale = 1.0 + (s == 0 ? kStrainH : -kStrainH);
                std::vector<std::array<double, 3>> pos(6);
                double cell[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
                cell[a] *= scale;
                for (int i = 0; i < 6; ++i) {
                    pos[static_cast<std::size_t>(i)] = {xs[i] * kAng2Bohr,
                                                        ys[i] * kAng2Bohr,
                                                        zs[i] * kAng2Bohr};
                    pos[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(a)] *= scale;
                }
                const auto r = cpu::gpw::mbd_rsscs_pbe(
                    Z, pos, cell, /*periodic=*/true, 0.83, 6.0, {}, mbd_mesh);
                e_pm[s] = r.energy;
                const double derr = std::abs(r.energy - kEDispStrain[a][s]);
                max_derr = std::max(max_derr, derr);
            }
            sigma_mbd_pin[a] = -(kEDispStrain[a][0] - kEDispStrain[a][1])
                               / (2.0 * kStrainH) / vol_bohr3;
            std::printf("  axis %d strained E_disp = {%.14e, %.14e} Ha  "
                        "sigma_MBD_pin=%.6e Ha/Bohr3\n",
                        a, e_pm[0], e_pm[1], sigma_mbd_pin[a]);
        }
        std::printf("B max|E_disp(strain) - pymbd| = %.3e Ha\n", max_derr);
        check(max_derr <= kEDispTol, "B strained E_disp vs pymbd pins");
    }

    // ── A + product: multi-k GPW evaluator with MBD, analytic stress ──────
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = xs[i];
        frame.span(kAtom_y)[i] = ys[i];
        frame.span(kAtom_z)[i] = zs[i];
    }
    frame.box = cpu::Box::cubic(L);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator eval({O, H});
    eval.cutoff_ry = 120.0;
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
    eval.k_mesh[0] = 1;
    eval.k_mesh[1] = 1;
    eval.k_mesh[2] = 2; // MP 1×1×2, TR-folded
    eval.mbd = true;
    eval.mbd_k_mesh[0] = eval.mbd_k_mesh[1] = eval.mbd_k_mesh[2] = 2;
    eval.compute_forces = false;
    eval.compute_stress = true;
    eval.stress_use_fd = false;
    eval.stress_strain_h = kStrainH;

    const auto t0 = clock::now();
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
    const auto t1 = clock::now();
    const double wall_product_s =
        std::chrono::duration<double>(t1 - t0).count();

    check(eval.last_result().converged, "A multi-k SCF converged");
    check(eval.last_result().n_kpts >= 1 && !eval.last_result().C_occ_k.empty(),
          "E multi-k C_occ_k payload present");
    const double e_disp = eval.last_result().e_disp;
    std::printf("A E_disp(multi-k) = %.14e Ha (Γ pin %.14e, |d|=%.2e)\n",
                e_disp, kEDispRef, std::abs(e_disp - kEDispRef));
    check(std::isfinite(e_disp) && std::abs(e_disp - kEDispRef) <= kEDispTol,
          "A E_disp equals the Γ pymbd pin (k-mesh independent)");

    const auto& san = eval.last_stress_ha_bohr3();
    for (int k = 3; k < 6; ++k)
        check(std::abs(san[static_cast<std::size_t>(k)]) <= kOffDiagTol,
              "E orthorhombic off-diagonal exactly 0");

    // ── C: multi-SCF strain-FD oracle on the same multi-k method ──────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                             zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, k.basis, k.pot, z[i]));
    }

    auto make_cfg = [&](bool mbd) {
        cpu::gpw::GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = L * kAng2Bohr;
        m.cutoff_ry = 120.0;
        m.n_occ = 8;
        m.max_iter = 120;
        m.tol = 1e-6;
        m.mix = 0.4;
        m.xc = cpu::gpw::XcKind::PBE;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.mbd = mbd;
        if (mbd)
            m.mbd_k_mesh[0] = m.mbd_k_mesh[1] = m.mbd_k_mesh[2] = 2;
        return m;
    };

    const auto t2 = clock::now();
    const auto sfd =
        cpu::gpw::internal::gpw_stress_fd_impl(atoms, make_cfg(true), kStrainH);
    const auto t3 = clock::now();
    const double wall_fd_s = std::chrono::duration<double>(t3 - t2).count();

    double max_diag_err = 0.0;
    double max_diag = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double an = san[static_cast<std::size_t>(a)];
        const double fd = sfd[static_cast<std::size_t>(a)];
        check(std::isfinite(an) && std::isfinite(fd),
              "C finite stress components");
        max_diag_err = std::max(max_diag_err, std::abs(an - fd));
        max_diag = std::max(max_diag, std::abs(an));
        std::printf("  sigma_%s  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], an, fd, std::abs(an - fd));
    }
    check(max_diag > kDiagFloor, "C stress diagonal nonzero");
    std::printf("C max|sigma_an - sigma_fd| = %.3e Ha/Bohr3\n", max_diag_err);
    check(max_diag_err <= kAnFdTol, "C analytic vs multi-SCF FD diagonal");

    // ── D: MBD wiring — sigma(mbd) − sigma(plain) vs the pymbd FD term ────
    const auto s_plain = cpu::gpw::internal::gpw_stress_analytic_impl(
        atoms, make_cfg(false), kStrainH);
    double max_wire = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double shift = san[static_cast<std::size_t>(a)]
                             - s_plain[static_cast<std::size_t>(a)];
        const double d = std::abs(shift - sigma_mbd_pin[a]);
        max_wire = std::max(max_wire, d);
        std::printf("  sigma_%s  shift=% .6e  pymbd_pin=% .6e  |d|=%.3e\n",
                    names[a], shift, sigma_mbd_pin[a], d);
    }
    check(max_wire <= kWireTol,
          "D sigma(mbd)-sigma(plain) equals the pymbd strain-FD term");

    // ── E: Frame kStress unit conversion (eV/Å³) ──────────────────────────
    // kStress is the extensive Voigt virial [eV]: the method's Ha/Bohr^3
    // return stays intensive, so the published value is sigma * V.
    const double kStressVolA3 = frame.box.volume();
    const auto& st = frame.meta.get(kStress);
    const double ha_bohr3_to_ev_a3 =
        units::kHartreeInEv
        / (units::kBohrInAngstrom * units::kBohrInAngstrom
           * units::kBohrInAngstrom);
    bool conv_ok = true;
    for (int k = 0; k < 3; ++k) {
        const double expect =
            san[static_cast<std::size_t>(k)] * ha_bohr3_to_ev_a3 * kStressVolA3;
        if (std::abs(st[static_cast<std::size_t>(k)] - expect)
            > 1e-9 * kStressVolA3)
            conv_ok = false;
    }
    check(conv_ok, "E kStress unit conversion");

    check(wall_product_s <= kWallBudget, "product wall <= 60 s");

    std::printf("gpw-mbd-kpoint-stress  |dE_disp|=%.2e Ha  "
                "max|an-fd|=%.3e  max|wire|=%.3e Ha/Bohr3  iters=%d  nk=%d  "
                "wall_product=%.2f s  wall_fd_twin=%.2f s\n",
                std::abs(e_disp - kEDispRef), max_diag_err, max_wire,
                eval.last_result().iters, eval.last_result().n_kpts,
                wall_product_s, wall_fd_s);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-mbd-kpoint-stress\n");
    return 0;
}
