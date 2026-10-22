/// @file regressions/gpw/gpw-xdm-kpoint-forces/reg_gpw-xdm-kpoint-forces.cpp
/// @brief Multi-k GPW analytic forces fold free-atom XDM.
///
/// Out-of-library consumer: public GPW headers + internal XDM kernel (same
/// pattern as gpw-xdm-forces / gpw-mbd-kpoint-forces). Geometry and pins in
/// REFERENCE.md (dual pure-Python offline; no live oracle in ctest).

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"
#include "atomiverse/molrs.hpp"

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

// Offline dual pure-Python pins (same as gpw-xdm-forces / gpw-xdm-ice).
constexpr double kEIsoRef = -2.1020435998516220e-03; // Ha
constexpr double kEPerRef = -2.1148157612879029e-03; // Ha
constexpr double kETol = 1.0e-8;                     // Ha
constexpr double kFTol = 1.0e-6;                     // Ha/Bohr
constexpr double kSumFTol = 1.0e-8;                  // Ha/Bohr
constexpr double kPinFTol = 1.0e-10;                 // Ha/Bohr vs dual-Python
constexpr double kWireTol = 1.0e-8;                  // Ha/Bohr

// Isolated free-atom force sample (dual pure-Python ≡ ATV tables).
constexpr double kFIsoPin[6][3] = {
    {1.6875829936003448e-04, 0.0, 0.0},
    {6.9736686381575340e-05, 0.0, -1.3172272218329521e-05},
    {6.9736686381575340e-05, 0.0, 1.3172272218329521e-05},
    {-1.1378113807647229e-04, 0.0, 0.0},
    {-9.7225267023356434e-05, 0.0, -1.5448899346549854e-05},
    {-9.7225267023356434e-05, 0.0, 1.5448899346549854e-05},
};

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::XcKind;
    using cpu::gpw::xdm_pbe;
    using cpu::gpw::XdmResult;

    constexpr double kA2B = 1.0 / 0.52917720859;

    // ── A: geometry-only free-atom XDM dual-gate (water dimer) ───────────
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    const int Zarr[6] = {8, 1, 1, 8, 1, 1};
    std::vector<int> Zw(Zarr, Zarr + 6);
    std::vector<std::array<double, 3>> posw(6);
    for (int i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            posw[static_cast<std::size_t>(i)][d] = ang[i][d] * kA2B;

    const double cell0[3] = {0.0, 0.0, 0.0};
    const XdmResult xdm_iso = xdm_pbe(Zw, posw, cell0, /*periodic=*/false);
    check(std::isfinite(xdm_iso.energy)
              && std::abs(xdm_iso.energy - kEIsoRef) <= kETol,
          "A1 |E_disp isolated - dual-Python pin|");
    check(xdm_iso.forces.size() == 6, "A forces size 6");

    constexpr double kRcut = 50.0 * kA2B;
    const double L = 12.0 * kA2B;
    const double cellp[3] = {L, L, L};
    const XdmResult xdm_per =
        xdm_pbe(Zw, posw, cellp, /*periodic=*/true, kRcut);
    check(std::isfinite(xdm_per.energy)
              && std::abs(xdm_per.energy - kEPerRef) <= kETol,
          "A2 |E_disp periodic - dual-Python pin|");

    double max_pin = 0.0;
    for (std::size_t i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            max_pin = std::max(max_pin,
                               std::abs(xdm_iso.forces[i][d] - kFIsoPin[i][d]));
    check(max_pin <= kPinFTol, "A3 max|F_iso - dual-Python sample|");

    const double h = 1.0e-5;
    auto dual_gate = [&](const XdmResult& xdm, bool periodic,
                         const double cell[3], const char* tag) {
        double max_err = 0.0;
        double sum[3] = {0.0, 0.0, 0.0};
        auto pos_fd = posw;
        for (std::size_t i = 0; i < 6; ++i) {
            for (int d = 0; d < 3; ++d) {
                const double save = pos_fd[i][d];
                pos_fd[i][d] = save + h;
                const double ep =
                    xdm_pbe(Zw, pos_fd, cell, periodic, kRcut).energy;
                pos_fd[i][d] = save - h;
                const double em =
                    xdm_pbe(Zw, pos_fd, cell, periodic, kRcut).energy;
                pos_fd[i][d] = save;
                const double fd = -(ep - em) / (2.0 * h);
                max_err = std::max(max_err, std::abs(fd - xdm.forces[i][d]));
                sum[d] += xdm.forces[i][d];
            }
        }
        const double sum_norm =
            std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        check(max_err <= kFTol, tag);
        check(sum_norm <= kSumFTol, "A |sum F_XDM|");
        std::printf("  %s: max|F-F_fd|=%.2e  |sum F|=%.2e Ha/Bohr\n", tag,
                    max_err, sum_norm);
    };
    dual_gate(xdm_iso, false, cell0, "A4 iso max|F_product-F_fd|");
    dual_gate(xdm_per, true, cellp, "A5 per max|F_product-F_fd|");

    // ── B: multi-k product wiring (H2 1×1×2 + free-atom XDM) ─────────────
    const double rA[3] = {5.5, 6.0, 6.0}; // Bohr
    const double rB[3] = {6.5, 6.0, 6.0}; // Bohr
    cpu::Frame frame(2);
    const double* r[2] = {rA, rB};
    for (int a = 0; a < 2; ++a) {
        frame.span(kAtom_type)[a] = 1;
        frame.span(kAtom_x)[a] = r[a][0] / kA2B;
        frame.span(kAtom_y)[a] = r[a][1] / kA2B;
        frame.span(kAtom_z)[a] = r[a][2] / kA2B;
    }
    frame.box = cpu::Box::cubic(12.0 / kA2B);

    auto make_cfg = [](bool xdm) {
        GpwDft m(
            {cpu::gpw::Kind{molrs::Element::H,
                            GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
                            GthPotential(molrs::Element::H, "GTH-PADE-q1")}},
            100.0, XcKind::PADE);
        m.n_occ = 1;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.eps_default = 1e-8;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.xdm = xdm;
        return m;
    };

    // One SCF + analytic forces with XDM folded (Frame forces in eV/Å).
    GpwDft cfg = make_cfg(true);
    cfg.build(frame);
    const auto& scf = cfg.forces(frame);
    auto frame_forces_au = [&frame]() {
        std::vector<std::array<double, 3>> F(2);
        for (int a = 0; a < 2; ++a)
            F[static_cast<std::size_t>(a)] = {
                frame.span(kAtom_fx)[a] / units::kHartreePerBohrInEvPerAngstrom,
                frame.span(kAtom_fy)[a] / units::kHartreePerBohrInEvPerAngstrom,
                frame.span(kAtom_fz)[a]
                    / units::kHartreePerBohrInEvPerAngstrom};
        return F;
    };
    const auto F_xdm = frame_forces_au();
    check(scf.converged, "B multi-k SCF converged");
    // 1×1×2 + TR may collapse to n_kpts==1; Γ path reports n_kpts==0.
    check(scf.n_kpts >= 1 && !scf.C_occ_k.empty(), "B multi-k C_occ_k payload");
    check(std::isfinite(scf.e_disp) && scf.e_disp < 0.0, "B e_disp finite <0");

    // Plain twin: free-atom XDM is geometry-only (no KS potential), so the
    // plain SCF converges to the same density; its forces are the
    // electronic part the XDM forces are folded onto.
    GpwDft plain = make_cfg(false);
    plain.build(frame);
    plain.forces(frame);
    const auto F_plain = frame_forces_au();
    check(F_xdm.size() == 2 && F_plain.size() == 2, "B two force vectors");

    std::vector<int> Zh = {1, 1};
    std::vector<std::array<double, 3>> posh = {{rA[0], rA[1], rA[2]},
                                               {rB[0], rB[1], rB[2]}};
    const double cellh[3] = {12.0, 12.0, 12.0};
    const XdmResult geom = xdm_pbe(Zh, posh, cellh, /*periodic=*/true, kRcut);

    double max_wire = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double d = (F_xdm[static_cast<std::size_t>(a)][k]
                              - F_plain[static_cast<std::size_t>(a)][k])
                             - geom.forces[static_cast<std::size_t>(a)][k];
            max_wire = std::max(max_wire, std::abs(d));
            check(std::isfinite(F_xdm[static_cast<std::size_t>(a)][k]),
                  "B finite force component");
        }
    }
    check(max_wire <= kWireTol, "B multi-k wiring |F_xdm-F_plain-F_geom|");
    check(std::abs(geom.forces[0][0]) > 1e-8, "B nontrivial XDM F0x");

    std::printf("PASS: gpw-xdm-kpoint-forces  E_iso=%.14e  E_per=%.14e  "
                "max|F_iso-pin|=%.2e  multi-k H2 e_disp=%.6e  max|wire|=%.2e  "
                "n_kpts=%d\n",
                xdm_iso.energy, xdm_per.energy, max_pin, scf.e_disp, max_wire,
                scf.n_kpts);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    return 0;
}
