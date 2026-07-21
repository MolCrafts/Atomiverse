/// @file cpu/pw/internal/ewald.cpp
/// @brief 3D Ewald summation implementation (energy + ionic forces).

#include "atomiverse/cpu/pw/internal/ewald.h"
#include "atomiverse/core/units.h"
#include <cmath>
#include <complex>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

EwaldSummation::EwaldSummation(const std::vector<Vec3>& pos_ang,
                               const std::vector<AccReal>& charges,
                               const ::ATV::Box& cell, AccReal ecut) {
    // Convert to Bohr internally
    AccReal a2b = AccReal(1.8897259886);
    Box cell_bohr = cell;
    cell_bohr.cell = cell_bohr.cell * a2b;

    const int n = static_cast<int>(charges.size());
    std::vector<Vec3> pos(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        pos[static_cast<std::size_t>(i)] = {
            pos_ang[static_cast<std::size_t>(i)][0] * a2b,
            pos_ang[static_cast<std::size_t>(i)][1] * a2b,
            pos_ang[static_cast<std::size_t>(i)][2] * a2b};

    forces_bohr_.assign(static_cast<std::size_t>(n), Vec3::zeros());

    AccReal vol = cell_bohr.volume();
    if (vol <= 0 || n == 0)
        return;

    // Ewald alpha: auto-select, calibrated to match QE ewald.f90
    AccReal rcut = cell_bohr.lengths().norm() * AccReal(0.5);
    alpha_ = std::sqrt(-std::log(AccReal(2e-17))) / rcut;
    AccReal alpha = alpha_;
    AccReal alpha_sq_4 = AccReal(4) * alpha * alpha;
    AccReal two_alpha_over_sqrt_pi =
        AccReal(2) * alpha / std::sqrt(AccReal(units::pi));

    // ── Real space ──
    e_real_ = 0;
    AccReal r_cut = AccReal(5) / alpha;
    auto lens = cell_bohr.lengths();
    int ncell_r = static_cast<int>(
                      std::ceil(r_cut / std::min({lens[0], lens[1], lens[2]})))
                  + 1;

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            AccReal zi = charges[static_cast<std::size_t>(i)];
            AccReal zj = charges[static_cast<std::size_t>(j)];
            AccReal fac = (i == j) ? AccReal(0.5) : AccReal(1);
            for (int hx = -ncell_r; hx <= ncell_r; ++hx)
                for (int hy = -ncell_r; hy <= ncell_r; ++hy)
                    for (int hz = -ncell_r; hz <= ncell_r; ++hz) {
                        Vec3 dr = {pos[static_cast<std::size_t>(i)][0]
                                       - pos[static_cast<std::size_t>(j)][0]
                                       + AccReal(hx) * cell_bohr.cell(0, 0)
                                       + AccReal(hy) * cell_bohr.cell(0, 1)
                                       + AccReal(hz) * cell_bohr.cell(0, 2),
                                   pos[static_cast<std::size_t>(i)][1]
                                       - pos[static_cast<std::size_t>(j)][1]
                                       + AccReal(hx) * cell_bohr.cell(1, 0)
                                       + AccReal(hy) * cell_bohr.cell(1, 1)
                                       + AccReal(hz) * cell_bohr.cell(1, 2),
                                   pos[static_cast<std::size_t>(i)][2]
                                       - pos[static_cast<std::size_t>(j)][2]
                                       + AccReal(hx) * cell_bohr.cell(2, 0)
                                       + AccReal(hy) * cell_bohr.cell(2, 1)
                                       + AccReal(hz) * cell_bohr.cell(2, 2)};
                        AccReal r = dr.norm();
                        if (r > AccReal(0) && r < r_cut) {
                            AccReal erfc_ar = std::erfc(alpha * r);
                            e_real_ += fac * zi * zj * erfc_ar / r;

                            // d/dr [erfc(αr)/r] = −erfc/r² − (2α/√π) e^{−α²r²}/r
                            // F_i = −∂E/∂R_i ⇒ along r̂ for the (i,j) term
                            AccReal inv_r = AccReal(1) / r;
                            AccReal inv_r2 = inv_r * inv_r;
                            AccReal dE_dr =
                                -erfc_ar * inv_r2
                                - two_alpha_over_sqrt_pi
                                      * std::exp(-alpha * alpha * r * r)
                                      * inv_r;
                            // force magnitude factor: −fac·zi·zj·dE_dr / r on r_vec
                            AccReal pref =
                                -fac * zi * zj * dE_dr * inv_r; // multiplies dr
                            Vec3 f = {pref * dr[0], pref * dr[1], pref * dr[2]};
                            forces_bohr_[static_cast<std::size_t>(i)] =
                                forces_bohr_[static_cast<std::size_t>(i)] + f;
                            if (i != j) {
                                forces_bohr_[static_cast<std::size_t>(j)] =
                                    forces_bohr_[static_cast<std::size_t>(j)]
                                    - f;
                            }
                        }
                    }
        }
    }

    // ── Reciprocal space ──
    e_recip_ = 0;
    AccReal g_cut = std::sqrt(AccReal(8) * ecut); // 4× E_cut for Ewald recip
    auto recip = cell_bohr.recip_cell();
    Vec3 b0{recip(0, 0), recip(0, 1), recip(0, 2)};
    Vec3 b1{recip(1, 0), recip(1, 1), recip(1, 2)};
    Vec3 b2{recip(2, 0), recip(2, 1), recip(2, 2)};
    int gmax =
        static_cast<int>(g_cut / std::min({b0.norm(), b1.norm(), b2.norm()}))
        + 1;

    const AccReal two_pi_over_vol = AccReal(2) * AccReal(units::pi) / vol;
    const AccReal four_pi_over_vol = AccReal(4) * AccReal(units::pi) / vol;

    for (int hx = -gmax; hx <= gmax; ++hx)
        for (int hy = -gmax; hy <= gmax; ++hy)
            for (int hz = -gmax; hz <= gmax; ++hz) {
                if (hx == 0 && hy == 0 && hz == 0)
                    continue;
                Vec3 g = b0 * AccReal(hx) + b1 * AccReal(hy) + b2 * AccReal(hz);
                AccReal g2 = g.norm_squared();
                if (g2 > g_cut * g_cut)
                    continue;
                std::complex<AccReal> sg(0, 0);
                for (int i = 0; i < n; ++i) {
                    AccReal phase =
                        g[0] * pos[static_cast<std::size_t>(i)][0]
                        + g[1] * pos[static_cast<std::size_t>(i)][1]
                        + g[2] * pos[static_cast<std::size_t>(i)][2];
                    sg += std::complex<AccReal>(
                        charges[static_cast<std::size_t>(i)] * std::cos(phase),
                        -charges[static_cast<std::size_t>(i)]
                            * std::sin(phase));
                }
                AccReal sgn = std::norm(sg);
                AccReal gauss = std::exp(-g2 / alpha_sq_4);
                e_recip_ += sgn * gauss / g2;

                // F_i,α = (4π/Ω) Z_i Σ_G G_α (exp/G²) (Sr sin + Si cos)
                // from E = (2π/Ω) Σ exp/G² |S|²  (see header notes)
                AccReal coeff = four_pi_over_vol * gauss / g2;
                AccReal Sr = sg.real();
                AccReal Si = sg.imag();
                for (int i = 0; i < n; ++i) {
                    AccReal phase =
                        g[0] * pos[static_cast<std::size_t>(i)][0]
                        + g[1] * pos[static_cast<std::size_t>(i)][1]
                        + g[2] * pos[static_cast<std::size_t>(i)][2];
                    AccReal cosph = std::cos(phase);
                    AccReal sinph = std::sin(phase);
                    AccReal zi = charges[static_cast<std::size_t>(i)];
                    AccReal scale = coeff * zi * (Sr * sinph + Si * cosph);
                    forces_bohr_[static_cast<std::size_t>(i)] =
                        forces_bohr_[static_cast<std::size_t>(i)]
                        + Vec3{scale * g[0], scale * g[1], scale * g[2]};
                }
            }
    e_recip_ *= two_pi_over_vol;

    // Self energy (position-independent)
    e_self_ = 0;
    for (int i = 0; i < n; ++i)
        e_self_ -= charges[static_cast<std::size_t>(i)]
                   * charges[static_cast<std::size_t>(i)] * alpha
                   / std::sqrt(AccReal(units::pi));
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
