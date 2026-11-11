/// @file cpu/planewave/internal/ewald.cpp
/// @brief 3D Ewald summation implementation (energy + ionic forces).

#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/core/units.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {
constexpr int virial_a[6] = {0, 1, 2, 0, 0, 1};
constexpr int virial_b[6] = {0, 1, 2, 1, 2, 2};
} // namespace

EwaldSummation::EwaldSummation(const std::vector<Vec3>& pos_ang,
                               const std::vector<AccReal>& charges,
                               const cpu::Box& cell) {
    if (pos_ang.size() != charges.size()
        || charges.size()
               > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Ewald: positions and charges must have "
                                    "matching representable sizes");
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!std::isfinite(cell.cell()(i, j)))
                throw std::invalid_argument("Ewald: cell must be finite");
    if (!std::isfinite(cell.volume()) || cell.volume() <= 0)
        throw std::invalid_argument(
            "Ewald: cell volume must be finite and positive");
    AccReal q_abs = 0, q_total = 0;
    for (std::size_t i = 0; i < charges.size(); ++i) {
        if (!std::isfinite(charges[i]))
            throw std::invalid_argument("Ewald: charges must be finite");
        q_abs += std::abs(charges[i]);
        q_total += charges[i];
        for (int d = 0; d < 3; ++d)
            if (!std::isfinite(pos_ang[i][d]))
                throw std::invalid_argument("Ewald: positions must be finite");
    }
    if (!std::isfinite(q_abs * q_abs))
        throw std::invalid_argument(
            "Ewald: charge scale exceeds the representable range");
    // Convert to Bohr internally
    AccReal a2b = AccReal(1.8897259886);
    cpu::Box cell_bohr;
    cell_bohr.set_cell(cell.cell() * a2b);
    const Mat3& cb = cell_bohr.cell();
    const auto lengths = cell_bohr.lengths();
    const AccReal min_face =
        std::min({cell_bohr.face_height(0), cell_bohr.face_height(1),
                  cell_bohr.face_height(2)});
    const AccReal vol = cell_bohr.volume();
    if (!std::isfinite(vol) || vol <= 0 || !std::isfinite(min_face)
        || min_face <= 0 || !std::isfinite(lengths.norm()))
        throw std::invalid_argument(
            "Ewald: converted cell must be finite and nonsingular");

    const int n = static_cast<int>(charges.size());
    positions_bohr_.resize(static_cast<std::size_t>(n));
    auto& pos = positions_bohr_;
    for (int i = 0; i < n; ++i)
        pos[static_cast<std::size_t>(i)] = {
            pos_ang[static_cast<std::size_t>(i)][0] * a2b,
            pos_ang[static_cast<std::size_t>(i)][1] * a2b,
            pos_ang[static_cast<std::size_t>(i)][2] * a2b};
    for (auto& p : pos) {
        for (int d = 0; d < 3; ++d)
            if (!std::isfinite(p[d]))
                throw std::invalid_argument("Ewald: converted position exceeds "
                                            "the representable range");
        // Periodic image reduction belongs to the same physical owner as the
        // sums. It also bounds every pair's fractional displacement, so the
        // precision-domain image box remains valid for unwrapped MD input.
        cell_bohr.wrap_into(p[0], p[1], p[2],
                            /*image=*/nullptr); // private copy
    }

    forces_bohr_.assign(static_cast<std::size_t>(n), Vec3::zeros());

    // Cell-based splitting balances the two sums; it does not set accuracy.
    AccReal rcut = cell_bohr.lengths().norm() * AccReal(0.5);
    alpha_ = std::sqrt(-std::log(AccReal(2e-17))) / rcut;
    AccReal alpha = alpha_;
    AccReal alpha_sq_4 = AccReal(4) * alpha * alpha;
    AccReal two_alpha_over_sqrt_pi =
        AccReal(2) * alpha / std::sqrt(AccReal(units::pi));

    // QE 7.5 PW/src/ewald.f90:94-103, revision
    // 770a0b2d12928a67048e2f3da8d10d057e52179e.
    // Copyright (C) 2001 PWSCF group, GPL; see QE License.
    // In Hartree, its reciprocal-tail bound is
    // Q_abs^2*alpha/sqrt(pi)*erfc(G_cut/(2*alpha)). Use sum |q| so the
    // bound also applies to neutral mixed-sign systems. The real-space
    // continuum tail has scale pi*Q_abs^2/(volume*alpha^2). A conservative
    // lattice envelope and Gaussian polynomial provide derivative headroom;
    // the same dimensionless Gaussian domain controls both sums and Hessians.
    constexpr AccReal tail_accuracy_ha = AccReal(1e-14);
    const AccReal q2 = q_abs * q_abs;
    const AccReal recip_scale = q2 * alpha / std::sqrt(AccReal(units::pi));
    const AccReal real_scale = AccReal(units::pi) * q2 / (vol * alpha * alpha);
    const AccReal lattice_envelope =
        std::pow(AccReal(1) + AccReal(2) / (alpha * min_face), 3);
    const AccReal scale = std::max(recip_scale, real_scale) * lattice_envelope;
    if (!std::isfinite(alpha) || alpha <= 0 || !std::isfinite(scale))
        throw std::invalid_argument(
            "Ewald: cell/charge scale cannot define finite precision domains");
    AccReal s = AccReal(6);
    while (recip_scale * std::erfc(s) > tail_accuracy_ha
           || scale * (AccReal(1) + s * s * s * s) * std::exp(-s * s)
                  > tail_accuracy_ha)
        s += AccReal(0.25);
    real_cutoff_ = s / alpha;
    reciprocal_cutoff_ = AccReal(2) * alpha * s;
    e_background_ = -AccReal(units::pi) * q_total * q_total
                    / (AccReal(2) * alpha * alpha * vol);
    for (int axis = 0; axis < 3; ++axis)
        virial_ha_[axis] = e_background_;
    if (q_abs == 0)
        return;

    // ── Real space ──
    e_real_ = 0;
    AccReal r_cut = real_cutoff_;
    auto lens = Vec3{cell_bohr.face_height(0), cell_bohr.face_height(1),
                     cell_bohr.face_height(2)};
    const AccReal image_bound =
        std::ceil(r_cut / std::min({lens[0], lens[1], lens[2]}));
    if (!std::isfinite(image_bound)
        || image_bound > AccReal(std::numeric_limits<int>::max() - 2))
        throw std::invalid_argument(
            "Ewald: real-space image domain is not representable");
    int ncell_r = static_cast<int>(image_bound) + 1;

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            AccReal zi = charges[static_cast<std::size_t>(i)];
            AccReal zj = charges[static_cast<std::size_t>(j)];
            AccReal fac = (i == j) ? AccReal(0.5) : AccReal(1);
            for (int hx = -ncell_r; hx <= ncell_r; ++hx)
                for (int hy = -ncell_r; hy <= ncell_r; ++hy)
                    for (int hz = -ncell_r; hz <= ncell_r; ++hz) {
                        // L = hx a + hy b + hz c; columns of cb are lattice
                        // vectors, matching Box H*f.
                        const AccReal Lx = AccReal(hx) * cb(0, 0)
                                           + AccReal(hy) * cb(0, 1)
                                           + AccReal(hz) * cb(0, 2);
                        const AccReal Ly = AccReal(hx) * cb(1, 0)
                                           + AccReal(hy) * cb(1, 1)
                                           + AccReal(hz) * cb(1, 2);
                        const AccReal Lz = AccReal(hx) * cb(2, 0)
                                           + AccReal(hy) * cb(2, 1)
                                           + AccReal(hz) * cb(2, 2);
                        Vec3 dr = {
                            pos[static_cast<std::size_t>(i)][0]
                                - pos[static_cast<std::size_t>(j)][0] + Lx,
                            pos[static_cast<std::size_t>(i)][1]
                                - pos[static_cast<std::size_t>(j)][1] + Ly,
                            pos[static_cast<std::size_t>(i)][2]
                                - pos[static_cast<std::size_t>(j)][2] + Lz};
                        AccReal r = dr.norm();
                        if (r == 0 && i != j && zi != 0 && zj != 0)
                            throw std::invalid_argument(
                                "Ewald: distinct nonzero point charges "
                                "coincide periodically");
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
                            // Affine motion strains both lattice images and
                            // ionic positions. Self images contribute virial
                            // even though their total atomic force is zero.
                            for (int lane = 0; lane < 6; ++lane)
                                virial_ha_[lane] += pref * dr[virial_a[lane]]
                                                    * dr[virial_b[lane]];
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
    AccReal g_cut = reciprocal_cutoff_;
    auto recip = cell_bohr.recip_cell();
    Vec3 b0{recip(0, 0), recip(1, 0), recip(2, 0)};
    Vec3 b1{recip(0, 1), recip(1, 1), recip(2, 1)};
    Vec3 b2{recip(0, 2), recip(1, 2), recip(2, 2)};
    const AccReal reciprocal_bound =
        g_cut
        * std::max({cell_bohr.lengths()[0], cell_bohr.lengths()[1],
                    cell_bohr.lengths()[2]})
        / (AccReal(2) * units::pi);
    if (!std::isfinite(reciprocal_bound)
        || reciprocal_bound > AccReal(std::numeric_limits<int>::max() - 2))
        throw std::invalid_argument(
            "Ewald: reciprocal image domain is not representable");
    int gmax = static_cast<int>(reciprocal_bound) + 1;

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
                // The structure factor is invariant at fixed fractional
                // coordinates. Differentiate 1/Omega, 1/G^2 and the Gaussian.
                const AccReal eg = two_pi_over_vol * sgn * gauss / g2;
                const AccReal metric =
                    AccReal(2) / g2 + AccReal(2) / alpha_sq_4;
                for (int lane = 0; lane < 6; ++lane) {
                    const int a = virial_a[lane], b = virial_b[lane];
                    virial_ha_[lane] += eg
                                        * ((a == b ? AccReal(1) : AccReal(0))
                                           - metric * g[a] * g[b]);
                }

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
