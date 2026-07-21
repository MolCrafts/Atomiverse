/// @file cpu/observer/multipole.cpp
/// @brief MultipoleObserver implementation. @see multipole.h

#include "atomiverse/cpu/observer/multipole.h"

#include <cmath>
#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/one_electron.h"

namespace ATV {
namespace cpu {

MultipoleObserver::MultipoleObserver(scf::MeanFieldOutputs handle)
    : handle_(std::move(handle)) {}

void MultipoleObserver::launch(const LaunchContext& ctx) {
    if (!handle_.ready || !*handle_.ready)
        return;
    const auto& det = *handle_.result;
    const auto& shells = *handle_.shells;
    auto& frame = ctx.frame;

    // Origin: center of nuclear charge [Bohr].
    const int natm = frame.nlocal();
    const auto x = frame.span(kAtom_x);
    const auto y = frame.span(kAtom_y);
    const auto z = frame.span(kAtom_z);
    const auto type = frame.span(kAtom_type);
    double zsum = 0.0, origin[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < natm; ++i) {
        const double zi = static_cast<double>(type[i]);
        zsum += zi;
        origin[0] += zi * x[i] * units::kAngstromInBohr;
        origin[1] += zi * y[i] * units::kAngstromInBohr;
        origin[2] += zi * z[i] * units::kAngstromInBohr;
    }
    for (double& c : origin)
        c /= zsum;
    for (int c = 0; c < 3; ++c)
        origin_ang_[c] = origin[c] * units::kBohrInAngstrom;

    // Multipole integrals (Cartesian engine convention), sandwiched to the
    // determinant's AO dimension for spherical-convention bases.
    const scf::MultipoleIntegrals ints{
        shells.data(), static_cast<int>(shells.size()), origin};
    const bool sph = handle_.c2s && handle_.c2s->size() > 0;
    const auto to_ao = [&](const xt::xtensor<double, 2>& M) {
        if (!sph)
            return M;
        const auto& C = *handle_.c2s;
        return xt::xtensor<double, 2>(xt::linalg::dot(
            xt::transpose(C), xt::xtensor<double, 2>(xt::linalg::dot(M, C))));
    };

    // Electronic expectation values via the determinant (single-electron
    // operators only — the expect() contract).
    double m1[3], m2[6];
    for (int a = 0; a < 3; ++a)
        m1[a] = det.expect(to_ao(ints.dipole(a)));
    for (int k = 0; k < 6; ++k)
        m2[k] = det.expect(to_ao(ints.second(k)));

    // Nuclear moments about the same origin.
    double n1[3] = {}, n2[6] = {};
    for (int i = 0; i < natm; ++i) {
        const double zi = static_cast<double>(type[i]);
        const double r[3] = {x[i] * units::kAngstromInBohr - origin[0],
                             y[i] * units::kAngstromInBohr - origin[1],
                             z[i] * units::kAngstromInBohr - origin[2]};
        for (int a = 0; a < 3; ++a)
            n1[a] += zi * r[a];
        const int pk[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
        for (int k = 0; k < 6; ++k)
            n2[k] += zi * r[pk[k][0]] * r[pk[k][1]];
    }

    // μ = μ_nuc − μ_el (electron charge −1).
    for (int a = 0; a < 3; ++a)
        dipole_au_[a] = n1[a] - m1[a];

    // Traceless Θ_ab = ½ Σ q (3 r_a r_b − r² δ_ab), q = +Z nuclei, −1 e⁻.
    const double tr_n = n2[0] + n2[3] + n2[5];
    const double tr_e = m2[0] + m2[3] + m2[5];
    const int diag_of[6] = {1, 0, 0, 1, 0, 1}; // xx,yy,zz carry δ_ab
    for (int k = 0; k < 6; ++k) {
        const double q_ab = (n2[k] - m2[k]);
        const double tr = diag_of[k] ? (tr_n - tr_e) : 0.0;
        quad_au_[k] = 0.5 * (3.0 * q_ab - tr);
    }

    // Frame meta (Debye / D·Å / Å).
    // First launch inserts the keys; later launches overwrite in place.
    if (!frame.meta.contains(kDipole.key)) {
        frame.meta.insert(kDipole, dipole_debye());
        frame.meta.insert(kQuadrupole, quadrupole_buckingham());
        frame.meta.insert(kMultipoleOrigin, origin_ang_);
    } else {
        frame.meta.set(kDipole, dipole_debye());
        frame.meta.set(kQuadrupole, quadrupole_buckingham());
        frame.meta.set(kMultipoleOrigin, origin_ang_);
    }
}

std::array<double, 3> MultipoleObserver::dipole_debye() const {
    std::array<double, 3> d{};
    for (int a = 0; a < 3; ++a)
        d[a] = dipole_au_[a] * units::kAuDipoleInDebye;
    return d;
}

std::array<double, 6> MultipoleObserver::quadrupole_buckingham() const {
    std::array<double, 6> q{};
    const double f = units::kAuDipoleInDebye * units::kBohrInAngstrom;
    for (int k = 0; k < 6; ++k)
        q[k] = quad_au_[k] * f;
    return q;
}

} // namespace cpu
} // namespace ATV
