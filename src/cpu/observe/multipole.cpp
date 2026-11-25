/// @file cpu/observe/multipole.cpp
/// @brief MultipoleObserver implementation. @see multipole.h

#include "atomiverse/cpu/observe/multipole.h"

#include <cmath>
#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/one_electron.h"
#include "atomiverse/cpu/units/frame.h"

namespace ATV {
namespace cpu {

MultipoleObserver::MultipoleObserver(scf::MeanFieldOutputs handle)
    : handle_(std::move(handle)) {
    prepare_units_(units::PhysicalUnits::from_style("electronic"));
}
void MultipoleObserver::build(const BuildContext& ctx) {
    prepare_units_(ctx.runtime.units.physical());
}
void MultipoleObserver::prepare_units_(const units::PhysicalUnits& run) {
    const units::Unit bohr{units::length_dimension,
                           1.e-10 * units::kBohrInAngstrom, 0};
    const units::Unit electron{units::charge_dimension,
                               units::elementary_charge_si, 0};
    position_to_bohr_ = units::Conversion(run.length(), bohr);
    dipole_to_run_ =
        units::Conversion(electron * bohr, run.charge() * run.length());
    quadrupole_to_run_ = units::Conversion(
        electron * bohr * bohr, run.charge() * run.length() * run.length());
    origin_to_run_ = units::Conversion(bohr, run.length());
}

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
    const auto type = frame.span(kAtom_atomic_number);
    double zsum = 0.0, origin[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < natm; ++i) {
        const double zi = static_cast<double>(type[i]);
        zsum += zi;
        origin[0] += zi * position_to_bohr_(x[i]);
        origin[1] += zi * position_to_bohr_(y[i]);
        origin[2] += zi * position_to_bohr_(z[i]);
    }
    for (double& c : origin)
        c /= zsum;
    for (int c = 0; c < 3; ++c)
        origin_ang_[c] = origin[c] * units::kBohrInAngstrom;

    // Multipole integrals (Cartesian engine convention), sandwiched to the
    // determinant's AO dimension for spherical-convention bases.
    const scf::MultipoleIntegrals ints{shells.data(),
                                       static_cast<int>(shells.size()), origin};
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
        const double r[3] = {position_to_bohr_(x[i]) - origin[0],
                             position_to_bohr_(y[i]) - origin[1],
                             position_to_bohr_(z[i]) - origin[2]};
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

    MetaF64x3 dipole{}, frame_origin{};
    MetaF64x6 quadrupole{};
    for (int a = 0; a < 3; ++a) {
        dipole[a] = dipole_to_run_(dipole_au_[a]);
        frame_origin[a] = origin_to_run_(origin[a]);
    }
    for (int a = 0; a < 6; ++a)
        quadrupole[a] = quadrupole_to_run_(quad_au_[a]);
    auto put = [&]<class T>(MetaKey<T> key, const T& value) {
        if (frame.meta.contains(key.key))
            frame.meta.set(key, value);
        else
            frame.meta.insert(key, value);
    };
    put(kDipole, dipole);
    put(kQuadrupole, quadrupole);
    put(kMultipoleOrigin, frame_origin);
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
