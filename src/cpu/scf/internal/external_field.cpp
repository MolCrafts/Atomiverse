/// @file cpu/scf/internal/external_field.cpp
/// @brief ExternalField: multipole dipole kernel + nuclear field energy.

#include "atomiverse/cpu/scf/internal/external_field.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/scf/internal/one_electron.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

constexpr double kFieldZero = 1e-30;

} // namespace

ExternalField::ExternalField(const std::array<double, 3>& field,
                             const std::vector<native::Atom>& atoms)
    : field_(field) {
    for (double f : field_) {
        if (!std::isfinite(f))
            throw std::invalid_argument(
                "ExternalField: non-finite field component [a.u.]");
    }
    if (is_zero()) {
        origin_ = {0.0, 0.0, 0.0};
        return;
    }
    if (atoms.empty())
        throw std::invalid_argument(
            "ExternalField: non-zero field requires at least one nucleus "
            "to define the center-of-nuclear-charge origin");

    double zsum = 0.0;
    origin_ = {0.0, 0.0, 0.0};
    for (const auto& A : atoms) {
        const double z = static_cast<double>(A.Z);
        zsum += z;
        origin_[0] += z * A.pos[0];
        origin_[1] += z * A.pos[1];
        origin_[2] += z * A.pos[2];
    }
    if (zsum == 0.0)
        throw std::invalid_argument(
            "ExternalField: total nuclear charge is zero — center of nuclear "
            "charge is undefined");
    origin_[0] /= zsum;
    origin_[1] /= zsum;
    origin_[2] /= zsum;
}

bool ExternalField::is_zero() const {
    return std::abs(field_[0]) < kFieldZero && std::abs(field_[1]) < kFieldZero
           && std::abs(field_[2]) < kFieldZero;
}

xt::xtensor<double, 2>
ExternalField::core_matrix(const std::vector<native::Shell>& shells) const {
    const int n = static_cast<int>(shells.size());
    xt::xtensor<double, 2> H = xt::zeros<double>({n, n});
    if (is_zero() || n == 0)
        return H;

    const double origin[3] = {origin_[0], origin_[1], origin_[2]};
    const MultipoleIntegrals mi{shells.data(), n, origin};
    // h += F · ⟨r − R₀⟩  (electron charge −1 a.u. → interaction +F·r)
    for (int a = 0; a < 3; ++a) {
        if (std::abs(field_[static_cast<std::size_t>(a)]) < kFieldZero)
            continue;
        H += field_[static_cast<std::size_t>(a)] * mi.dipole(a);
    }
    return H;
}

double ExternalField::nuclear_energy(
    const std::vector<native::Atom>& atoms) const {
    if (is_zero())
        return 0.0;
    double e = 0.0;
    for (const auto& A : atoms) {
        const double z = static_cast<double>(A.Z);
        e -= z
             * (field_[0] * (A.pos[0] - origin_[0])
                + field_[1] * (A.pos[1] - origin_[1])
                + field_[2] * (A.pos[2] - origin_[2]));
    }
    return e;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
