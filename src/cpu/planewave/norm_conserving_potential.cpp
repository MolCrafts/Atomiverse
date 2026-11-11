/// @file cpu/planewave/norm_conserving_potential.cpp
#include "atomiverse/cpu/planewave/norm_conserving_potential.h"
#include "atomiverse/cpu/planewave/internal/gth_radial.h"
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include "atomiverse/cpu/planewave/internal/upf_reader.h"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace ATV::cpu::pw {
namespace {

// Both built-ins expose the same immutable physical tables. No Fourier or
// electronic state lives here; those belong to the representation consumer.
struct RadialStorage {
    internal::NcPotentialData radial;
    std::vector<::ATV::electronic::NcProjectorView> projectors;
    std::vector<::ATV::electronic::NcAtomicOrbitalView> orbitals;
    explicit RadialStorage(internal::NcPotentialData data)
        : radial(std::move(data)) {
        radial.validate();
        for (const auto& p : radial.kb)
            projectors.push_back(
                {p.l, std::span(p.beta).first(p.radial_points)});
        for (int i = 0; i < radial.nwfc; ++i)
            orbitals.push_back({radial.lchi[i], radial.oc[i], radial.chi[i]});
    }
    ::ATV::electronic::NcProjectorSetView view() const {
        return {radial.r,    radial.rab,      projectors,
                radial.d_ij, radial.rho_atom, orbitals};
    }
};

} // namespace

struct GthPseudopotential::Data {
    gpw::GthPotential analytic;
    RadialStorage samples;
    explicit Data(gpw::GthPotential p)
        : analytic(std::move(p)), samples(internal::sample_gth(analytic)) {}
};

GthPseudopotential::GthPseudopotential(molrs::Element element,
                                       std::string_view name)
    : GthPseudopotential(name.empty() ? throw std::invalid_argument(
                                            "GTH: empty potential name")
                                      : gpw::GthPotential(element, name)) {}
GthPseudopotential::GthPseudopotential(gpw::GthPotential potential)
    : data_(std::make_shared<const Data>(std::move(potential))) {}
GthPseudopotential::~GthPseudopotential() = default;
int GthPseudopotential::atomic_number() const noexcept {
    return static_cast<int>(data_->analytic.element);
}
int GthPseudopotential::valence() const noexcept {
    return data_->analytic.zion;
}
::ATV::electronic::NcProjectorSetView GthPseudopotential::projectors() const {
    return data_->samples.view();
}
void GthPseudopotential::evaluate_local(std::span<const AccReal> radius,
                                        std::span<AccReal> value,
                                        std::span<AccReal> derivative) const {
    internal::gth_local_r(data_->analytic, radius, value, derivative);
}

struct UpfPseudopotential::Data {
    RadialStorage samples;
    std::vector<AccReal> slopes;
    explicit Data(const std::string& path)
        : samples(internal::read_upf_v1(path)) {
        const auto& r = samples.radial.r;
        const auto& v = samples.radial.v_loc;
        slopes.resize(r.size());
        // C1 piecewise cubic Hermite interpolation of the supplied local table.
        // The exact table values, used by PW quadrature, are never smoothed.
        slopes.front() = 0;
        for (std::size_t i = 1; i + 1 < r.size(); ++i) {
            const AccReal a = r[i] - r[i - 1], b = r[i + 1] - r[i];
            slopes[i] = (b * (v[i] - v[i - 1]) / a + a * (v[i + 1] - v[i]) / b)
                        / (a + b);
        }
        slopes.back() = samples.radial.z_val / (r.back() * r.back());
    }
};
UpfPseudopotential::UpfPseudopotential(const std::string& path)
    : data_(path.empty()
                ? throw std::invalid_argument("UPF: empty potential path")
                : std::make_shared<const Data>(path)) {}
UpfPseudopotential::~UpfPseudopotential() = default;
int UpfPseudopotential::atomic_number() const noexcept {
    return data_->samples.radial.atomic_number;
}
int UpfPseudopotential::valence() const noexcept {
    return data_->samples.radial.z_val;
}
::ATV::electronic::NcProjectorSetView UpfPseudopotential::projectors() const {
    return data_->samples.view();
}
void UpfPseudopotential::evaluate_local(std::span<const AccReal> radius,
                                        std::span<AccReal> value,
                                        std::span<AccReal> derivative) const {
    const auto& p = data_->samples.radial;
    for (std::size_t i = 0; i < radius.size(); ++i) {
        const AccReal r = radius[i];
        if (r <= p.r.front()) {
            value[i] = p.v_loc.front();
            derivative[i] = 0;
        } else if (r > p.r.back()) {
            value[i] = -AccReal(p.z_val) / r;
            derivative[i] = AccReal(p.z_val) / (r * r);
            // Preserve the final supplied node and its Coulomb slope. UPF
            // rounding leaves a small residual there; join it C1 to the exact
            // tail over one final mesh interval, rather than introduce a jump.
            const AccReal h = p.r.back() - p.r[p.r.size() - 2];
            const AccReal t = (r - p.r.back()) / h;
            if (t < 1) {
                const AccReal residual =
                    p.v_loc.back() + AccReal(p.z_val) / p.r.back();
                value[i] += residual * (1 - 3 * t * t + 2 * t * t * t);
                derivative[i] += residual * 6 * t * (t - 1) / h;
            }
        } else {
            const auto right = std::upper_bound(p.r.begin(), p.r.end(), r);
            const auto j =
                std::min<std::size_t>(right - p.r.begin() - 1, p.r.size() - 2);
            const AccReal h = p.r[j + 1] - p.r[j], t = (r - p.r[j]) / h;
            const AccReal v0 = p.v_loc[j], v1 = p.v_loc[j + 1];
            const AccReal d0 = data_->slopes[j], d1 = data_->slopes[j + 1];
            value[i] = ((2 * t - 3) * t * t + 1) * v0
                       + ((t - 2) * t + 1) * t * h * d0
                       + (-2 * t + 3) * t * t * v1 + (t - 1) * t * t * h * d1;
            derivative[i] = 6 * t * (t - 1) * (v0 - v1) / h
                            + (3 * t * t - 4 * t + 1) * d0
                            + (3 * t * t - 2 * t) * d1;
        }
    }
}

} // namespace ATV::cpu::pw
