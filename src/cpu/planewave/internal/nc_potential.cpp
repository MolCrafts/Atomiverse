/// @file cpu/planewave/internal/nc_potential.cpp
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/electronic/norm_conserving_potential.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ATV::cpu::pw::internal {

namespace {
void require_finite(std::span<const AccReal> values, const char* message) {
    for (const auto value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument(message);
}
void validate_frame_species(const cpu::Frame& frame, int z, int valence) {
    if (z < 1 || z > 118 || valence < 1 || valence > z)
        throw std::invalid_argument(
            "NC potential: invalid atomic number or valence");
    if (frame.nlocal() == 0)
        throw std::invalid_argument(
            "PW requires a nonempty single-species Frame");
    const auto types = frame.span(kAtom_atomic_number);
    for (std::size_t i = 0; i < frame.nlocal(); ++i)
        if (types[i] != z)
            throw std::invalid_argument(
                "PW requires every Frame species to match the NC potential "
                "atomic number; mixed species are unsupported");
}

} // namespace

void validate_potential_frame(
    const cpu::Frame& frame,
    const ::ATV::electronic::NormConservingPotential& potential) {
    validate_frame_species(frame, potential.atomic_number(),
                           potential.valence());
}

void validate_potential_frame(const cpu::Frame& frame,
                              const NcPotentialData& potential) {
    validate_frame_species(frame, potential.atomic_number, potential.z_val);
}

NcPotentialData::NcPotentialData(
    const ::ATV::electronic::NormConservingPotential& provider)
    : atomic_number(provider.atomic_number()), z_val(provider.valence()) {
    const auto view = provider.projectors();
    r.assign(view.radius.begin(), view.radius.end());
    rab.assign(view.jacobian.begin(), view.jacobian.end());
    validate_radial_mesh();
    if (view.projectors.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("NC potential: too many projectors");
    for (const auto& projector : view.projectors) {
        kb.push_back({projector.l,
                      {projector.beta.begin(), projector.beta.end()},
                      projector.beta.size()});
        l_max = std::max(l_max, projector.l);
    }
    d_ij.assign(view.coupling.begin(), view.coupling.end());
    rho_atom.assign(view.atomic_density.begin(), view.atomic_density.end());
    for (const auto& orbital : view.atomic_orbitals) {
        chi.emplace_back(orbital.radial.begin(), orbital.radial.end());
        lchi.push_back(orbital.l);
        oc.push_back(orbital.occupation);
    }
    nwfc = static_cast<int>(chi.size());
    v_loc.resize(r.size());
    std::vector<AccReal> derivative(r.size());
    // Validate metadata and views before invoking a user callback.
    validate();
    provider.local_r(r, v_loc, derivative);
    validate();
}

void NcPotentialData::validate() const {
    if (atomic_number < 1 || atomic_number > 118 || z_val < 1
        || z_val > atomic_number)
        throw std::invalid_argument("NC potential: provide a valid atomic "
                                    "number and integral valence in [1,Z]");
    validate_radial_mesh();
    if (l_max < 0 || l_max > 2)
        throw std::invalid_argument(
            "NC potential: scalar angular momenta l<=2 are supported");
    if (v_loc.size() != r.size())
        throw std::invalid_argument(
            "NC potential: local samples must match the radial mesh");
    require_finite(v_loc, "NC potential: local samples must be finite");
    for (const auto& projector : kb) {
        if (projector.l < 0 || projector.l > 2 || projector.l > l_max)
            throw std::invalid_argument(
                "NC potential: unsupported projector angular momentum");
        if (projector.radial_points < 3
            || projector.radial_points > projector.beta.size()
            || projector.beta.size() > r.size())
            throw std::invalid_argument(
                "NC potential: projector support must be an exact radial "
                "prefix with at least three samples");
        require_finite(projector.beta,
                       "NC potential: projector samples must be finite");
    }
    if (d_ij.size() != kb.size() * kb.size())
        throw std::invalid_argument(
            "NC potential: coupling matrix shape mismatch");
    for (std::size_t i = 0; i < kb.size(); ++i)
        for (std::size_t j = 0; j < kb.size(); ++j) {
            const auto value = d_ij[i * kb.size() + j];
            if (!std::isfinite(value) || value != d_ij[j * kb.size() + i])
                throw std::invalid_argument("NC potential: couplings must be "
                                            "finite and real symmetric");
            if (kb[i].l != kb[j].l && value != 0)
                throw std::invalid_argument(
                    "NC potential: different angular momenta cannot couple");
        }
    if (!rho_atom.empty() && rho_atom.size() != r.size())
        throw std::invalid_argument(
            "NC potential: atomic density must match the radial mesh");
    require_finite(rho_atom, "NC potential: atomic density must be finite");
    if (nwfc != static_cast<int>(chi.size()) || chi.size() != lchi.size()
        || chi.size() != oc.size())
        throw std::invalid_argument(
            "NC potential: atomic orbital metadata shape mismatch");
    for (std::size_t i = 0; i < chi.size(); ++i) {
        if (lchi[i] < 0 || lchi[i] > 2 || !std::isfinite(oc[i]) || oc[i] < 0
            || chi[i].size() != r.size())
            throw std::invalid_argument("NC potential: invalid atomic orbital");
        require_finite(chi[i],
                       "NC potential: atomic orbital samples must be finite");
    }
}

void NcPotentialData::validate_radial_mesh() const {
    if (r.size() < 3 || rab.size() != r.size())
        throw std::invalid_argument(
            "NC potential: provide matching radius and Jacobian "
            "arrays with at least three points");
    for (std::size_t i = 0; i < r.size(); ++i)
        if (!std::isfinite(r[i]) || r[i] < 0 || !std::isfinite(rab[i])
            || rab[i] <= 0 || (i && r[i] <= r[i - 1]))
            throw std::invalid_argument("NC potential: radius must be finite, "
                                        "nonnegative and increasing; Jacobian "
                                        "must be finite and positive");
}

AccReal
NcPotentialData::integrate_radial(std::span<const AccReal> values) const {
    // Ported from Quantum ESPRESSO qe-7.5 upflib/simpsn.f90, simpson,
    // revision 770a0b2d12928a67048e2f3da8d10d057e52179e,
    // source blob a9cfb5b33810860f53abbc30a85cb9a8a8618434.
    // Copyright (C) 2001-2012 Quantum ESPRESSO group.
    // GNU General Public License; see QE License and this repository LICENSE.
    // Even meshes use QE's endpoint correction over the full domain.
    validate_radial_mesh();
    const std::size_t n = values.size();
    if (n < 3 || n > r.size())
        throw std::invalid_argument("NC radial integral: provide at least "
                                    "three samples within the radial mesh");
    for (AccReal value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument("NC radial integral: nonfinite sample");
    AccReal sum = 0;
    for (std::size_t i = 1; i + 1 < n; ++i)
        sum += (i % 2 ? AccReal(4) : AccReal(2)) * values[i] * rab[i];
    if (n % 2)
        sum += values[0] * rab[0] + values[n - 1] * rab[n - 1];
    else
        sum += values[0] * rab[0] - AccReal(0.25) * values[n - 3] * rab[n - 3]
               + values[n - 2] * rab[n - 2]
               + AccReal(1.25) * values[n - 1] * rab[n - 1];
    return sum / AccReal(3);
}

} // namespace ATV::cpu::pw::internal
