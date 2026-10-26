/// @file electronic/norm_conserving_potential.h
/// @brief Immutable scalar norm-conserving radial potential contract.
#pragma once

#include "atomiverse/core/precision.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>

namespace ATV::electronic {

/// @brief One radial projector. No automatic rescaling or normalization.
struct NcProjectorView {
    int l; ///< Angular momentum; CPU PW currently admits 0, 1 and 2.
    /// beta(r)=r*p_l(r) [Bohr^-1/2], on an exact prefix of radius.
    /// The span ends at the integration cutoff, including its endpoint.
    std::span<const AccReal> beta;
};

/// @brief Optional atomic orbital used only to seed the eigensolver.
struct NcAtomicOrbitalView {
    int l;              ///< Angular momentum, 0 <= l <= 2.
    AccReal occupation; ///< Nonnegative atomic occupation.
    std::span<const AccReal>
        radial; ///< r*R_l(r) [Bohr^-1/2], full radial mesh.
};

/// @brief Borrowed immutable radial data; owned for the provider's lifetime.
///
/// Integrals use QE's Simpson rule on the index coordinate with dr/di in
/// jacobian. For odd n, endpoint weights are 1, interior weights 4,2,...,4;
/// for even n, include the first endpoint, subtract 1/4 at n-3, add 1 at
/// n-2 and 5/4 at n-1 to the ordinary interior sum; divide by 3. Each projector uses its own prefix.
/// This admits uniform, logarithmic and tabulated UPF meshes without guessing
/// quadrature from coordinate differences. Radius may start at zero.
struct NcProjectorSetView {
    std::span<const AccReal>
        radius; ///< Strictly increasing radius [Bohr], >=0.
    std::span<const AccReal> jacobian; ///< Positive dr/di [Bohr], same length.
    std::span<const NcProjectorView> projectors; ///< Radial order of D rows.
    /// Full row-major real symmetric D [Ha], n_projectors squared.
    /// Different l blocks must not couple. p need not have unit radial norm:
    /// the supplied D and beta together define the operator.
    std::span<const AccReal> coupling;
    std::span<const AccReal>
        atomic_density; ///< Optional 4*pi*r^2*rho [Bohr^-1].
    std::span<const NcAtomicOrbitalView> atomic_orbitals; ///< Optional guess.
};

/// @brief One element's immutable scalar NC physics, independent of execution.
///
/// V_NL=sum_{ijm}|p_i Y_lm> D_ij <p_j Y_lm| with normalized real spherical
/// harmonics. CPU PW owns Fourier transforms, phases and force/stress/response
/// contractions. Its radial form factor is integral r*beta*j_l(|G+k|r) dr;
/// the full projector has 4*pi/sqrt(volume), (-i)^l and the atomic Bloch phase.
/// The local G=0 value is 4*pi/volume * integral r*(r*V+Z_val) dr; the divergent
/// Coulomb term is cancelled with the existing Ewald/Hartree convention.
///
/// Providers supply a finite smooth origin (dV/dr=0 at r=0), Coulomb tail
/// -Z_val/r, and a mesh extending past all short-range support. No overlap
/// augmentation, NLCC or spin-orbit channel is represented by this contract.
/// Providers must be safe for concurrent const calls, keep no execution state
/// and retain no caller output spans. CPU PW snapshots this data at setup;
/// no virtual callback runs inside its electronic iteration or matrix loop.
class NormConservingPotential {
  public:
    virtual ~NormConservingPotential() = default;
    /// @brief Chemical identity Z, independent of pseudopotential valence.
    virtual int atomic_number() const noexcept = 0;
    /// @brief Integral number of valence electrons, 0 < valence <= Z.
    virtual int valence() const noexcept = 0;
    /// @brief Physical radial data only; no backend or solver workspace.
    virtual NcProjectorSetView projectors() const = 0;

    /// @brief Evaluate V [Ha] and dV/dr [Ha/Bohr] at radii [Bohr].
    /// Input and the two output spans must not overlap.
    /// @throws std::invalid_argument for invalid radii or output shapes,
    /// before output mutation; std::runtime_error for incomplete output.
    void local_r(std::span<const AccReal> radius, std::span<AccReal> value,
                 std::span<AccReal> derivative) const {
        if (value.size() != radius.size() || derivative.size() != radius.size())
            throw std::invalid_argument(
                "NC potential: local output shape mismatch");
        const auto overlaps = [](std::span<const AccReal> a,
                                 std::span<const AccReal> b) {
            if (a.empty() || b.empty())
                return false;
            const std::less<const AccReal*> before;
            return before(a.data(), b.data() + b.size())
                   && before(b.data(), a.data() + a.size());
        };
        if (overlaps(radius, value) || overlaps(radius, derivative)
            || overlaps(value, derivative))
            throw std::invalid_argument(
                "NC potential: local input and outputs must not overlap");
        for (AccReal r : radius)
            if (!std::isfinite(r) || r < 0)
                throw std::invalid_argument(
                    "NC potential: radius must be finite and nonnegative");
        std::fill(value.begin(), value.end(),
                  std::numeric_limits<AccReal>::quiet_NaN());
        std::fill(derivative.begin(), derivative.end(),
                  std::numeric_limits<AccReal>::quiet_NaN());
        evaluate_local(radius, value, derivative);
        for (std::size_t i = 0; i < radius.size(); ++i)
            if (!std::isfinite(value[i]) || !std::isfinite(derivative[i]))
                throw std::runtime_error(
                    "NC potential: incomplete or nonfinite local output");
    }

  protected:
    /// @brief Implement a checked radial batch; write both outputs completely.
    virtual void evaluate_local(std::span<const AccReal> radius,
                                std::span<AccReal> value,
                                std::span<AccReal> derivative) const = 0;
};

} // namespace ATV::electronic
