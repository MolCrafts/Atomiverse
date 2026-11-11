/// @file cpu/planewave/internal/pw_operators.h
/// @brief Read-only plane-wave operator views and their table element types.
#pragma once

#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include <complex>
#include <span>
#include <vector>
#include <xtensor/containers/xarray.hpp>

namespace ATV::cpu::pw::internal {

struct KBProjector {
    int l;
    AccReal d;               ///< Diagonalized coupling eigenvalue [Ha].
    xt::xarray<AccReal> f_G; ///< Radial Fourier integral on |G+k|.
    /// Normalized β_lm including the atomic phase, shape [2*l+1, n_g].
    xt::xarray<std::complex<AccReal>> beta_G;
};

struct AtomicOrbital {
    int l;
    xt::xarray<AccReal> chi_G; ///< Radial atomic orbital Fourier integral.
};

using KBProjectors = std::vector<std::vector<KBProjector>>;
using AtomicOrbitals = std::vector<std::vector<AtomicOrbital>>;
using KBProjectorView = std::span<const std::vector<KBProjector>>;
using AtomicOrbitalView = std::span<const std::vector<AtomicOrbital>>;

struct PWKPointView {
    const GVectorList& modes;
    KBProjectorView projectors{};
    AtomicOrbitalView orbitals{};
};

struct PWProjectorView {
    const GVectorList& modes;
    KBProjectorView projectors;
    std::span<const Vec3> positions;
    AccReal volume;
};

struct PWDensityView {
    std::span<const std::complex<AccReal>> total;
    std::span<const xt::xarray<std::complex<AccReal>>> spin;
    int channels;
};

struct PWGridView {
    const GVectorList& modes;
    std::span<const std::complex<AccReal>> local_potential;
    AccReal volume;
};

} // namespace ATV::cpu::pw::internal
