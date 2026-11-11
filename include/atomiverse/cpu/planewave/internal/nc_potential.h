/// @file cpu/planewave/internal/nc_potential.h
/// @brief Owning scalar norm-conserving radial data used by PW operators.
#pragma once

#include "atomiverse/core/precision.h"
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ATV::cpu {
class Frame;
}

namespace ATV::electronic {
class NormConservingPotential;
}

namespace ATV::cpu::pw::internal {

/// Atom-order assignment of the element providers and their ionic charges.
struct NcComposition {
    std::vector<std::size_t> atom_species;
    std::vector<AccReal> ionic_charges;
    int n_elec = 0;
};

using NcProviders = std::span<
    const std::shared_ptr<const ::ATV::electronic::NormConservingPotential>>;

NcComposition assign_nc_species(const cpu::Frame& frame,
                                NcProviders potentials);

/// Physical radial data only. Fourier tables and electronic state belong to
/// PWSetup/PWWorkspace, never to a shared potential provider.
struct NcPotentialData {
    int atomic_number = 0;
    int z_val = 0;
    int l_max = 0;
    std::vector<AccReal> r;        ///< Radius [Bohr].
    std::vector<AccReal> rab;      ///< dr/di Jacobian [Bohr].
    std::vector<AccReal> v_loc;    ///< Local potential [Ha].
    std::vector<AccReal> rho_atom; ///< Optional 4*pi*r^2*rho [Bohr^-1].
    struct KBProjector {
        int l = 0;
        std::vector<AccReal> beta;     ///< r*p(r) [Bohr^-1/2].
        std::size_t radial_points = 0; ///< Exact integration prefix.
    };
    std::vector<KBProjector> kb;
    std::vector<AccReal> d_ij; ///< Real symmetric full matrix [Ha], row-major.
    std::vector<std::vector<AccReal>> chi; ///< Optional r*R(r) [Bohr^-1/2].
    std::vector<int> lchi;
    std::vector<AccReal> oc;
    int nwfc = 0;

    NcPotentialData() = default;
    explicit NcPotentialData(const ::ATV::electronic::NormConservingPotential&);
    int n_proj() const { return static_cast<int>(kb.size()); }

    /// Validate species, radial shapes, scalar angular channels and couplings.
    void validate() const;
    void validate_radial_mesh() const;
    /// QE Simpson quadrature over the full mesh or an exact prefix.
    AccReal integrate_radial(std::span<const AccReal> values) const;
};

/// Resolve a new geometry against already admitted radial data.
NcComposition assign_nc_species(
    const cpu::Frame& frame,
    std::span<const std::shared_ptr<const NcPotentialData>> species);

} // namespace ATV::cpu::pw::internal
