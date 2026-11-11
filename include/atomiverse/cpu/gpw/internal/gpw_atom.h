/// @file cpu/gpw/internal/gpw_atom.h
/// @brief Engine-private per-atom GPW payload and the `GpwDft` atom-level door.
///
/// `GpwAtom` is the flattened (basis-on-centre + GTH) form every GPW kernel
/// consumes. It is **not** public currency: the public surface is
/// `GpwDft::run(Frame&)` / `forces(Frame&)`, and the method derives these
/// from the Frame (Å → Bohr, Z → `GpwDft::kinds`). Engine code that must
/// drive the solver at a hand-built geometry (the `GPWEvaluator` AIMD
/// branches, strained-cell stress samples, FD oracles, CUDA product shell)
/// goes through `GpwDftAccess`.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {

class GpwDft;
struct GpwResult;

namespace internal {

struct GpwGeometry;
struct GpwWorkspace;

/// @brief One atom's GPW data: valence charge, basis, and GTH pseudopotential.
struct GpwAtom {
    double z_val = 0.0;        ///< valence charge (electrons it contributes).
    int z = 0;                 ///< atomic number Z (for DFT-D3; 0 = unset).
    double pos[3] = {0, 0, 0}; ///< position [Bohr].
    std::vector<GridBasisFunction>
        basis; ///< contracted GTOs, already centered at pos.
    /// @brief Angular momentum ℓ of each entry in `basis` (same length; for DFT+U).
    std::vector<int> basis_l;
    double rloc = 0.0;        ///< GTH local radius.
    std::vector<double> cloc; ///< GTH local short-range C coefficients.
    std::vector<GthNonlocalChannel>
        nonlocal; ///< GTH nonlocal projector channels.

    GpwAtom() = default;

    /// @brief Construct from a contracted basis + GTH potential.
    ///
    /// Converts each contracted shell of `basis` into Cartesian
    /// GridBasisFunctions centered at `pos`, folding the per-primitive Gaussian
    /// normalization N(alpha,L) = (2 alpha/pi)^{3/4} (4 alpha)^{L/2} into the
    /// coefficient (the L alpha-power is what fixes the spanned space to match
    /// CP2K; the constant factor is irrelevant to the energy). Angular
    /// functions are the (2L+1) pure-spherical real solid harmonics from
    /// real_solid_harmonics(L), for any L.
    /// @param pos   Atom position [Bohr]; z_val is taken from `pot.zion`.
    /// @param basis Contracted Gaussian basis to center at `pos`.
    /// @param pot   GTH pseudopotential (local + nonlocal channels).
    /// @param z     Atomic number Z (needed only for DFT-D3, GpwDft.d3).
    GpwAtom(const double pos[3], const GaussianBasis& basis,
            const GthPotential& pot, int z = 0);
};

/// @brief GPW cell [Bohr] from a Frame box [Å].
///
/// GPW carries the cell as three edge lengths and builds its lattice images
/// from them, so a sheared box would be solved on `diag(h)` — a different
/// Bravais lattice, wrong at O(shear). It is refused, not approximated.
/// @param box Frame cell [Å].
/// @return Orthorhombic edge lengths [Bohr].
/// @throws std::invalid_argument when `box` has a non-zero off-diagonal.
std::array<double, 3> cell_from_box(const Box& box);

/// @brief Atom-level door into `GpwDft` for engine-internal callers.
///
/// Every public entry of `GpwDft` takes a `Frame`; these static members
/// reach the private atom-level bodies the Frame entries forward to. Only
/// engine code (GPW internals, `GPWEvaluator`, the CUDA product shell and
/// unit tests of internal kernels) may call them.
struct GpwDftAccess {
    /// @brief Set the orthorhombic cell the next build/run solves in.
    /// @param m Method to update.
    /// @param cell_bohr Box edge lengths [Bohr].
    static void set_cell(GpwDft& m, const std::array<double, 3>& cell_bohr);

    /// @brief Fail-fast configuration check against an atom set (cell must
    ///        already be set).
    /// @param m Method to validate.
    /// @param reference Atoms at the reference geometry [Bohr].
    static void build(GpwDft& m, const std::vector<GpwAtom>& reference);

    /// @brief Atom-level SCF with optional warm starts (see `GpwDft` docs of
    ///        each guess on the private body).
    /// @param m Built method.
    /// @param atoms Geometry [Bohr].
    /// @param guess_C Optional warm-start MO coefficients.
    /// @param guess_P Optional initial density (RKS total / UKS α).
    /// @param guess_P_beta Optional UKS β initial density.
    /// @param guess_P_k Optional multi-k initial densities (RKS / UKS α).
    /// @param guess_P_k_beta Optional multi-k UKS β initial densities.
    /// @return The method's owned result (valid until the next run).
    static GpwResult&
    run(GpwDft& m, const std::vector<GpwAtom>& atoms,
        const linalg::RealMatrix* guess_C = nullptr,
        const linalg::RealMatrix* guess_P = nullptr,
        const linalg::RealMatrix* guess_P_beta = nullptr,
        const std::vector<linalg::CplxMatrix>* guess_P_k = nullptr,
        const std::vector<linalg::CplxMatrix>* guess_P_k_beta = nullptr);

    /// @brief Γ overlap S at `atoms` (basis assembly only).
    /// @param m Built method (Γ path).
    /// @param atoms Geometry [Bohr].
    /// @return S [nb × nb], dimensionless.
    static linalg::RealMatrix overlap(const GpwDft& m,
                                      const std::vector<GpwAtom>& atoms);

    /// @brief Multi-k Bloch overlaps S_k at `atoms`.
    /// @param m Built method (k path).
    /// @param atoms Geometry [Bohr].
    /// @return One S_k [nb × nb] per sampled k.
    static std::vector<linalg::CplxMatrix>
    overlap_k(const GpwDft& m, const std::vector<GpwAtom>& atoms);

    /// @brief Unbuilt copy of every parameter, the kinds and the cell.
    /// @param m Source method.
    /// @return Fresh method; call `build` before `run`.
    static GpwDft clone(const GpwDft& m);

    /// @brief Assemble the atom set at the Frame's geometry (Å → Bohr) from
    ///        `m.kinds` (Z in `kAtom_atomic_number`).
    /// @param m Method owning the kinds.
    /// @param frame Geometry [Å] + Z.
    /// @return One `GpwAtom` per local atom.
    /// @throws std::invalid_argument when a Z has no configured `Kind`.
    static std::vector<GpwAtom> atoms_from_frame(const GpwDft& m,
                                                 const Frame& frame);

    /// @brief Read-only operators and potentials from the most recent run.
    static const GpwGeometry& geometry(const GpwDft& m);
    static const GpwWorkspace& workspace(const GpwDft& m);
};

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
