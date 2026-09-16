/// @file cpu/semiempirical/internal/mio_sk_block.h
/// @brief **PRIVATE WORKSPACE** — the mio-style sketch Slater–Koster two-centre
///        block and its bond-vector derivative. Shared by every DFTB method so
///        the value and the gradient cannot drift apart.
#pragma once

#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

/// @brief One ordered element pair (A,B) of the static mio-style SK tables,
///        evaluated at a bond vector.
///
/// Owns the two-centre Slater–Koster algebra every DFTB method needs: the
/// radial mio-1-1 channels \f$H_{ss\sigma},H_{sp\sigma},H_{pp\sigma},H_{pp\pi}\f$
/// (and their overlap partners) from `MioSkTable`, the direction-cosine
/// rotation into the AO block, and the **analytic derivative of that same
/// block** with respect to the bond vector. All H/C/N/O ordered pairs read the
/// transcribed mio-1-1 grids (no exponential sketch on this path).
///
/// The value routine (@ref add_block) and the derivative routine
/// (@ref set_block_derivative) live on one type on purpose: DFTB2 and DFTB3
/// previously carried private copies of the value kernel, and a third copy of
/// the derivative would have made \f$F=-\nabla E\f$ a coincidence rather than
/// an identity.
///
/// Bond convention: the vector argument is \f$\mathbf{R}=\mathbf{R}_B -
/// \mathbf{R}_A\f$ and the written block is the **directed** A→B block
/// `(ia0+i, ib0+j)`; the transpose is the caller's (lattice sum walks both
/// orders, a min-image walk mirrors).
///
/// Internal — never on a public include path. Units: distances [Å], H⁰ [eV],
/// S dimensionless, derivatives per Å.
class MioSketchPair {
  public:
    /// @brief Bind one ordered element pair and its static SK row.
    /// @param a Element row of atom A (the block's row atom).
    /// @param b Element row of atom B (the block's column atom).
    /// @param sk Legacy sketch row; ignored — radial data come from MioSkTable.
    /// @note Holds references to `constexpr` static table rows — construct at
    ///       the use site, do not outlive the caller's frame.
    MioSketchPair(const basis_data::DftbElementRow& a,
                  const basis_data::DftbElementRow& b,
                  const basis_data::DftbPairSkRow& sk)
        : a_(a), b_(b), sk_(sk) {}

    /// @brief Accumulate the directed A→B \f$H^0\f$ / \f$S\f$ block.
    /// @param dx Bond x component \f$R_{Bx}-R_{Ax}\f$ [Å].
    /// @param dy Bond y component [Å].
    /// @param dz Bond z component [Å].
    /// @param r Bond length [Å]; below 1e-12 the call is a no-op.
    /// @param h0 AO-basis \f$H^0\f$ [eV], accumulated in place.
    /// @param s AO-basis overlap (dimensionless), accumulated in place.
    /// @param ia0 First AO index of atom A.
    /// @param ib0 First AO index of atom B.
    void add_block(double dx, double dy, double dz, double r,
                   linalg::RealMatrix& h0, linalg::RealMatrix& s, int ia0,
                   int ib0) const;

    /// @brief Assign the directed A→B \f$\partial H^0/\partial\mathbf{R}\f$ and
    ///        \f$\partial S/\partial\mathbf{R}\f$ blocks (bond
    ///        \f$\mathbf{R}=\mathbf{R}_B-\mathbf{R}_A\f$).
    /// @param dx Bond x component [Å].
    /// @param dy Bond y component [Å].
    /// @param dz Bond z component [Å].
    /// @param r Bond length [Å]; below 1e-12 the call is a no-op.
    /// @param dh0 Three Cartesian \f$\partial H^0/\partial R_k\f$ [eV/Å];
    ///        **assigned** (not accumulated) on the touched block — the caller
    ///        zeroes them.
    /// @param ds Three Cartesian \f$\partial S/\partial R_k\f$ [1/Å]; same
    ///        assignment contract.
    /// @param ia0 First AO index of atom A.
    /// @param ib0 First AO index of atom B.
    void set_block_derivative(double dx, double dy, double dz, double r,
                              linalg::RealMatrix (&dh0)[3],
                              linalg::RealMatrix (&ds)[3], int ia0,
                              int ib0) const;

  private:
    const basis_data::DftbElementRow& a_;
    const basis_data::DftbElementRow& b_;
    const basis_data::DftbPairSkRow& sk_;
};

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
