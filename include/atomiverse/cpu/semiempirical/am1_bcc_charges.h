/// @file cpu/semiempirical/am1_bcc_charges.h
/// @brief AM1-BCC / ABCG2 partial charges — the GAFF / GAFF2 charge model.
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Bond-charge-correction table applied on top of the AM1 charges.
enum class BccParameterSet {
    Bcc,   ///< AM1-BCC, `BCCPARM.DAT` — the GAFF charge model.
    Abcg2, ///< ABCG2, `BCCPARM_ABCG2.DAT` — the GAFF2 charge model.
};

/// @brief AM1-BCC charges (Jakalian, Jack & Bayly, *J. Comput. Chem.* **23**,
///        1623 (2002)) and their ABCG2 refit, as antechamber `-c bcc` /
///        `-c abcg2` compute them.
///
/// The model is three steps:
/// 1. Atomiverse AM1 at the Frame's geometry, which gives the Mulliken
///    (Coulson) base charges.
/// 2. Averaging over topologically equivalent atoms (antechamber `-eq 1`).
/// 3. Bond charge increments looked up by antechamber BCC atom and bond
///    types.
///
/// Steps 2 and 3, including the bond-type perception from connectivity,
/// are molrs `BccModel`.
///
/// The charges belong to the geometry given: nothing here optimises it.
/// antechamber's default optimises with sqm first. To match that, compose
/// a `Minimize` over a mounted `AM1` before calling `assign`.
class AM1BCCCharges {
  public:
    /// @param am1 AM1 method that supplies the base charges. Its `charge`
    ///        member [e] is the net molecular charge; closed-shell singlet only.
    /// @param parameter_set Bond-charge-correction table.
    explicit AM1BCCCharges(
        AM1 am1 = AM1{}, BccParameterSet parameter_set = BccParameterSet::Bcc);

    /// @brief Compute AM1-BCC charges and write them to the Frame.
    ///
    /// Reads `kAtom_type` (atomic number Z), positions [Å], and the bond
    /// topology (`num_bond` / `bond_atom`) of the local atoms. Every bond
    /// partner must be a local atom: the molecule must be whole.
    /// @param frame One molecule; overwrites `kAtom_charge` [e].
    /// @throws std::invalid_argument if a multi-atom Frame carries no bonds,
    ///         or a bond names an atom that is not on the Frame.
    /// @throws std::runtime_error if the AM1 SCF does not converge, or molrs
    ///         cannot type the molecule (an atom or bond with no BCC row).
    void assign(Frame& frame);

    /// @brief AM1 result of the last `assign`: energies and the base
    ///        Mulliken charges [e], before equivalencing and correction.
    /// @return The wrapped method's `last_result()`.
    const AM1Result& last_am1_result() const { return am1_.last_result(); }

    /// @brief The correction table this model applies.
    /// @return The constructor's `parameter_set`.
    BccParameterSet parameter_set() const { return parameter_set_; }

  private:
    AM1 am1_;
    BccParameterSet parameter_set_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
