/// @file cpu/scf/internal/integral_view.h
/// @brief Non-owning view of method-owned integral **fields**.
///
/// Not a pack/bag: does not own storage and cannot be constructed from Frame.
/// Engines borrow matrices/shells from `MeanFieldMethod` private members.
/// Hold by **value** (refs inside); do not store a reference to a temporary
/// view. Lifetime of every bound field must outlive the view.
#pragma once

#include <array>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/ecp_integrals.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/scf_enums.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Const view of integral state owned by a concrete method.
///
/// Parameter group for engines (Fock, gradient, post-HF). Field names match
/// the historical pack surface so call sites read `ints.S` / `ints.shells`
/// without a bag type.
///
/// **Every** storage field is a const reference (or ref-like) into method
/// (or fill-temp) members so a view taken before environment writes still
/// observes later mutations of Hcore / V_nuc / embed_charges / etc.
struct IntegralView {
    const xt::xtensor<double, 2>& S;
    const xt::xtensor<double, 2>& T;
    const xt::xtensor<double, 2>& V;
    const xt::xtensor<double, 2>& Hcore;
    const xt::xtensor<double, 2>& W;
    const xt::xtensor<double, 4>& ERI;
    const double& V_nuc;
    const int& n_basis;
    const int& n_basis_cart;
    const xt::xtensor<double, 2>& c2s;
    const int& n_electrons;
    const std::string& ecp;
    const bool& has_ecp;
    const std::vector<EcpCenter>& ecp_centers;
    const std::vector<cpu::EmbeddingSite>& embed_charges;
    const bool& embed_self_energy;
    const std::array<double, 3>& external_field;
    const double& solvent_eps;
    const double& solvent_radii_scale;
    const double& solvent_x;
    const SolventModel& solvent_model;
    const std::vector<native::Atom>& atoms;
    const std::vector<native::Shell>& shells;
    const std::vector<std::vector<double>>& shell_data;

    int n_occ() const { return n_electrons / 2; }
    bool spherical() const { return c2s.size() > 0; }
    int n_ao_cart() const { return spherical() ? n_basis_cart : n_basis; }

    /// @brief Fused G = J − ½K (RHF stored-ERI path). Requires non-empty ERI.
    void coulomb_exchange_fused(const RealMatrix& P_total, RealMatrix& G) const;

    /// @brief Materialize dense (ij|kl) in the working AO dimension from shells.
    /// @return (n_basis)^4 chemist-notation tensor [Hartree].
    /// @throws std::runtime_error when shells are empty.
    [[nodiscard]] xt::xtensor<double, 4> eri_working() const;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
