/// @file cpu/semiempirical/am1_bcc_charges.h
/// @brief Pure molecular AM1-BCC / ABCG2 charge assignment.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/molpy.hpp"

namespace ATV::cpu::semiempirical {

/// AM1 Mulliken populations, topological equivalencing and bond corrections.
///
/// The chemical graph retains its formal charges, bond orders and properties.
/// Assignment returns charges in graph atom order and never changes the graph.
/// The current geometry is used without optimization. To match antechamber's
/// default optimized geometry, optimize separately with AM1 first.
class AM1BCCCharges {
  public:
    /// @param method AM1 method; its charge is the net molecular charge [e].
    /// @param correction "bcc" (AM1-BCC) or "abcg2" (ABCG2).
    explicit AM1BCCCharges(AM1 method = AM1{},
                           std::string_view correction = "bcc");

    /// Compute owned partial charges [e] for one complete chemical molecule.
    /// @throws std::invalid_argument for missing coordinates or connectivity.
    /// @throws std::runtime_error for SCF failure or unsupported chemistry.
    std::vector<double> assign(const molrs::Atomistic& molecule);

    /// AM1 result before equivalencing and bond corrections; borrowed.
    const AM1Result& last_am1_result() const { return am1_.last_result(); }

    /// The selected correction family, independent of GAFF generation.
    const std::string& correction() const { return correction_; }

    /// Base charges are averaged over topologically equivalent atoms.
    bool needs_equivalencing() const { return true; }

  private:
    AM1 am1_;
    std::string correction_;
};

} // namespace ATV::cpu::semiempirical
