/// @file cpu/vibration/mass_weighting.h
/// @brief cpu::vib::MassWeighting — mass-weights the assembled Hessian.
///
/// Bound at `vib::Phase::diagonalize` immediately after link 1's
/// `Symmetrize` (which runs at order -1) at the default order 0. On the
/// finalize pass it (a) converts the workspace Hessian from the link-1
/// working units (eV/Angstrom^2) to atomic units (Hartree/Bohr^2), and
/// (b) divides element-wise by `sqrt(m_i * m_j)` to form the
/// mass-weighted Hessian `H^mw` in place.
///
/// Masses are the IUPAC-2013 standard atomic weights from
/// `atomic_masses.h`, looked up from each atom's atomic number
/// (`frame.span(kAtom_type)`) — see `cpu::vib::mass_amu`.
///
/// Verbatim port of the `mass_hess` line of PySCF
/// `pyscf/hessian/thermo.py::harmonic_analysis`:
/// @code
///   mass_hess = einsum('pqxy,p,q->pqxy', hess, mass**-.5, mass**-.5)
/// @endcode
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/vibration/hessian_workspace.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Structural component that mass-weights the assembled Hessian.
///
/// `role() == None` and `capabilities()` empty — it is a structural
/// transform on the shared workspace, not a frame-field writer.
class MassWeighting : public cpu::Component {
  public:
    /// @brief Construct bound to a driver-owned workspace.
    /// @param ws Non-owning pointer to the shared Hessian workspace.
    explicit MassWeighting(HessianWorkspace* ws) : ws_(ws) {}

    std::string name() const override { return "vib_mass_weighting"; }
    /* role default default role 0 */

    /// @brief Default binding: `vib::Phase::diagonalize` @ 0.
    ///
    /// Runs after `Symmetrize` (@ -1) and before `HessianDiagonalizer` (@ +1).
    /// @return Single binding placing this at the diagonalize phase.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::diagonalize, 0)};
    }

    /// @brief Convert the Hessian to atomic units and mass-weight it.
    ///
    /// Active only on the finalize pass (`ctx.step >= 6N`), once every
    /// Hessian row has been written and symmetrized. Reads each atom's
    /// atomic number from `frame.span(kAtom_type)`, fetches the IUPAC-2013
    /// mass, then overwrites `ws_->hessian` with
    /// `H^mw_{ia,jb} = H_{ia,jb} * unit_conv / sqrt(m_i m_j)`.
    ///
    /// @param ctx Launch context; `ctx.step` is the displacement index.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    HessianWorkspace* ws_; ///< Non-owning; driver-owned.
};

} // namespace vib
} // namespace cpu
} // namespace ATV
