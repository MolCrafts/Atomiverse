/// @file cpu/vibration/hessian_diagonalizer.h
/// @brief cpu::vib::HessianDiagonalizer — TR projection + diagonalization of the
///        mass-weighted Hessian into normal-mode frequencies.
///
/// Bound at `vib::Phase::diagonalize` at order +1 (last), after link 1's
/// `Symmetrize` (@ -1) and this spec's `MassWeighting` (@ 0). On the
/// finalize pass it faithfully reproduces PySCF
/// `pyscf/hessian/thermo.py::harmonic_analysis`:
///
///   1. Recenter atomic coordinates on the mass center.
///   2. Build the 6 (or 5) mass-weighted translation/rotation vectors
///      (`_get_TR`): translations are `sqrt(m) * unit-axis`; rotations
///      come from the coordinates in the principal-axis frame.
///   3. Classify the rotor (`_get_rotor_type`) on the principal-axis
///      rotational constants — LINEAR keeps only 5 TR vectors.
///   4. QR-orthonormalize the TR vectors, form `P = I - Q Q^T`,
///      diagonalize `P` and keep the complement subspace `bvec`.
///   5. Restrict the mass-weighted Hessian to `bvec`, diagonalize it,
///      yielding exactly `3N-6` (`3N-5` linear) force constants.
///   6. Convert force constants to wavenumbers via `nist_constants.h`
///      (negative for imaginary `lambda < 0`).
///   7. Lift the eigenvectors back to the full `3N` Cartesian basis.
///
/// `cpu::vib::HessianDiagonalizer` is namespace-scoped — distinct from
/// `cpu::scf::internal::HessianDiagonalizer` (a different namespace, no clash).
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

/// @brief Structural component that diagonalizes the mass-weighted
///        Hessian into projected vibrational frequencies.
///
/// `role() == None` and `capabilities()` empty — it writes the
/// frequency fields of the shared workspace, not a `cpu::Frame` field.
class HessianDiagonalizer : public cpu::Component {
  public:
    /// @brief Construct bound to a driver-owned workspace.
    /// @param ws Non-owning pointer to the shared Hessian workspace.
    explicit HessianDiagonalizer(HessianWorkspace* ws) : ws_(ws) {}

    std::string name() const override { return "hessian_diagonalizer"; }
    /* role default default role 0 */

    /// @brief Default binding: `vib::Phase::diagonalize` @ +1 (last).
    /// @return Single binding placing this after `MassWeighting`.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::diagonalize, +1)};
    }

    /// @brief Project out translations/rotations, diagonalize, convert.
    ///
    /// Active only on the finalize pass (`ctx.step >= 6N`), after
    /// `MassWeighting` has produced `H^mw`. Writes `frequencies`,
    /// `normal_modes`, `n_imaginary`, and `is_linear` into the shared
    /// workspace; `cpu::vib::Hessian::run()` copies them into
    /// `HessianResult`.
    ///
    /// @param ctx Launch context; `ctx.step` is the displacement index.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    HessianWorkspace* ws_; ///< Non-owning; driver-owned.
};

} // namespace vib
} // namespace cpu
} // namespace ATV
