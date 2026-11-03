/// @file cuda/dynamics/fix_efield.cuh
/// @brief CUDA uniform external electric field — LAMMPS `fix efield`
///        constant-field path. CUDA-free header.
///
/// Port of the constant-field charge term of LAMMPS `src/fix_efield.cpp`
/// (`FixEfield::post_force`), checkout SHA e8c4c278a8. LAMMPS: Copyright
/// (2003) Sandia Corporation, GNU General Public License.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/potential.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief CUDA sibling of `cpu::md::FixEfield`: a uniform field on charges.
///
///   \[
///     \mathbf{F}_i \mathrel{+}= q_i \mathbf{E},\qquad
///     U \mathrel{+}= -\sum_i q_i\,\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}},
///     \qquad \mathbf r^{\mathrm u} = \mathbf r + H\,\mathbf n
///   \]
///
/// \(\mathbf E\) in V/Å, \(q\) in e, \(\mathbf F\) in eV/Å, \(U\) in eV
/// (metal `qe2f = 1`). \(H\) is the current device cell (lattice vectors in
/// columns, any shape) and \(\mathbf n\) the device image flags, which the
/// CUDA integrators advance on every wrap. Same energy convention as the CPU
/// class (LAMMPS `fsum[0]`, `fix_modify energy yes`): unbounded under PBC,
/// drifting by the work the field does on migrating ions. **No virial**
/// (LAMMPS `fix_modify virial` is off by default), so `SystemScalars::stress`
/// excludes the field.
///
/// Device-only and graph-capturable: one thread per owned atom, a fixed-point
/// owned force add, and an FP64 block reduction of the energy on steps that
/// request it. Nothing is read back. The field components are baked into the
/// captured launch; `graph_epoch()` changes when they do.
///
/// Registered through `set_evaluator` (force phase) or, with
/// `default_bindings()`, at `post_force @ +100` like the CPU class.
class FixEfield : public cuda::Potential {
  public:
    /// @param Ex Field x-component [V/Å].
    /// @param Ey Field y-component [V/Å].
    /// @param Ez Field z-component [V/Å].
    FixEfield(double Ex, double Ey, double Ez) noexcept
        : Ex(Ex), Ey(Ey), Ez(Ez) {}

    /// Field components [V/Å] — public members, as on `cpu::md::FixEfield`.
    double Ex = 0.0;
    double Ey = 0.0;
    double Ez = 0.0;

    std::string name() const override { return "fix_efield"; }
    bool graph_capturable() const override { return true; }
    std::uint64_t graph_epoch() const override;

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, +100)};
    }

    void launch(const LaunchContext& ctx) override;
    void initialize_forces(const LaunchContext& ctx) override { launch(ctx); }
};

} // namespace md
} // namespace cuda
} // namespace ATV
