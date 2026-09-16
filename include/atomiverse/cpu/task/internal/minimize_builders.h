/// @file cpu/task/internal/minimize_builders.h
/// @brief Typed Minimize params -> the components and criterion they mirror.
///
/// Every optimizer below is built the way its C++ header asks for: the
/// constructor takes the arguments it has, and every remaining public member is
/// assigned afterwards in DECLARATION order — the order the params mirror
/// declares, the codec writes, and a hand-built twin spells. Assigning in that
/// order is what makes a dropped or swapped field a visible diff instead of a
/// silent C++ default.
///
/// They are declared functions rather than inline blocks because each has THREE
/// call sites — the launcher replay's explicit binding and its categorical
/// setter (`src/cpu/task/task_materializer.cpp`), plus the Python handle's
/// `set_optimizer` (`python/src/bindings/bind_minimizer.cpp`) — and a knob
/// reaching only some of them would be invisible until a launcher run diverged
/// from a hand-built or a Python-built one. `LBFGSParams` and `CGParams` need
/// no builder: their fields ARE the constructor arguments.
///
/// `internal`, not public API: these are params-mirror-to-component factories,
/// which the public C++ surface does not have — a user constructs
/// `cpu::minimizer::FIRE` and assigns its members directly.
#pragma once

#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"
#include "atomiverse/cpu/minimize/cell_relax.h"
#include "atomiverse/cpu/minimize/fire.h"
#include "atomiverse/cpu/minimize/joint_cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/minimize/triclinic_cell_relax.h"
#include "atomiverse/cpu/task/task_spec.h"

namespace ATV::cpu::task::internal {

/// @brief Build a `FIRE` with all eight knobs assigned.
/// @param p Typed FIRE params (pseudo-timesteps [fs], mixing and caps).
/// @return Optimizer ready for any registration path.
cpu::minimizer::FIRE tuned_fire(const FIREParams& p);

/// @brief Build an `IsotropicCellRelax` with its two member knobs assigned.
///
/// `P_target` and `alpha0` are the constructor's; `max_scale` and `bulk_mod`
/// are members. The constructor validates the two members against their
/// DEFAULTS, so an out-of-range payload is caught on the assignment's own
/// terms at `launch()` rather than here — the same split the hand-built
/// `regressions/minimizer/cell_lj_pressure` recipe has.
///
/// @param p Typed isotropic relax params (pressure [eV/Å³], gains).
/// @return Optimizer ready for any registration path.
cpu::minimizer::IsotropicCellRelax
tuned_isotropic_cell_relax(const IsotropicCellRelaxParams& p);

/// @brief Build an `AnisotropicCellRelax` with its five member knobs assigned.
/// @param p Typed orthorhombic relax params (pressure [eV/Å³], gains, mask).
/// @return Optimizer ready for any registration path.
cpu::minimizer::AnisotropicCellRelax
tuned_anisotropic_cell_relax(const AnisotropicCellRelaxParams& p);

/// @brief Build a `TriclinicCellRelax` with its member knob assigned.
///
/// Constructor order is `(alpha, bulk)` — the two ends of the mirror's three
/// fields — so only `max_strain` is left to assign.
///
/// @param p Typed full-cell relax params (strain gain, cap, modulus).
/// @return Optimizer ready for any registration path.
cpu::minimizer::TriclinicCellRelax
tuned_triclinic_cell_relax(const TriclinicCellRelaxParams& p);

/// @brief Build a `JointIsotropicRelax` with its three member knobs assigned.
/// @param p Typed joint relax params (pressure [eV/Å³], atom step [Å], gains).
/// @return Optimizer ready for any registration path.
cpu::minimizer::JointIsotropicRelax
tuned_joint_isotropic_relax(const JointIsotropicRelaxParams& p);

/// @brief Attach the task's single stop policy to the minimizer.
///
/// `Minimize::install_defaults` supplies its own `ForceConverged` only when no
/// criterion was added, so naming one here replaces it — which is the whole
/// point of the v5 variant: a variable-cell run terminates on pressure or
/// stress, never on forces.
///
/// The criterion is CONSTRUCTED here, so its own invariants (`ptol > 0`,
/// `stol > 0`, at least one unmasked axis) throw at this call rather than at
/// `build()` — which is what lets the Python handle report them at the
/// `add_criterion(...)` line that caused them.
///
/// @param driver    Target minimizer driver, still unbuilt.
/// @param criterion The typed stop policy carried by the `MinimizeTask`.
void add_minimize_criterion(minimizer::Minimize& driver,
                            const MinimizeCriterionParams& criterion);

} // namespace ATV::cpu::task::internal
