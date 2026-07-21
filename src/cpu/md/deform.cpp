/// @file cpu/md/deform.cpp
/// @brief CPU uniaxial affine deformation -- LAMMPS `fix deform` aligned.
///
/// ERATE / TRATE formulas and the `L0`-at-fix-issue snapshot are mirrored
/// from `lammps/src/fix_deform.cpp`:
///   * `FixDeform::init()` captures `set[i].lo_start`/`hi_start` from
///     `domain->boxlo/boxhi` at fix-issue time (NOT the simulation-initial
///     box) and sets `lo_stop`/`hi_stop` for ERATE / TRATE.
///   * `FixDeform::apply_strain()` sets the per-step `lo_target`/`hi_target`.
///     For ERATE it linearly interpolates start->stop, which reduces to
///     `L = L0*(1 + R*t_elapsed)`. For TRATE it sets the target directly
///     as `L = L0*exp(R*t_elapsed)`.

#include "atomiverse/cpu/md/deform.h"

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

Deform::Deform(Mode mode, double rate, double dt, int axis)
    : mode_(mode), rate_(rate), dt_(dt), axis_(axis) {}

void Deform::build(const BuildContext& /*ctx*/) {
    // Re-arm the activation snapshot. L0_ is captured on the next launch()
    // so a rebuilt driver references strain to the box at that point --
    // mirroring LAMMPS re-issuing `fix deform` (init() re-snapshots
    // lo_start/hi_start).
    origin_step_ = -1;
    L0_ = 0.0;
}

void Deform::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;

    // Snapshot the deformed-axis reference length on the first launch --
    // the LAMMPS `fix deform` fix-issue-time snapshot (FixDeform::init()
    // capturing lo_start/hi_start). Strain is referenced to the box edge
    // at Deform activation, not the simulation-initial box.
    if (origin_step_ < 0) {
        L0_ = f.simbox.length(axis_);
        origin_step_ = ctx.step;
    }

    // Elapsed time since activation. The target is set FOR the current
    // step, so after the launch at step s the box reflects
    // (s - origin + 1) integrated steps -- matches LAMMPS apply_strain()
    // using delta = ntimestep - beginstep evaluated at the step end.
    const double t_elapsed =
        static_cast<double>(ctx.step - origin_step_ + 1) * dt_;

    // Target edge length for the deformed axis.
    double L_target;
    switch (mode_) {
    case Mode::erate:
        // Engineering strain: L = L0*(1 + R*t).
        L_target = L0_ * (1.0 + rate_ * t_elapsed);
        break;
    case Mode::trate:
        // True (logarithmic) strain: L = L0*exp(R*t).
        L_target = L0_ * std::exp(rate_ * t_elapsed);
        break;
    default:
        L_target = L0_;
        break;
    }

    // Convert the absolute target to a multiplicative scale on the
    // current edge (Box exposes scale_axis, not set_axis). For an
    // unperturbed box this is exact; the affine atom remap below uses
    // the same factor (= LAMMPS `remap x`).
    const double L_prev = f.simbox.length(axis_);
    if (L_prev <= 0.0)
        return;
    const double scale = L_target / L_prev;
    if (scale == 1.0)
        return;

    f.simbox.scale_axis(axis_, scale);

    const int n = owned_count(ctx);
    // Resolve only the deformed axis's column -- matches the original
    // single-array selection (never resolve all three just to pick one).
    StorageReal* __restrict coord;
    switch (axis_) {
    case 0:
        coord = f.span(kAtom_x);
        break;
    case 1:
        coord = f.span(kAtom_y);
        break;
    default:
        coord = f.span(kAtom_z);
        break;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        coord[i] *= scale;
    }

    // Ghost rows must track the box transform too. Without this, force
    // evaluators between this deform call and the next skin-gated
    // HaloSend see owner→ghost separations corrupted by
    // `(scale-1)*pos` on the deformed axis (mirrors the NPT case;
    // LAMMPS `fix deform`'s `remap` also operates on nlocal+nghost).
    // Velocity drift is intentionally omitted — the owner rank applies
    // it; the next forward halo propagates the result.
    const int n_total = static_cast<int>(f.nlocal());
    if (n_total > n) {
#pragma omp parallel for schedule(static)
        for (int i = n; i < n_total; ++i) {
            coord[i] *= scale;
        }
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
