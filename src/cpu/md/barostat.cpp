/// @file cpu/md/barostat.cpp
/// @brief CPU Berendsen isotropic barostat.

#include "atomiverse/cpu/md/barostat.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

Barostat::Barostat(double P_target, double tau_P, double dt,
                   double bulk_modulus)
    : P_target_(P_target), tau_P_(tau_P), dt_(dt), bulk_modulus_(bulk_modulus) {
}

void Barostat::launch(const cpu::LaunchContext& ctx) {
    if (!enabled_)
        return;
    auto& f = ctx.frame;

    // Volume from box (orthorhombic). For multi-rank the box is
    // identical across ranks, so V is the global volume.
    const double V = f.simbox.volume();
    if (V <= 0.0)
        return;

    // LAMMPS-standard scalar pressure P = +trace(σ)/(3V), positive ⇔
    // compressive — `compute pressure::compute_scalar()`:
    //   P = (2·KE + Σ virial) / (3·V).
    // `frame.meta.get(kStress)` holds the pair-virial part Σ r·F; the kinetic
    // part Σ m·v² (trace = 2·KE) lives in `frame.meta.get(kStressKinetic)`.
    // Berendsen runs at `end_of_step`, so both fields are at the same
    // post-force velocity snapshot. This `+trace/(3V)` convention is
    // identical to `NoseHooverNPT` and LAMMPS — the link
    // `lammps-api-align-04` flipped the previous `−trace/(3V)`.
    const double trace_total =
        (f.meta.get(kStress)[0] + f.meta.get(kStress)[1]
         + f.meta.get(kStress)[2])
        + (f.meta.get(kStressKinetic)[0] + f.meta.get(kStressKinetic)[1]
           + f.meta.get(kStressKinetic)[2]);
    const double P = +trace_total / (3.0 * V);

    // Berendsen dilation factor — `fix_press_berendsen.cpp::end_of_step`:
    //   μ = (1 − dt/(Pdamp·β) · (P_target − P))^(1/3)
    // with β = bulk_modulus (LAMMPS `bulkmodulus`). Because the sign of
    // P flipped together with the `(P_target − P)` ordering (was
    // `(P − P_target)` under `P = −trace/(3V)`), the μ³ argument is
    // numerically unchanged for the same physical state — a
    // representation change, not a behaviour change.
    // Clamped to [0.5, 2.0] so a single step cannot collapse or explode
    // the box if P diverges (e.g. transient force spike on step 0).
    const double mu_cubed =
        1.0 - (dt_ / (tau_P_ * bulk_modulus_)) * (P_target_ - P);
    const double mu = std::cbrt(std::max(0.5, std::min(2.0, mu_cubed)));

    // Scale box edges. Single-rank or multi-rank: deterministic, identical
    // computation on every rank (P, V, μ are all global) so the box stays
    // consistent across ranks without any explicit broadcast.
    f.simbox.scale_isotropic(mu);

    // Affine-rescale owned atom positions.
    const int n = owned_count(ctx);
    // Hoisted once for both the owned and ghost rescale loops below.
    StorageReal* __restrict sp_x = f.span(kAtom_x);
    StorageReal* __restrict sp_y = f.span(kAtom_y);
    StorageReal* __restrict sp_z = f.span(kAtom_z);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_x[i] *= mu;
        sp_y[i] *= mu;
        sp_z[i] *= mu;
    }

    // Ghost rows must also track the box transform. The next
    // HaloSend is skin-gated and may skip the refresh; if
    // ghosts retain their pre-scale absolute positions while the box
    // (and owner positions) have moved to the new scale, force
    // evaluators see owner→ghost separations corrupted by
    // `(μ-1)*pos`. Mirrors LAMMPS `fix press/berendsen::end_of_step`
    // calling `domain->x2lamda(nlocal+nghost) / lamda2x(...)` so
    // every atom slot tracks the rescale.
    const int n_total = static_cast<int>(f.nlocal());
    if (n_total > n) {
#pragma omp parallel for schedule(static)
        for (int i = n; i < n_total; ++i) {
            sp_x[i] *= mu;
            sp_y[i] *= mu;
            sp_z[i] *= mu;
        }
    }

    // Keep this rank's slab bounds in lockstep with the rescaled box —
    // same as NoseHooverNPT. Migration classifies owners against
    // `sublo/subhi` (absolute) while `dest_rank_for` uses box-length
    // fractions, and the subdomain-local cell grid derives its extents
    // from these bounds; letting them go stale degrades both as the box
    // breathes.
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1) {
        ctx.runtime.md.dd->compute_sublo_subhi(f.simbox);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
