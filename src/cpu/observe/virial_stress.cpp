/// @file cpu/observe/virial_stress.cpp
/// @brief Kinetic-stress contribution.

#include "atomiverse/cpu/observe/virial_stress.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {
namespace md {

void Virial::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // Hoisted once before the OMP region -- span() is not safe to call from
    // inside a parallel region (data race + the runtime resolution-budget
    // assertion).
    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    const StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    const StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    const StorageReal* __restrict sp_vz = f.span(kAtom_vz);

    double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;

#pragma omp parallel for schedule(static) reduction(+ : v_xx)                  \
    reduction(+ : v_yy) reduction(+ : v_zz) reduction(+ : v_xy)                \
    reduction(+ : v_xz) reduction(+ : v_yz)
    for (int i = 0; i < n; ++i) {
        const double m = sp_mass[i];
        const double vxi = sp_vx[i];
        const double vyi = sp_vy[i];
        const double vzi = sp_vz[i];
        v_xx += m * vxi * vxi;
        v_yy += m * vyi * vyi;
        v_zz += m * vzi * vzi;
        v_xy += m * vxi * vyi;
        v_xz += m * vxi * vzi;
        v_yz += m * vyi * vzi;
    }

    // mass·v² is in amu·(Å/fs)², the units::mass_v2_to_energy factor brings it to eV.
    // Overwrite (not +=) since this is the canonical kinetic contribution and
    // each integrator step computes it once from current velocities. The pair
    // virial accumulates separately into `frame.meta.get(kStress)`.
    f.meta.get(kStressKinetic)[0] = units::mass_v2_to_energy * v_xx;
    f.meta.get(kStressKinetic)[1] = units::mass_v2_to_energy * v_yy;
    f.meta.get(kStressKinetic)[2] = units::mass_v2_to_energy * v_zz;
    f.meta.get(kStressKinetic)[3] = units::mass_v2_to_energy * v_xy;
    f.meta.get(kStressKinetic)[4] = units::mass_v2_to_energy * v_xz;
    f.meta.get(kStressKinetic)[5] = units::mass_v2_to_energy * v_yz;
}

} // namespace md
} // namespace cpu
} // namespace ATV
