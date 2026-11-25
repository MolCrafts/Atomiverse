/// @file cpu/dynamics/fix_dfield.cpp
#include "atomiverse/cpu/dynamics/fix_dfield.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace md {

void FixDfield::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!f.box.is_orthorhombic()) {
        throw std::runtime_error(
            "FixDfield: triclinic boxes are not supported");
    }
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto q = f.span(kAtom_charge);
    auto fx = f.span(kAtom_fx);
    auto fy = f.span(kAtom_fy);
    auto fz = f.span(kAtom_fz);
    auto img = f.span(kAtom_image);
    const int n = f.nlocal();
    const double Lx = f.box.length(0);
    const double Ly = f.box.length(1);
    const double Lz = f.box.length(2);
    const double V = f.box.volume();
    if (polarization_feedback && !(V > 0.0)) {
        throw std::runtime_error(
            "FixDfield: polarization_feedback requires a periodic cell "
            "with positive volume (E_z = D_z - 4*pi*k_e*P_z is ill-defined)");
    }

    // Pass 1 — itinerant polarization P_z = sum q_i z_i^u / V [e/Å²]
    // (unwrapped z via image flags; Zhang–Sprik constant-D uses the
    // itinerant, not the wrapped, polarization).
    double Pz_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double qi = static_cast<double>(q[i]);
        if (qi == 0.0)
            continue;
        const double iz = static_cast<double>(img[3 * i + 2]);
        Pz_sum += qi * (static_cast<double>(z[i]) + iz * Lz);
    }
    const double Pz = (V > 0.0) ? Pz_sum / V : 0.0;

    // Maxwell field along z: foothold E ≡ D, feedback E = D - 4π k_e P.
    // D and E are in the run's electric-field unit, like FixEfield; their
    // prepared values and 4π k_e P (k_e = the run's Coulomb constant) are
    // forces per charge.
    const double k_e = ctx.runtime.units.coulomb_constant();
    const double four_pi_k = 4.0 * units::pi * k_e;
    const double dx = ctx.runtime.units.electric_field(Dx);
    const double dy = ctx.runtime.units.electric_field(Dy);
    const double dz = ctx.runtime.units.electric_field(Dz);
    const double ez = polarization_feedback ? dz - four_pi_k * Pz : dz;

    // Pass 2 — forces + field-coupling energy.
    double U = 0.0;
    for (int i = 0; i < n; ++i) {
        const double qi = static_cast<double>(q[i]);
        if (qi == 0.0)
            continue;
        // Unwrapped r (orthorhombic)
        const double ix = static_cast<double>(img[3 * i + 0]);
        const double iy = static_cast<double>(img[3 * i + 1]);
        const double iz = static_cast<double>(img[3 * i + 2]);
        const double rx = static_cast<double>(x[i]) + ix * Lx;
        const double ry = static_cast<double>(y[i]) + iy * Ly;
        const double rz = static_cast<double>(z[i]) + iz * Lz;
        fx[i] += static_cast<AccReal>(qi * dx);
        fy[i] += static_cast<AccReal>(qi * dy);
        fz[i] += static_cast<AccReal>(qi * ez);
        U -= qi * (dx * rx + dy * ry);
        if (!polarization_feedback)
            U -= qi * dz * rz;
    }
    if (polarization_feedback) {
        // Constant-D electric enthalpy U_D = V E_z² / (8π k_e) with
        // E_z = D_z - 4π k_e P_z (forces per charge), so
        // F_iz = -∂U_D/∂z_i = q_i E_z exactly (KE + U_D is conserved).
        U += V * ez * ez / (8.0 * units::pi * k_e);
    }
    last_u_ = U;
    last_Pz_ = Pz;
    last_Ez_ = ctx.runtime.units.electric_field.inverse(ez);
    f.meta.get(kEnergy) += U;
}

} // namespace md
} // namespace cpu
} // namespace ATV
