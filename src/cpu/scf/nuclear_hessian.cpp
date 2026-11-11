/// @file cpu/scf/nuclear_hessian.cpp
/// @brief Nuclear FD Hessian and analytic-Hessian preconditions.
///        Moved off MeanFieldMethod. cphf.solve / hess_elec are not SCF methods.

#include "atomiverse/cpu/scf/nuclear_hessian.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace scf {

vib::HessianResult central_difference_nuclear_hessian(
    Frame& frame, const std::function<GradientResult(Frame&)>& forces_at) {
    const int natm = static_cast<int>(frame.nlocal());
    const int n3 = 3 * natm;
    // Central FD of the caller's analytic forces. Match the offline PySCF pin
    // step (h = 1e-4 Bohr); convert forces eV/Å → Ha/Bohr for the FD, then
    // scale the Hessian to eV/Å². A fresh solver per ±h displacement (owned by
    // forces_at) avoids DIIS/MO state bleed.
    //   g = −F;  H_ha = ∂g/∂R [Ha/Bohr²];  H_ev = H_ha * scale.
    const double h_bohr = 1e-4;
    const double h_ang = h_bohr * units::kBohrInAngstrom;
    const double f_to_ha_bohr = units::kBohrInAngstrom / units::kHartreeInEv;
    const double hess_to_ev_ang2 =
        units::kHartreeInEv / (units::kBohrInAngstrom * units::kBohrInAngstrom);

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto coord = [&](int A, int d) -> StorageReal& {
        if (d == 0)
            return x[A];
        if (d == 1)
            return y[A];
        return z[A];
    };

    std::vector<double> r0(static_cast<std::size_t>(n3));
    for (int A = 0; A < natm; ++A)
        for (int d = 0; d < 3; ++d)
            r0[static_cast<std::size_t>(3 * A + d)] =
                static_cast<double>(coord(A, d));

    xt::xtensor<double, 2> H = xt::zeros<double>({n3, n3});

    for (int A = 0; A < natm; ++A) {
        for (int d = 0; d < 3; ++d) {
            const int col = 3 * A + d;
            for (int B = 0; B < natm; ++B)
                for (int e = 0; e < 3; ++e)
                    coord(B, e) = static_cast<StorageReal>(
                        r0[static_cast<std::size_t>(3 * B + e)]);
            coord(A, d) = static_cast<StorageReal>(
                r0[static_cast<std::size_t>(col)] + h_ang);
            const GradientResult gp = forces_at(frame);

            coord(A, d) = static_cast<StorageReal>(
                r0[static_cast<std::size_t>(col)] - h_ang);
            const GradientResult gm = forces_at(frame);

            // g = −F; H = ∂g/∂R = −∂F/∂R.
            for (int B = 0; B < natm; ++B)
                for (int e = 0; e < 3; ++e) {
                    const int row = 3 * B + e;
                    const double Fp =
                        static_cast<double>(gp.forces(B, e)) * f_to_ha_bohr;
                    const double Fm =
                        static_cast<double>(gm.forces(B, e)) * f_to_ha_bohr;
                    const double dF_dR = (Fp - Fm) / (2.0 * h_bohr);
                    H(row, col) = (-dF_dR) * hess_to_ev_ang2;
                }
        }
    }

    // Restore geometry to the reference r0.
    for (int A = 0; A < natm; ++A)
        for (int d = 0; d < 3; ++d)
            coord(A, d) = static_cast<StorageReal>(
                r0[static_cast<std::size_t>(3 * A + d)]);

    // Symmetrize (FD noise).
    for (int i = 0; i < n3; ++i)
        for (int j = 0; j < i; ++j) {
            const double s = 0.5 * (H(i, j) + H(j, i));
            H(i, j) = s;
            H(j, i) = s;
        }

    vib::HessianResult out;
    out.n_atoms = natm;
    out.hessian = std::move(H);
    return out;
}

void check_hessian_preconditions(std::string_view method_name, bool converged,
                                 std::string_view stop_reason,
                                 const internal::IntegralView& pack) {
    if (!converged)
        throw std::runtime_error(std::string(method_name)
                                 + ": SCF did not converge — no reference "
                                   "wavefunction to differentiate ("
                                 + std::string(stop_reason) + ")");
    if (pack.spherical())
        throw std::runtime_error(
            std::string(method_name)
            + ": analytic second derivatives are not implemented "
              "for spherical-harmonic basis sets (def2 family) yet — the "
              "derivative c2s transform is a follow-up chain link. Use a "
              "Cartesian-convention basis or the finite-difference "
              "vib::Hessian driver.");
    if (pack.has_ecp)
        throw std::runtime_error(
            std::string(method_name)
            + ": analytic second derivatives are not implemented "
              "for ECP packs — d²U/dR² of the semi-local potential is a "
              "follow-up chain link. Use the finite-difference vib::Hessian "
              "driver.");
}

} // namespace scf
} // namespace cpu
} // namespace ATV
