/// @file cpu/xc/pade.h
/// @brief Goedecker-Teter-Hutter Padé LDA exchange-correlation functional.
///
/// This is CP2K's *native* PADE functional (CP2K evaluates it in
/// `src/xc/xc_pade.F`, NOT through libxc). It is the LDA used together with the
/// GTH-PADE pseudopotentials, and is exactly the functional driving the first
/// GPW parity targets `tests/QS/regtest-gpw-1/Ar.inp` and `H2.inp`. Ported here
/// so the native XC library (spec `native-xc-libxc-parity`, item X2) can
/// reproduce CP2K's PADE energy/potential.
///
/// The functional is a Padé [3/4] fit (Goedecker, Teter, Hutter, PRB 54, 1703
/// (1996); Padé form of the Perdew-Wang/Ceperley-Alder LDA) with a
/// spin-interpolation between the para- and ferromagnetic parameter sets.
///
/// Convention matches the project XCOutput (see cpu/scf/xc_functional.h):
///   - `exc` is the XC energy density PER ELECTRON [Hartree], i.e. E_xc =
///     integral rho(r) exc(r) dr.
///   - `vrho_a`, `vrho_b` are d(rho*exc)/d(rho_sigma) (the XC potential).
#pragma once

namespace ATV {
namespace cpu {
namespace xc {

/// @brief PADE LDA output at a single point (LDA: no gradient channels).
struct PadeXC {
    double exc = 0.0;    ///< XC energy per electron [Hartree] (= -p/q).
    double vrho_a = 0.0; ///< d(rho*exc)/drho_alpha.
    double vrho_b = 0.0; ///< d(rho*exc)/drho_beta.
};

/// @brief Spin-polarized Goedecker-Teter-Hutter Padé LDA at one grid point.
///
/// Returns zero for all channels when the total density is below the project
/// cutoff (rho_total < 1e-12). The unpolarized case is recovered exactly at
/// rho_a == rho_b (spin function and its derivative both vanish).
///
/// @param rho_a Alpha density [electrons/Bohr^3].
/// @param rho_b Beta density [electrons/Bohr^3].
/// @return PadeXC with exc and the two spin-resolved potentials.
PadeXC pade_lsd(double rho_a, double rho_b);

} // namespace xc
} // namespace cpu
} // namespace ATV
