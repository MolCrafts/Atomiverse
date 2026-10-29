/// Closed-form reciprocal radial/Cartesian derivatives for NC PW operators.
#pragma once
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include <array>

namespace ATV::cpu::pw::internal {
struct RegularizedBesselJet {
    AccReal value;        // j_l(x)/x^l, including its finite x=0 limit.
    AccReal slope_over_x; // [d/dx(j_l(x)/x^l)]/x, including x=0 limit.
};
RegularizedBesselJet regularized_bessel_jet(int l, AccReal x);

struct LocalFourierJet {
    AccReal value; // v(g)=U(g)/Omega, Ha; G0 is the existing finite part.
    AccReal slope; // dv/dg at fixed Omega, Ha*Bohr; G0 branch returns 0.
};
LocalFourierJet nc_vloc_fourier_jet(const NcPotentialData& data, AccReal g,
                                    AccReal omega);
/// Same local transform without derivative scratch or derivative quadrature.
AccReal nc_vloc_fourier_value(const NcPotentialData& data, AccReal g,
                              AccReal omega);

struct ProjectorRadialJet {
    AccReal value;        // R_l(q)=f_l(q)/q^l.
    AccReal slope_over_q; // R'_l(q)/q; a regular function at q=0.
};
ProjectorRadialJet nc_projector_radial_jet(const NcPotentialData& data,
                                           std::size_t projector, AccReal q);
/// f_l(q), using the same regularized transform as the jet. No slope scratch.
AccReal nc_projector_radial_value(const NcPotentialData& data,
                                  std::size_t projector, AccReal q);

struct ProjectorCartesianJet {
    int components;
    // f_l(q)*Y_lm(qhat), in the exact current compute_ylm m ordering.
    // No atom phase, 4*pi/sqrt(Omega), or (-i)^l is included.
    std::array<AccReal, 5> value{};
    std::array<std::array<AccReal, 3>, 5> gradient{};
};
ProjectorCartesianJet nc_projector_cartesian_jet(const NcPotentialData& data,
                                                 std::size_t projector,
                                                 const Vec3& q);
} // namespace ATV::cpu::pw::internal
