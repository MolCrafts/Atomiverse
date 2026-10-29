#pragma once
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cuda/planewave/internal/frozen_functional_virial.cuh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
namespace dev = ATV::cuda::pw::internal;
namespace hp = ATV::cpu::pw::internal;
using Complex = std::complex<double>;
inline void require(bool yes, const char* s) {
    if (!yes)
        throw std::runtime_error(s);
}
inline void close(double x, double y, double gate, const char* s) {
    if (!std::isfinite(x) || std::abs(x - y) > gate) {
        std::printf("FAIL %s actual=%.17g ref=%.17g error=%.17g gate=%.17g\n",
                    s, x, y, std::abs(x - y), gate);
        throw std::runtime_error(s);
    }
}
inline bool same(const ATV::cpu::pw::PWResult& a,
                 const ATV::cpu::pw::PWResult& b) {
    auto d = [](double x, double y) {
        return std::bit_cast<std::uint64_t>(x)
               == std::bit_cast<std::uint64_t>(y);
    };
    return d(a.energy, b.energy) && d(a.energy_ha, b.energy_ha)
           && d(a.magnetization, b.magnetization) && d(a.e_fermi, b.e_fermi)
           && d(a.n_elec_occ, b.n_elec_occ) && d(a.e_entropy, b.e_entropy)
           && a.converged == b.converged && a.iterations == b.iterations
           && a.n_kpts == b.n_kpts && a.n_kpts_local == b.n_kpts_local;
}
struct DiagnosticState {
    int owner_calls = 0, scf_calls = 0, budget = 20;
    bool inside = false, reject_certificate = false;
    ATV::cpu::pw::PWResult reference;
    dev::FrozenFunctionalOutcome outcome;
    std::vector<dev::FinalBandStatus> statuses;
    std::shared_ptr<const hp::NcPotentialData> potential;
    hp::GVectorList wave, density;
    double volume = 0;
    std::vector<ATV::Vec3> positions;
    std::vector<Complex> coefficients, rho_ref, rho_orb;
    std::array<std::vector<Complex>, 2> spin_ref, spin_orb;
    std::vector<double> occupations;
    std::array<dev::FrozenOperatorVirial, 7> terms{};
    std::vector<std::array<double, 3>> forces;
    bool want_dump = false;
};
inline DiagnosticState diagnostic;
inline void validate_fourier_density() {
    auto& d = diagnostic;
    const int ng = d.wave.count(), nb = int(d.occupations.size()) / 2;
    std::map<std::array<int, 3>, int> lookup;
    for (int i = 0; i < d.density.count(); ++i)
        lookup[{d.density.h[i], d.density.k[i], d.density.l[i]}] = i;
    std::array<std::vector<Complex>, 2> expected;
    for (auto& r : expected)
        r.resize(d.density.count());
    for (int spin = 0; spin < 2; ++spin)
        for (int b = 0; b < nb; ++b) {
            const int row = spin * nb + b;
            const double f = d.occupations[row];
            require(f == (b < (spin ? 5 : 7) ? 1. : 0.),
                    "actual fixed alpha/beta occupation prefix");
            for (int i = 0; i < ng; ++i)
                for (int j = 0; j < ng; ++j) {
                    auto iter = lookup.find({d.wave.h[i] - d.wave.h[j],
                                             d.wave.k[i] - d.wave.k[j],
                                             d.wave.l[i] - d.wave.l[j]});
                    require(iter != lookup.end(),
                            "density G closes orbital products");
                    expected[spin][iter->second] +=
                        f / d.volume * d.coefficients[row * ng + i]
                        * std::conj(d.coefficients[row * ng + j]);
                }
        }
    for (int spin = 0; spin < 2; ++spin) {
        double gap = 0;
        for (std::size_t i = 0; i < expected[spin].size(); ++i)
            gap = std::max(gap,
                           std::abs(expected[spin][i] - d.spin_orb[spin][i]));
        require(gap < 1e-12, "independent same-spin C/f Fourier density");
        std::printf("SAME_CF spin=%d FourierGap=%.17g\n", spin, gap);
    }
    double gap = 0;
    for (std::size_t i = 0; i < d.rho_orb.size(); ++i)
        gap = std::max(
            gap, std::abs(d.rho_orb[i] - d.spin_orb[0][i] - d.spin_orb[1][i]));
    require(gap < 1e-13, "orbital total density exact spin sum");
}
