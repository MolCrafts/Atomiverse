/// @file regressions/semiempirical/dftb0-analytic-forces/reg_dftb0-analytic-forces.cpp
/// @brief DFTB0 closed-form nuclear forces vs offline DFTB+ 24.1 non-SCC.
///
/// oracle: dftb+  oracle_version: 24.1
/// Public API only. Exit non-zero on failure. <= 60 s.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb0.h"
#include "harness/registry.h"

namespace atv_reg_semiempirical_dftb0_analytic_forces {

using namespace ATV;

namespace {

struct Atom {
    int elem;
    double x, y, z;
    double mass;
};

struct Result {
    double energy = 0;
    std::vector<std::array<double, 3>> f;
    double seconds = 0;
};

cpu::Frame make_frame(const std::vector<Atom>& atoms, double box) {
    cpu::Frame f(static_cast<int>(atoms.size()));
    for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
        f.span(kAtom_type)[i] = atoms[static_cast<std::size_t>(i)].elem;
        f.span(kAtom_x)[i] = atoms[static_cast<std::size_t>(i)].x;
        f.span(kAtom_y)[i] = atoms[static_cast<std::size_t>(i)].y;
        f.span(kAtom_z)[i] = atoms[static_cast<std::size_t>(i)].z;
        f.span(kAtom_mass)[i] = atoms[static_cast<std::size_t>(i)].mass;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(box);
    f.meta.get(kEnergy) = 0;
    return f;
}

Result run(cpu::Frame f, bool forces) {
    f.meta.get(kEnergy) = 0;
    const int n = f.nlocal();
    for (int i = 0; i < n; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::semiempirical::DFTB0 dftb0;
    dftb0.with_forces = forces;
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb0));
    eval.build();
    const auto t0 = std::chrono::steady_clock::now();
    eval.run(f);
    const auto t1 = std::chrono::steady_clock::now();
    Result r;
    r.energy = f.meta.get(kEnergy);
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.f.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        r.f[static_cast<std::size_t>(i)] = {
            static_cast<double>(f.span(kAtom_fx)[i]),
            static_cast<double>(f.span(kAtom_fy)[i]),
            static_cast<double>(f.span(kAtom_fz)[i])};
    }
    return r;
}

double energy_at(cpu::Frame f) { return run(std::move(f), false).energy; }

int fail(const char* msg) {
    std::printf("FAIL %s\n", msg);
    return 1;
}

/// Central FD of the public non-SCC energy, h = 1e-4 Å.
int check_fd(const char* tag, const std::vector<Atom>& atoms, double box,
             const Result& analytic, double tol) {
    constexpr double h = 1e-4;
    double max_d = 0;
    for (int a = 0; a < static_cast<int>(atoms.size()); ++a) {
        for (int k = 0; k < 3; ++k) {
            auto plus = atoms;
            auto minus = atoms;
            double* pp = (k == 0)   ? &plus[static_cast<std::size_t>(a)].x
                         : (k == 1) ? &plus[static_cast<std::size_t>(a)].y
                                    : &plus[static_cast<std::size_t>(a)].z;
            double* mm = (k == 0)   ? &minus[static_cast<std::size_t>(a)].x
                         : (k == 1) ? &minus[static_cast<std::size_t>(a)].y
                                    : &minus[static_cast<std::size_t>(a)].z;
            *pp += h;
            *mm -= h;
            const double fi = -(energy_at(make_frame(plus, box))
                                - energy_at(make_frame(minus, box)))
                              / (2.0 * h);
            const double d = std::fabs(analytic.f[static_cast<std::size_t>(a)]
                                                 [static_cast<std::size_t>(k)]
                                       - fi);
            if (d > max_d)
                max_d = d;
        }
    }
    std::printf("%s FD identity max |F_analytic - F_FD| = %.3e eV/A\n", tag,
                max_d);
    if (!(max_d <= tol)) {
        std::printf("FAIL %s FD identity\n", tag);
        return 1;
    }
    return 0;
}

int check_net(const char* tag, const Result& r, double tol) {
    double net[3] = {0, 0, 0};
    for (const auto& fi : r.f)
        for (int k = 0; k < 3; ++k)
            net[k] += fi[static_cast<std::size_t>(k)];
    const double nrm =
        std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]);
    std::printf("%s |sum F| = %.3e eV/A\n", tag, nrm);
    if (!(nrm <= tol) || !std::isfinite(nrm)) {
        std::printf("FAIL %s translational invariance\n", tag);
        return 1;
    }
    return 0;
}

int check_pin(const char* tag, const Result& r, double e_pin,
              const std::vector<std::array<double, 3>>& f_pin, double e_tol,
              double f_tol) {
    std::printf("%s E_ATV=%.10f E_pin=%.10f |dE|=%.3e eV  (%.4f s)\n", tag,
                r.energy, e_pin, std::fabs(r.energy - e_pin), r.seconds);
    if (!std::isfinite(r.energy) || std::fabs(r.energy - e_pin) > e_tol) {
        std::printf("FAIL %s energy\n", tag);
        return 1;
    }
    for (std::size_t a = 0; a < f_pin.size(); ++a) {
        for (int k = 0; k < 3; ++k) {
            const double got = r.f[a][static_cast<std::size_t>(k)];
            const double expect = f_pin[a][static_cast<std::size_t>(k)];
            if (!std::isfinite(got) || std::fabs(got - expect) > f_tol) {
                std::printf("FAIL %s F[%zu][%d]=%.10f expected %.10f\n", tag, a,
                            k, got, expect);
                return 1;
            }
        }
        std::printf("  F[%zu]=(%.8f, %.8f, %.8f) eV/A\n", a, r.f[a][0],
                    r.f[a][1], r.f[a][2]);
    }
    return 0;
}

// DFTB+ 24.1 detailed.out, SCC = No, mio-1-1. Forces are Ha/Bohr.
constexpr double kHa = units::kHartreeInEv;
constexpr double kHaBohr = units::kHartreePerBohrInEvPerAngstrom;

std::array<double, 3> au_to_evA(double x, double y, double z) {
    return {x * kHaBohr, y * kHaBohr, z * kHaBohr};
}

} // namespace

int run() {
    constexpr double kE_tol = 1e-6;   // eV
    constexpr double kF_tol = 1e-5;   // eV/Å
    constexpr double kFd_tol = 1e-5;  // eV/Å
    constexpr double kNet_tol = 1e-8; // eV/Å

    // H2 r = 0.74 Å. Energy is the existing mio-1-1 pin; the force is the
    // same non-SCC transcript (not an equilibrium, |F| ≠ 0).
    const std::vector<Atom> h2_074 = {{1, 0.0, 0.0, 0.0, 1.008},
                                      {1, 0.74, 0.0, 0.0, 1.008}};
    const auto r074 = run(make_frame(h2_074, 30.0), true);
    if (check_pin("H2/0.74", r074, -0.6749509324 * kHa,
                  {au_to_evA(-0.001976354621, 0.0, 0.0),
                   au_to_evA(0.001976354621, 0.0, 0.0)},
                  kE_tol, kF_tol))
        return 1;
    // Sibling reg_dftb0-h2 publishes this rounded eV figure. It must not move.
    if (std::fabs(r074.energy - (-18.36635052)) > 1e-6)
        return fail("H2/0.74 drifted from the published -18.36635052 eV pin");
    if (check_fd("H2/0.74", h2_074, 30.0, r074, kFd_tol))
        return 1;
    if (check_net("H2/0.74", r074, kNet_tol))
        return 1;

    // Stretched H2. Homonuclear, so an SCC run would give the same number;
    // the transcript is SCC=No anyway.
    const std::vector<Atom> h2_090 = {{1, 0.0, 0.0, 0.0, 1.008},
                                      {1, 0.90, 0.0, 0.0, 1.008}};
    const auto r090 = run(make_frame(h2_090, 30.0), true);
    if (check_pin("H2/0.90", r090, -0.6630166170 * kHa,
                  {au_to_evA(0.065975227977, 0.0, 0.0),
                   au_to_evA(-0.065975227977, 0.0, 0.0)},
                  kE_tol, kF_tol))
        return 1;
    if (check_fd("H2/0.90", h2_090, 30.0, r090, kFd_tol))
        return 1;
    if (check_net("H2/0.90", r090, kNet_tol))
        return 1;

    // Off-equilibrium water. Heteronuclear: Δq ≠ 0, so this is not the
    // homonuclear "SCC reduces to DFTB0" licence. SCC=No is required.
    const std::vector<Atom> h2o = {{8, 0.00, 0.00, 0.00, 15.999},
                                   {1, 1.05, 0.00, 0.00, 1.008},
                                   {1, -0.26, 1.00, 0.00, 1.008}};
    const auto rw = run(make_frame(h2o, 30.0), true);
    if (check_pin("H2O", rw, -4.0964033746 * kHa,
                  {au_to_evA(0.039206443065, 0.036588999897, 0.0),
                   au_to_evA(-0.048418613962, -0.000229747051, 0.0),
                   au_to_evA(0.009212170898, -0.036359252846, 0.0)},
                  kE_tol, kF_tol))
        return 1;
    if (check_fd("H2O", h2o, 30.0, rw, kFd_tol))
        return 1;
    if (check_net("H2O", rw, kNet_tol))
        return 1;

    // Tilted CO: ppσ/ppπ blocks and a non-zero transverse force.
    const std::vector<Atom> co = {{6, 0.00, 0.00, 0.00, 12.011},
                                  {8, 1.20, 0.15, 0.00, 15.999}};
    const auto rc = run(make_frame(co, 30.0), true);
    if (check_pin("CO", rc, -5.0270725023 * kHa,
                  {au_to_evA(0.183901804021, 0.022987724205, 0.0),
                   au_to_evA(-0.183901804021, -0.022987724205, 0.0)},
                  kE_tol, kF_tol))
        return 1;
    if (check_fd("CO", co, 30.0, rc, kFd_tol))
        return 1;
    if (check_net("CO", rc, kNet_tol))
        return 1;

    // Min-image pair that crosses the box face. No DFTB+ cluster twin —
    // the gate is F = −∇E of this same min-image energy.
    const std::vector<Atom> wrapped = {{1, 0.20, 0.0, 0.0, 1.008},
                                       {1, 7.60, 0.0, 0.0, 1.008}};
    const auto rwrap = run(make_frame(wrapped, 8.0), true);
    std::printf("wrapped H2 E=%.8f eV F0x=%.8f (%.4f s)\n", rwrap.energy,
                rwrap.f[0][0], rwrap.seconds);
    if (!std::isfinite(rwrap.energy) || !(std::fabs(rwrap.f[0][0]) > 1.0))
        return fail("wrapped H2 force is flat");
    if (check_fd("wrapped-H2", wrapped, 8.0, rwrap, kFd_tol))
        return 1;
    if (check_net("wrapped-H2", rwrap, kNet_tol))
        return 1;

    std::printf("PASS\n");
    return 0;
}

} // namespace atv_reg_semiempirical_dftb0_analytic_forces

ATV_REGRESSION("semiempirical/dftb0", "dftb0-analytic-forces",
               atv_reg_semiempirical_dftb0_analytic_forces::run);
