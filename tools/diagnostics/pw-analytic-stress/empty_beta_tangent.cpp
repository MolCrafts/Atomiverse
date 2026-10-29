#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/frozen_functional_virial.h"
#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/pw_stress.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "empty_beta_fixture.h"
#include "empty_beta_value_oracles.h"
#include <cstdio>
#include <optional>
using namespace ATV;
using namespace ATV::cpu::pw;
using namespace ATV::cpu::pw::internal;
using namespace primitive_values;
const PWProblem* accepted_problem = nullptr;
std::optional<FinalBandState> accepted_bands;
FrozenFunctionalVirial accepted_tensor;
int D = 0, extracts = 0, assemblies = 0, legacy = 0;
void real_d(DavidsonSolver*, PWWorkspace&) asm(
    "__real__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver*, PWWorkspace&) asm(
    "__wrap__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver* x, PWWorkspace& w) {
    ++D;
    real_d(x, w);
}
FinalBandState real_f(const PWProblem&, const PW&, const PWResult&, int) asm(
    "__real__ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_"
    "9PWProblemERKNS1_2PWERKNS1_8PWResultEi");
FinalBandState wrap_f(const PWProblem&, const PW&, const PWResult&, int) asm(
    "__wrap__ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_"
    "9PWProblemERKNS1_2PWERKNS1_8PWResultEi");
FinalBandState wrap_f(const PWProblem& p, const PW& m, const PWResult& r,
                      int n) {
    ++extracts;
    accepted_problem = &p;
    auto b = real_f(p, m, r, n);
    accepted_bands = b;
    return b;
}
FrozenFunctionalVirial
real_a(const PWProblem&, const FinalBandState&, cpu::xc::XCType,
       const EwaldSummation&,
       XCBuilder&) asm("__real__ZN3ATV3cpu2pw8internal33assemble_frozen_"
                       "functional_virialERKNS2_9PWProblemERKNS2_"
                       "14FinalBandStateENS0_2xc6XCTypeERKNS2_"
                       "14EwaldSummationERNS2_9XCBuilderE");
FrozenFunctionalVirial
wrap_a(const PWProblem&, const FinalBandState&, cpu::xc::XCType,
       const EwaldSummation&,
       XCBuilder&) asm("__wrap__ZN3ATV3cpu2pw8internal33assemble_frozen_"
                       "functional_virialERKNS2_9PWProblemERKNS2_"
                       "14FinalBandStateENS0_2xc6XCTypeERKNS2_"
                       "14EwaldSummationERNS2_9XCBuilderE");
FrozenFunctionalVirial wrap_a(const PWProblem& p, const FinalBandState& b,
                              cpu::xc::XCType x, const EwaldSummation& e,
                              XCBuilder& s) {
    ++assemblies;
    accepted_tensor = real_a(p, b, x, e, s);
    return accepted_tensor;
}
std::array<double, 6>
real_s(const cpu::Frame&, const PW&, PWResult*, PWProblem*) asm(
    "__real__ZN3ATV3cpu2pw8internal23pw_stress_analytic_implERKNS0_"
    "5FrameERKNS1_2PWEPNS1_8PWResultEPNS2_9PWProblemE");
std::array<double, 6>
wrap_s(const cpu::Frame&, const PW&, PWResult*, PWProblem*) asm(
    "__wrap__ZN3ATV3cpu2pw8internal23pw_stress_analytic_implERKNS0_"
    "5FrameERKNS1_2PWEPNS1_8PWResultEPNS2_9PWProblemE");
std::array<double, 6> wrap_s(const cpu::Frame& f, const PW& m, PWResult* r,
                             PWProblem* p) {
    ++legacy;
    return real_s(f, m, r, p);
}
void need(bool b, const char* m) {
    if (!b)
        throw std::runtime_error(m);
}
Mat3 identity() {
    Mat3 F{};
    for (int i = 0; i < 3; ++i)
        F(i, i) = 1;
    return F;
}
cpu::Frame deform(cpu::Frame f, const Mat3& F) {
    f.box.set_cell(F * f.box.cell());
    for (int i = 0; i < f.nlocal(); ++i) {
        auto x =
            F
            * Vec3{f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]};
        f.span(kAtom_x)[i] = x[0];
        f.span(kAtom_y)[i] = x[1];
        f.span(kAtom_z)[i] = x[2];
    }
    return f;
}
std::vector<C> snapshot(const PWWorkspace& w) {
    std::vector<C> v;
    auto add = [&](const auto& a) {
        for (auto x : a)
            v.emplace_back(x);
    };
    add(w.rho_G);
    add(w.rho_G_old);
    add(w.V_H_G);
    add(w.V_xc_G);
    add(w.V_eff_G);
    add(w.V_loc_G);
    add(w.T_G);
    add(w.wfc);
    add(w.eigenvalues);
    for (const auto& a : w.rho_spin_G)
        add(a);
    for (const auto& a : w.V_xc_spin_G)
        add(a);
    return v;
}
double primal(const cpu::Frame& f, const PWProblem& ref,
              const FinalBandState& s, double scale, XCBuilder& xc) {
    PWProblem p;
    setup_pw_problem_fixed_basis(f, ref, 6, 1, p);
    auto& w = p.ws;
    need(p.gv0.h == ref.gv0.h && p.gv0.k == ref.gv0.k && p.gv0.l == ref.gv0.l
             && w.fft_nx == ref.ws.fft_nx && w.fft_ny == ref.ws.fft_ny
             && w.fft_nz == ref.ws.fft_nz,
         "fixed Miller/FFT");
    for (std::size_t k = 0; k < p.gv_k.size(); ++k)
        need(p.gv_k[k].h == ref.gv_k[k].h && p.gv_k[k].k == ref.gv_k[k].k
                 && p.gv_k[k].l == ref.gv_k[k].l,
             "fixed wave Miller");
    for (std::size_t g = 0; g < s.orbital_density_G.size(); ++g)
        w.rho_G[g] = s.orbital_density_G[g] * scale;
    w.nspin = 2;
    w.rho_spin_G.resize(2);
    for (int c = 0; c < 2; ++c) {
        w.rho_spin_G[c].resize({s.orbital_density_G.size()});
        for (std::size_t g = 0; g < s.orbital_density_G.size(); ++g)
            w.rho_spin_G[c][g] = s.orbital_spin_density_G[c][g] * scale;
    }
    for (auto x : w.rho_spin_G[1])
        need(x == C{}, "beta tangent remains exact0");
    std::size_t g0 = 0;
    for (std::size_t g = 0; g < p.gv0.h.size(); ++g)
        if (!p.gv0.h[g] && !p.gv0.k[g] && !p.gv0.l[g])
            g0 = g;
    need(std::abs(w.rho_G[g0].real() * w.omega - 2) < 1e-8,
         "norm-volume density scale");
    xc.prepare(w);
    xc.step_spin(w, cpu::xc::XCType::LSDA);
    double e = w.E_xc + p.ew_total;
    for (const auto& b : s.local_blocks) {
        FrozenBandView v{{b.coefficients.data(), b.coefficients.size()},
                         b.occupations,
                         b.k_weight};
        e +=
            value_t(p.gv_k[b.ik], v)
            + value_nl(*p.potential, p.gv_k[b.ik], w.atom_pos_bohr, v, w.omega);
    }
    std::span<const C> rho{w.rho_G.data(), w.rho_G.size()};
    return e + value_h(p.gv0, rho, w.omega)
           + value_local(*p.potential, p.gv0, w.atom_pos_bohr, rho, w.omega);
}
int main() {
    try {
        auto f = empty_beta_fixture::frame();
        auto m = empty_beta_fixture::method(cpu::xc::XCType::LSDA);
        auto sigma = m->stress(f);
        (void)sigma;
        need(m->last_result().converged && extracts == 1 && assemblies == 1
                 && legacy == 0 && D == m->last_result().iterations + 1,
             "actual oneSCF oneactive-spin sweep");
        const auto& p = *accepted_problem;
        const auto& b = *accepted_bands;
        const auto before = snapshot(p.ws);
        const auto* fft = p.ws.fft.get();
        const auto* gvec = p.ws.gvec;
        need(accepted_tensor.max_antisymmetric_ha < 1e-9,
             "unprojected antisymmetric tensor evidence");
        double shear = 0;
        for (int a = 0; a < 3; ++a)
            for (int z = 0; z < 3; ++z)
                if (a != z)
                    shear = std::max(
                        shear,
                        std::abs(accepted_tensor.total.virial_ha[3 * a + z]));
        need(shear > 1e-7, "nontrivial shear fixture");
        XCBuilder scratch;
        double e0 = primal(f, p, b, 1, scratch);
        need(std::abs(e0 - accepted_tensor.total.energy_ha) < 1e-11,
             "independent primal internal energy");
        double maximum = 0;
        int points = 0;
        for (double h : {3e-5, 1e-5})
            for (int a = 0; a < 3; ++a)
                for (int z = 0; z < 3; ++z) {
                    auto plus = identity(), minus = identity();
                    plus(a, z) += h;
                    minus(a, z) -= h;
                    double ep = primal(deform(f, plus), p, b, 1 / plus.det(),
                                       scratch),
                           em = primal(deform(f, minus), p, b, 1 / minus.det(),
                                       scratch);
                    points += 2;
                    double fd = -(ep - em) / (2 * h),
                           actual = accepted_tensor.total.virial_ha[3 * a + z],
                           error = std::abs(fd - actual),
                           gate = 1e-8 + 1e-7 * std::abs(actual);
                    std::printf(
                        "EMPTY_BETA_STRAIN lane=%d%d h=%.17g Eplus_Ha=%.17g "
                        "Eminus_Ha=%.17g analytic_W_Ha=%.17g FD_W_Ha=%.17g "
                        "error_Ha=%.17g gate_Ha=%.17g beta_exact0=1 "
                        "fixed_Miller_FFT=1\n",
                        a, z, h, ep, em, actual, fd, error, gate);
                    need(error < gate, "all9 tangent FD original unit gate");
                    maximum = std::max(maximum, error);
                }
        Mat3 q{};
        q(0, 1) = -1;
        q(1, 0) = 1;
        q(2, 2) = 1;
        double er = primal(deform(f, q), p, b, 1, scratch);
        need(std::abs(er - e0) < 1e-10, "independent primal global rotation");
        need(snapshot(p.ws) == before && p.ws.fft.get() == fft
                 && p.ws.gvec == gvec,
             "accepted caches immutable");
        need(D == m->last_result().iterations + 1 && extracts == 1
                 && assemblies == 1 && legacy == 0,
             "primal oracle has zero band/SCF solves");
        std::printf(
            "PASS EMPTY_BETA_TANGENT lanes=9 hcount=2 "
            "independent_strained_primal_evaluations=%d additional_SCF=0 "
            "additional_bands=0 public_SCF=1 public_active_sweep=1 "
            "max_W9_error_Ha=%.17g antisymmetric_Ha=%.17g max_shear_Ha=%.17g "
            "global_rotation_dE_Ha=%.17g accepted_unchanged=1\n",
            points, maximum, accepted_tensor.max_antisymmetric_ha, shear,
            std::abs(er - e0));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL EMPTY_BETA_TANGENT %s\n", e.what());
        return 1;
    }
}
