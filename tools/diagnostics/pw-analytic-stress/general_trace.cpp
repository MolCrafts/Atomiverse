#include "general_trace.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/frozen_functional_virial.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/pw_forces.h"
#include "atomiverse/cpu/planewave/internal/pw_stress.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "pw_density_domain_audit.h"
#include <cstdio>
#include <limits>
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
namespace trace {
Counts counts;
SharedBands shared_bands;
} // namespace trace
using namespace ATV;
using namespace ATV::cpu::pw;
using namespace ATV::cpu::pw::internal;
namespace {
std::vector<std::complex<AccReal>> bands_payload(const FinalBandState& s) {
    std::vector<std::complex<AccReal>> out;
    for (const auto& b : s.local_blocks) {
        out.emplace_back(b.spin, b.ik);
        out.emplace_back(b.k_weight, b.occupations.size());
        for (auto c : b.coefficients)
            out.push_back(c);
        for (auto f : b.occupations)
            out.emplace_back(f);
    }
    return out;
}
} // namespace
void real_d(DavidsonSolver*, PWWorkspace&) asm(
    "__real__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver*, PWWorkspace&) asm(
    "__wrap__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver* s, PWWorkspace& w) {
    ++trace::counts.davidson;
    real_d(s, w);
}
FinalBandState real_f(const PWProblem&, const PW&, const PWResult&, int) asm(
    "__real__ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_"
    "9PWProblemERKNS1_2PWERKNS1_8PWResultEi");
FinalBandState wrap_f(const PWProblem&, const PW&, const PWResult&, int) asm(
    "__wrap__ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_"
    "9PWProblemERKNS1_2PWERKNS1_8PWResultEi");
FinalBandState wrap_f(const PWProblem& p, const PW& m, const PWResult& r,
                      int n) {
    // A failed extract must not leave the previous state/payload or MP tag live.
    trace::shared_bands = trace::SharedBands{};
    ++trace::counts.extract;
    auto result = real_f(p, m, r, n);
    auto& observed = trace::shared_bands;
    observed.mp_active = m.smearing == electronic::Smearing::methfessel_paxton;
    if (observed.mp_active) {
        observed.mp_order = m.mp_order;
        observed.mp_width_ha = m.degauss * qe_units::rydberg_to_hartree;
        observed.mp_target_ha = m.tol * .1;
        observed.mp_reference_energy_ha = r.energy_ha;
    }
    return result;
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
    ++trace::counts.assembly;
    auto& observed = trace::shared_bands;
    const bool is_mp = observed.mp_active;
    const int order = observed.mp_order;
    const double width = observed.mp_width_ha, target = observed.mp_target_ha;
    const double reference_energy = observed.mp_reference_energy_ha;
    observed = trace::SharedBands{};
    if (is_mp) {
        observed.mp_active = true;
        observed.mp_order = order;
        observed.mp_width_ha = width;
        observed.mp_target_ha = target;
        observed.mp_reference_energy_ha = reference_energy;
    }
    // Preserve the exact payload for mutation checks, but let the collective
    // production admission run before indexing diagnostic spin/density data.
    const auto original_payload = bands_payload(b);
    auto result = real_a(p, b, x, e, s);
    if (bands_payload(b) != original_payload)
        throw std::runtime_error("assembly mutated C/f/w");
    observed.state = &b;
    observed.ionic = &e;
    observed.payload = original_payload;
    if (b.fixed_spin_occupations) {
        auto& o = trace::shared_bands;
        o.fixed_na = b.fixed_spin_occupations->n_alpha;
        o.fixed_nb = b.fixed_spin_occupations->n_beta;
        if (o.fixed_na > 0 && o.fixed_nb == 0) {
            o.beta_layout_zero = true;
            o.beta_density_exact_zero = true;
            for (const auto& block : b.local_blocks)
                if (block.spin == 1) {
                    ++o.empty_beta_blocks;
                    o.beta_layout_zero = o.beta_layout_zero
                                         && block.coefficients.dimension() == 2
                                         && block.coefficients.shape(0) == 0
                                         && block.coefficients.shape(1)
                                                == p.gv_k[block.ik].g.size()
                                         && block.occupations.empty()
                                         && block.epsilon_ha.empty();
                }
            for (std::size_t g = 0; g < p.gv0.g.size(); ++g)
                o.beta_density_exact_zero =
                    o.beta_density_exact_zero
                    && p.ws.rho_spin_G[1][g] == std::complex<AccReal>{}
                    && b.reference_spin_density_G[1][g]
                           == std::complex<AccReal>{}
                    && b.orbital_spin_density_G[1][g]
                           == std::complex<AccReal>{};
        }
    }

    double filling[3] = {0, 0, is_mp ? 1.0 : 0.0};
    for (const auto& block : b.local_blocks)
        for (double f : block.occupations)
            filling[block.spin] += block.k_weight * f;
    int world = 1;
#ifdef ATV_HAS_MPI
    int mpi_active = 0;
    MPI_Initialized(&mpi_active);
    if (mpi_active) {
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        MPI_Allreduce(MPI_IN_PLACE, filling, 3, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
    }
#endif
    // All ranks must take the same optional MP reduction branch, including
    // Gamma ranks with no local blocks. Reuse the existing observer SUM.
    if (filling[2] != 0 && filling[2] != world)
        throw std::runtime_error(
            "diagnostic filling kind differs across ranks");
    observed.fcounts[0] = filling[0];
    observed.fcounts[1] = filling[1];
    int g0 = -1;
    for (std::size_t g = 0; g < p.gv0.h.size(); ++g)
        if (p.gv0.h[g] == 0 && p.gv0.k[g] == 0 && p.gv0.l[g] == 0)
            g0 = int(g);
    if (g0 < 0)
        throw std::runtime_error("observed density G0 missing");
    if (p.ws.nspin == 2)
        for (int spin = 0; spin < 2; ++spin)
            observed.rho_counts[spin] =
                b.orbital_spin_density_G[spin][g0].real() * p.vol_bohr;
    else
        observed.rho_counts[0] = b.orbital_density_G[g0].real() * p.vol_bohr;
    observed.aux_ne = b.auxiliary_electron_count;
    observed.aux_m = b.auxiliary_magnetization;
    observed.aux_entropy = b.auxiliary_entropy;
    observed.completion_residual = b.max_local_eigen_residual_ha;
    if (is_mp) {
        double limits[2] = {std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity()};
        for (const auto& block : b.local_blocks)
            for (std::size_t i = 0; i < block.occupations.size(); ++i) {
                const double f = block.occupations[i];
                limits[0] = std::min(limits[0], f);
                limits[1] = std::min(limits[1], -f);
                std::printf("MP_BAND spin=%d ik=%d band=%zu epsilon_Ha=%.17g "
                            "f=%.17g wk=%.17g mu_Ha=%.17g\n",
                            block.spin, block.ik, i, block.epsilon_ha[i], f,
                            block.k_weight, b.auxiliary_mu_ha);
            }
#ifdef ATV_HAS_MPI
        if (mpi_active)
            MPI_Allreduce(MPI_IN_PLACE, limits, 2, MPI_DOUBLE, MPI_MIN,
                          MPI_COMM_WORLD);
#endif
        if (!std::isfinite(limits[0]) || !std::isfinite(limits[1]))
            throw std::runtime_error("MP extrema require a global band state");
        observed.mp_min_f = limits[0];
        observed.mp_max_f = -limits[1];
        observed.mp_mu = b.auxiliary_mu_ha;
        if (p.ws.nspin == 1) {
            std::array<std::vector<std::complex<AccReal>>, 2> reference_half,
                orbital_half;
            for (int c = 0; c < 2; ++c)
                for (std::size_t g = 0; g < p.gv0.g.size(); ++g) {
                    reference_half[c].push_back(p.ws.rho_G[g] * .5);
                    orbital_half[c].push_back(b.orbital_density_G[g] * .5);
                }
            std::printf("RKS_XC_MAPPING physical_channels=1 "
                        "point_rho_a=point_rho_b=rho_total/2 zeta=0\n");
            pw_density_domain_audit::inspect("RKS_reference_mapped_half", p,
                                             reference_half);
            pw_density_domain_audit::inspect("RKS_orbital_mapped_half", p,
                                             orbital_half);
        }
        if (p.ws.nspin == 2) {
            pw_density_domain_audit::inspect("accepted_reference", p,
                                             p.ws.rho_spin_G);
            pw_density_domain_audit::inspect("accepted_orbital", p,
                                             b.orbital_spin_density_G);
        }
        std::printf("MP_AUX internal_E_Ha=%.17g generalized_S=%.17g "
                    "aux_free_Ha=%.17g reference_free_Ha=%.17g "
                    "density_mismatch=%.17g\n",
                    result.total.energy_ha, b.auxiliary_entropy,
                    result.total.energy_ha - width * b.auxiliary_entropy,
                    reference_energy, b.density_mismatch_l2);
    }

    return result;
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
    ++trace::counts.legacy_stress;
    return real_s(f, m, r, p);
}

using ForceValues = std::vector<std::array<double, 3>>;
ForceValues real_fb(
    const cpu::Frame&, const PWProblem&, const FinalBandState&,
    const EwaldSummation*) asm("__real__ZN3ATV3cpu2pw8internal20pw_forces_from_"
                               "bandsERKNS0_5FrameERKNS2_9PWProblemERKNS2_"
                               "14FinalBandStateEPKNS2_14EwaldSummationE");
ForceValues wrap_fb(
    const cpu::Frame&, const PWProblem&, const FinalBandState&,
    const EwaldSummation*) asm("__wrap__ZN3ATV3cpu2pw8internal20pw_forces_from_"
                               "bandsERKNS0_5FrameERKNS2_9PWProblemERKNS2_"
                               "14FinalBandStateEPKNS2_14EwaldSummationE");
ForceValues wrap_fb(const cpu::Frame& f, const PWProblem& p,
                    const FinalBandState& b, const EwaldSummation* e) {
    auto& s = trace::shared_bands;
    ++s.force_calls;
    // Force-only public calls need no preceding stress assembly.
    if (s.state) {
        s.same_inputs =
            s.state == &b && s.ionic == e && s.payload == bands_payload(b);
        if (!s.same_inputs)
            throw std::runtime_error("force/stress did not share C/f/Ewald");
    }
    auto result = real_fb(f, p, b, e);
    if (s.state && s.payload != bands_payload(b))
        throw std::runtime_error("forces mutated C/f");
    return result;
}
