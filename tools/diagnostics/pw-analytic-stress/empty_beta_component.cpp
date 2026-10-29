#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/frozen_functional_virial.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "empty_beta_fixture.h"
#include <algorithm>
#include <cstdio>
#include <functional>
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
using namespace ATV;
using namespace ATV::cpu::pw;
using namespace ATV::cpu::pw::internal;
int actual_D = 0, rank_id = 0, world = 1;
void real_d(DavidsonSolver*, PWWorkspace&) asm(
    "__real__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver*, PWWorkspace&) asm(
    "__wrap__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver* s, PWWorkspace& w) {
    ++actual_D;
    real_d(s, w);
}
void need(bool x, const char* m) {
    if (!x)
        throw std::runtime_error(m);
}
std::vector<std::complex<AccReal>> payload(const FinalBandState& s) {
    std::vector<std::complex<AccReal>> v;
    auto add = [&](const auto& a) { v.insert(v.end(), a.begin(), a.end()); };
    add(s.reference_density_G);
    add(s.orbital_density_G);
    for (const auto& a : s.reference_spin_density_G)
        add(a);
    for (const auto& a : s.orbital_spin_density_G)
        add(a);
    for (const auto& b : s.local_blocks) {
        v.emplace_back(b.spin, b.ik);
        v.emplace_back(b.coefficients.shape(0), b.coefficients.shape(1));
        add(b.coefficients);
        for (auto x : b.occupations)
            v.emplace_back(x);
        for (auto x : b.epsilon_ha)
            v.emplace_back(x);
    }
    return v;
}
void one(cpu::xc::XCType xc, const char* label) {
    auto f = empty_beta_fixture::frame();
    auto method = empty_beta_fixture::method(xc);
    PWProblem p;
    setup_pw_problem(f, *method->potential, 6, 1, p);
    auto& w = p.ws;
    w.nspin = 2;
    w.rho_spin_G = {w.rho_G, xt::zeros<std::complex<AccReal>>(w.rho_G.shape())};
    w.rho_spin_G_old = w.rho_spin_G;
    w.V_xc_spin_G = {xt::zeros<std::complex<AccReal>>(w.rho_G.shape()),
                     xt::zeros<std::complex<AccReal>>(w.rho_G.shape())};
    PWResult reference;
    reference.energy = 123;
    reference.energy_ha = 456;
    reference.converged = false;
    reference.iterations = 1984;
    reference.magnetization = 77;
    reference.e_fermi = -9;
    reference.n_elec_occ = 88;
    reference.e_entropy = -3;
    reference.n_kpts = 47;
    reference.n_kpts_local = 29;
    const auto rho0 = w.rho_G;
    const auto spin0 = w.rho_spin_G;
    const int start = actual_D;
    auto state = extract_final_bands(p, *method, reference);
    const int owned = rank_id == 0 ? 1 : 0;
    need(actual_D - start == owned && state.local_band_solves == owned,
         "real fixed empty sweep budget");
    need(state.local_blocks.size() == std::size_t(2 * owned),
         "complete spin-k layout");
    need(state.fixed_spin_occupations
             && state.fixed_spin_occupations->n_alpha == 2
             && state.fixed_spin_occupations->n_beta == 0,
         "producer typed filling");
    need(!state.reference_result.converged
             && state.reference_result.energy == 123
             && state.reference_result.energy_ha == 456
             && state.reference_result.iterations == 1984
             && state.reference_result.magnetization == 77
             && state.reference_result.e_fermi == -9
             && state.reference_result.n_elec_occ == 88
             && state.reference_result.e_entropy == -3
             && state.reference_result.n_kpts == 47
             && state.reference_result.n_kpts_local == 29,
         "false synthetic reference retained");
    std::vector<Vec3> pos;
    for (int i = 0; i < 2; ++i)
        pos.push_back(
            {f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]});
    EwaldSummation ionic(pos, std::vector<AccReal>(2, 1), f.box);
    XCBuilder scratch;
    const auto original = payload(state);
    auto value =
        assemble_frozen_functional_virial(p, state, xc, ionic, scratch);
    need(std::isfinite(value.total.energy_ha), "finite valid empty tensor");
    for (double x : value.total.virial_ha)
        need(std::isfinite(x), "finite W9");
    need(payload(state) == original, "assembly input unchanged");
    auto signed_zero = state;
    for (auto& x : signed_zero.reference_spin_density_G[1])
        x = {-0., 0.};
    for (auto& x : signed_zero.orbital_spin_density_G[1])
        x = {0., -0.};
    auto z =
        assemble_frozen_functional_virial(p, signed_zero, xc, ionic, scratch);
    need(std::abs(z.total.energy_ha - value.total.energy_ha) < 1e-13,
         "signed zero legal");
    int rejects = 0;
    auto reject = [&](const char* name, int owner,
                      std::function<void(FinalBandState&)> mutate) {
        auto bad = state;
        if (rank_id == owner)
            mutate(bad);
        auto before = payload(bad);
        bool caught = false;
        try {
            (void)assemble_frozen_functional_virial(p, bad, xc, ionic, scratch);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        int all = caught ? 1 : 0;
#ifdef ATV_HAS_MPI
        MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        need(all == 1,
             "all ranks must reject malformed/undeclared empty state");
        need(payload(bad) == before, "rejected state unchanged");
        ++rejects;
        std::printf("EMPTY_COMPONENT_REJECT %s case=%s rank=%d all_rejected=1 "
                    "unchanged=1\n",
                    label, name, rank_id);
    };
    const int remote = world > 1 ? 1 : 0;
    reject("remote_missing_declaration", remote,
           [](auto& s) { s.fixed_spin_occupations.reset(); });
    reject("wrong_counts", remote, [](auto& s) {
        s.fixed_spin_occupations = FixedSpinOccupations{1, 1};
    });
    reject("nonzero_beta_payload", remote,
           [](auto& s) { s.orbital_spin_density_G[1][0] = {1e-30, 0}; });
    reject("aux_M_mismatch", remote,
           [](auto& s) { s.auxiliary_magnetization += .5; });
    reject("missing_beta_block", 0, [](auto& s) { s.local_blocks.pop_back(); });
    reject("malformed_beta_shape", 0, [](auto& s) {
        auto& b = s.local_blocks.back();
        b.coefficients = xt::zeros<std::complex<AccReal>>(
            {std::size_t(0), b.coefficients.shape(1) - 1});
    });
    reject("wrong_weight", 0,
           [](auto& s) { s.local_blocks.back().k_weight = .5; });
    reject("missing_alpha_rows", 0, [](auto& s) {
        auto& b = s.local_blocks.front();
        b.coefficients = xt::zeros<std::complex<AccReal>>(
            {std::size_t(0), b.coefficients.shape(1)});
        b.occupations.clear();
        b.epsilon_ha.clear();
    });
    reject("fake_thermal_beta_virtual", 0, [](auto& s) {
        auto& b = s.local_blocks.back();
        b.coefficients = xt::zeros<std::complex<AccReal>>(
            {std::size_t(1), b.coefficients.shape(1)});
        b.occupations = {0};
        b.epsilon_ha = {0};
    });
    need(std::equal(w.rho_G.begin(), w.rho_G.end(), rho0.begin()),
         "reference total rho unchanged");
    for (int s = 0; s < 2; ++s)
        need(std::equal(w.rho_spin_G[s].begin(), w.rho_spin_G[s].end(),
                        spin0[s].begin()),
             "reference spin rho unchanged");
    need(actual_D - start == owned, "no hidden band solves in validation");
    std::printf("PASS EMPTY_COMPONENT %s rank=%d ranks=%d SCF_calls=0 "
                "local_Davidson=%d local_blocks=%zu typed_Na=2 typed_Nb=0 "
                "rejects=%d synthetic_false_preserved=1 signed_zero_legal=1\n",
                label, rank_id, world, owned, state.local_blocks.size(),
                rejects);
}
int main(int argc, char** argv) {
#ifdef ATV_HAS_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
#endif
    int code = 0;
    try {
        one(cpu::xc::XCType::LSDA, "LSDA");
        one(cpu::xc::XCType::PBE, "PBE");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL EMPTY_COMPONENT rank=%d %s\n", rank_id,
                     e.what());
        code = 1;
    }
#ifdef ATV_HAS_MPI
    MPI_Finalize();
#endif
    return code;
}
