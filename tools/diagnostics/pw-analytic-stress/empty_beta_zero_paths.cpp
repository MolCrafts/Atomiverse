#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/frozen_functional_virial.h"
#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/pw_forces.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "empty_beta_fixture.h"
#include <cstdio>
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
using namespace ATV;
using namespace ATV::cpu::pw;
using namespace ATV::cpu::pw::internal;
int D = 0;
void real_d(DavidsonSolver*, PWWorkspace&) asm(
    "__real__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver*, PWWorkspace&) asm(
    "__wrap__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver* s, PWWorkspace& w) {
    ++D;
    real_d(s, w);
}
void need(bool v, const char* m) {
    if (!v)
        throw std::runtime_error(m);
}
int main(int argc, char** argv) {
    int rank = 0, size = 1;
#ifdef ATV_HAS_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    int code = 0;
    try {
        cpu::Frame si(1);
        si.box = cpu::Box::cubic(6);
        si.span(kAtom_type)[0] = 14;
        si.span(kAtom_x)[0] = si.span(kAtom_y)[0] = si.span(kAtom_z)[0] = 2;
        GthPseudopotential pp(molrs::Element::Si, "GTH-PBE-q4");
        PWProblem ps;
        setup_pw_problem(si, pp, 6, 1, ps);
        need(ps.potential->n_proj() > 0, "nontrivial silicon NL data required");
        FrozenBandView empty;
        need(empty.coefficients.data() == nullptr
                 && empty.occupations.data() == nullptr,
             "actual null spans");
        auto t = kinetic_frozen_virial(ps.gv_k[0], empty);
        auto nl = nonlocal_frozen_virial(
            *ps.potential, ps.gv_k[0], ps.ws.atom_pos_bohr, empty, ps.vol_bohr);
        need(t.energy_ha == 0 && nl.energy_ha == 0, "empty T NL energies zero");
        for (int a = 0; a < 9; ++a)
            need(t.virial_ha[a] == 0 && nl.virial_ha[a] == 0,
                 "empty T NL nine components zero");
        auto f = empty_beta_fixture::frame();
        auto m = empty_beta_fixture::method(cpu::xc::XCType::LSDA);
        PWProblem p;
        setup_pw_problem(f, *m->potential, 6, 1, p);
        p.ws.nspin = 2;
        p.ws.rho_spin_G = {
            p.ws.rho_G, xt::zeros<std::complex<AccReal>>(p.ws.rho_G.shape())};
        p.ws.rho_spin_G_old = p.ws.rho_spin_G;
        PWResult ref;
        ref.converged = false;
        ref.magnetization = 77;
        auto s = extract_final_bands(p, *m, ref);
        std::vector<Vec3> pos;
        for (int i = 0; i < 2; ++i)
            pos.push_back(
                {f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]});
        EwaldSummation ew(pos, std::vector<AccReal>(2, 1), f.box);
        auto force = pw_forces_from_bands(f, p, s, &ew);
        for (const auto& v : force)
            for (double x : v)
                need(std::isfinite(x),
                     "finite force consumer on legal empty beta");
        auto gap = extract_final_bands(p, *m, ref, 3);
        for (const auto& b : gap.local_blocks)
            if (b.spin == 1)
                need(b.coefficients.dimension() == 2
                         && b.coefficients.shape(0) == 0
                         && b.coefficients.shape(1) == p.gv_k[b.ik].g.size()
                         && b.occupations.empty() && b.epsilon_ha.empty(),
                     "extra gap request keeps beta canonical zero rows");
        XCBuilder scratch;
        auto tensor = assemble_frozen_functional_virial(
            p, gap, cpu::xc::XCType::LSDA, ew, scratch);
        need(std::isfinite(tensor.total.energy_ha),
             "extra alpha zero-f state valid");
        need(D == (rank == 0 ? 2 : 0),
             "actual two separately counted auxiliary sweeps no SCF");
        std::printf("PASS EMPTY_ZERO_PATHS rank=%d ranks=%d "
                    "Si_NL_projectors=%d null_data_T_NL_exact0=1 "
                    "F_legal_empty_beta=1 alpha_extra_zero_f_beta_rows0=1 "
                    "SCF_calls=0 default_plus_extra_local_Davidson=%d\n",
                    rank, size, ps.potential->n_proj(), D);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL EMPTY_ZERO_PATHS rank=%d %s\n", rank,
                     e.what());
        code = 1;
    }
#ifdef ATV_HAS_MPI
    MPI_Finalize();
#endif
    return code;
}
