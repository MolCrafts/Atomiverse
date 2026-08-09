/// @file cpu/pw/internal/scf_loop.cpp
/// @brief Shared minimal restricted fixed-occupation plane-wave SCF loop.

#include "atomiverse/cpu/planewave/internal/scf_loop.h"

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/smearing.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/qe_units.h"

#include <xtensor/containers/xarray.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

struct MpiLayout {
    int rank = 0;
    int nprocs = 1;
    bool active = false;
};

MpiLayout mpi_layout() {
    MpiLayout layout;
#ifdef ATV_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &layout.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &layout.nprocs);
        layout.active = (layout.nprocs > 1);
    }
#endif
    return layout;
}

#ifdef ATV_HAS_MPI
MPI_Datatype mpi_real_type() {
    return sizeof(AccReal) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
}
#endif

} // namespace

PW copy_pw_params(const PW& m) {
    PW p;
    // Pseudopotential source — exactly one of these is set, and dropping
    // either makes the copy throw "no default pseudopotential" at build().
    p.upf_path = m.upf_path;
    p.gth_name = m.gth_name;
    // Basis / k-mesh
    p.ecut_ry = m.ecut_ry;
    p.k_mesh = m.k_mesh;
    // SCF control
    p.tol = m.tol;
    p.max_iter = m.max_iter;
    p.mix_beta = m.mix_beta;
    p.pulay_depth = m.pulay_depth;
    // Hamiltonian / occupations
    p.xc = m.xc;
    p.spin_polarized = m.spin_polarized;
    p.multiplicity = m.multiplicity;
    p.smearing = m.smearing;
    p.degauss = m.degauss;
    p.mp_order = m.mp_order;
    p.extra_bands = m.extra_bands;
    return p;
}

PWResult run_scf_fixed_occ(const cpu::Frame& frame, const PW& method,
                           PWProblem& prob, PoissonSolver& poisson,
                           XCBuilder& xc, VeffBuilder& veff,
                           DensityBuilder& density, DavidsonSolver& davidson,
                           const char* tag) {
    PWResult result{};
    const auto layout = mpi_layout();

    setup_pw_problem(frame, method.upf_path, method.gth_name,
                     method.ecut_ry, method.k_mesh,
                     prob);
    PWWorkspace& ws = prob.ws;
    auto& gv0 = prob.gv0;
    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    const int n_bands = prob.n_bands;
    const AccReal vol_bohr = prob.vol_bohr;
    ws.mix_beta = AccReal(method.mix_beta);
    ws.pulay_depth = method.pulay_depth;

    xc.prepare(ws);
    veff.prepare(ws);
    density.prepare(ws);
    density.reset(ws);
    davidson.prepare(ws);
    ConvergenceCheck convergence{AccReal(method.tol), 2};

    for (int iter = 0; iter < method.max_iter; ++iter) {
        poisson.step(ws);
        xc.step(ws, method.xc);
        veff.step(ws);

        density.begin(ws);
        AccReal e_band = 0;
        auto* saved_gvec = ws.gvec;
        for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
            auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
            ws.gvec = &gvk;
            ws.n_g = gvk.count();
            ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
            for (int ig = 0; ig < ws.n_g; ++ig)
                ws.T_G[ig] = gvk.ekin[ig];

            if (!prob.kb_proj_k.empty())
                ws.kb_proj = prob.kb_proj_k[static_cast<std::size_t>(ik)];
            if (!prob.atom_wfc_k.empty())
                ws.atom_wfc = prob.atom_wfc_k[static_cast<std::size_t>(ik)];

            davidson.step(ws);

            const AccReal weight =
                prob.kg.weights[static_cast<std::size_t>(ik)];
            for (int b = 0; b < n_bands; ++b)
                e_band += weight * ws.eigenvalues[b] * AccReal(2);

            density.accumulate_kpoint(ws, weight, n_bands);

            ws.gvec = saved_gvec;
            ws.n_g = gv0.count();
        }

#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, density.accumulation(), ws.nr,
                          mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(), MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif

        density.finish(ws);
        result.iterations = iter + 1;

        const auto terms =
            iteration_energy(ws, e_band, prob.ew_total, vol_bohr);
        result.energy_ha = static_cast<double>(terms.total);
        result.energy = result.energy_ha / qe_units::rydberg_to_hartree;

        if (convergence.step(ws)) {
            result.converged = true;
            if (layout.rank == 0)
                ATV_INFO("[pw] {}-scf converged iter={} energy_ry={:.8f}", tag,
                         iter + 1, result.energy);
            break;
        }
    }
    if (!result.converged && layout.rank == 0)
        ATV_WARN("[pw] {}-scf not converged after {} iters", tag,
                 method.max_iter);
    return result;
}

PWResult run_scf_smear_restricted(const cpu::Frame& frame, const PW& method,
                                  PWProblem& prob, PoissonSolver& poisson,
                                  XCBuilder& xc, VeffBuilder& veff,
                                  DensityBuilder& density,
                                  DavidsonSolver& davidson, int n_bands,
                                  const char* tag) {
    if (method.smearing == Smearing::none)
        throw std::invalid_argument(
            "run_scf_smear_restricted: method.smearing must be set");
    if (!(method.degauss > 0.0))
        throw std::invalid_argument(
            "run_scf_smear_restricted: degauss must be > 0 [Ry]");
    if (method.spin_polarized)
        throw std::invalid_argument(
            "run_scf_smear_restricted: collinear spin unsupported "
            "(use PW::run)");
    if (n_bands < 1)
        throw std::invalid_argument(
            "run_scf_smear_restricted: n_bands must be >= 1");

    PWResult result{};
    const auto layout = mpi_layout();

    setup_pw_problem(frame, method.upf_path, method.gth_name,
                     method.ecut_ry, method.k_mesh,
                     prob);
    PWWorkspace& ws = prob.ws;
    auto& gv0 = prob.gv0;
    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    const AccReal vol_bohr = prob.vol_bohr;
    ws.mix_beta = AccReal(method.mix_beta);
    ws.pulay_depth = method.pulay_depth;

    SmearKind kind = SmearKind::fermi_dirac;
    if (method.smearing == Smearing::methfessel_paxton)
        kind = SmearKind::methfessel_paxton;
    else if (method.smearing == Smearing::gaussian)
        kind = SmearKind::gaussian;
    const AccReal degauss_ha =
        static_cast<AccReal>(method.degauss) * AccReal(0.5);
    const int mp_order = method.mp_order;

    xc.prepare(ws);
    veff.prepare(ws);
    density.prepare(ws);
    density.reset(ws);
    davidson.prepare(ws);
    ConvergenceCheck convergence{AccReal(method.tol), 2};

    std::vector<std::vector<AccReal>> evals_k(
        static_cast<std::size_t>(n_kpts));
    std::vector<xt::xarray<std::complex<AccReal>>> wfc_k(
        static_cast<std::size_t>(n_kpts));

    for (int iter = 0; iter < method.max_iter; ++iter) {
        poisson.step(ws);
        xc.step(ws, method.xc);
        veff.step(ws);

        // Pass 1: diagonalize owned k-points; stash evals + wfc.
        auto* saved_gvec = ws.gvec;
        for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
            auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
            ws.gvec = &gvk;
            ws.n_g = gvk.count();
            ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
            for (int ig = 0; ig < ws.n_g; ++ig)
                ws.T_G[ig] = gvk.ekin[ig];
            if (!prob.kb_proj_k.empty())
                ws.kb_proj = prob.kb_proj_k[static_cast<std::size_t>(ik)];
            if (!prob.atom_wfc_k.empty())
                ws.atom_wfc = prob.atom_wfc_k[static_cast<std::size_t>(ik)];

            davidson.step(ws);

            evals_k[static_cast<std::size_t>(ik)].assign(
                ws.eigenvalues.begin(),
                ws.eigenvalues.begin()
                    + std::min(n_bands, static_cast<int>(ws.eigenvalues.size())));
            if (static_cast<int>(evals_k[static_cast<std::size_t>(ik)].size())
                > n_bands)
                evals_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(n_bands));
            wfc_k[static_cast<std::size_t>(ik)] = ws.wfc;
            ws.gvec = saved_gvec;
            ws.n_g = gv0.count();
        }

#ifdef ATV_HAS_MPI
        if (layout.active) {
            std::vector<AccReal> pack(
                static_cast<std::size_t>(n_kpts * n_bands), AccReal(0));
            for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                for (int b = 0;
                     b < n_bands && b < static_cast<int>(ev.size()); ++b)
                    pack[static_cast<std::size_t>(ik * n_bands + b)] = ev[b];
            }
            MPI_Allreduce(MPI_IN_PLACE, pack.data(), n_kpts * n_bands,
                          mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
            for (int ik = 0; ik < n_kpts; ++ik) {
                evals_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(n_bands));
                for (int b = 0; b < n_bands; ++b)
                    evals_k[static_cast<std::size_t>(ik)][b] =
                        pack[static_cast<std::size_t>(ik * n_bands + b)];
            }
        }
#endif

        std::vector<AccReal> all_evals;
        std::vector<AccReal> all_weights;
        all_evals.reserve(static_cast<std::size_t>(n_kpts * n_bands));
        all_weights.reserve(static_cast<std::size_t>(n_kpts * n_bands));
        for (int ik = 0; ik < n_kpts; ++ik) {
            const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            for (int b = 0; b < n_bands && b < static_cast<int>(ev.size());
                 ++b) {
                all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                all_weights.push_back(w);
            }
        }

        const AccReal mu_ha =
            find_fermi_level(all_evals, all_weights, AccReal(prob.n_elec),
                             degauss_ha, kind, mp_order);

        AccReal entropy_sum = AccReal(0);
        AccReal n_occ = AccReal(0);
        std::vector<std::vector<AccReal>> occ_k(
            static_cast<std::size_t>(n_kpts));
        for (int ik = 0; ik < n_kpts; ++ik) {
            const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            std::vector<AccReal> w_one(ev.size(), w);
            auto fill = fill_occupations(ev, w_one, mu_ha, degauss_ha, kind,
                                         mp_order);
            occ_k[static_cast<std::size_t>(ik)].resize(ev.size());
            for (std::size_t b = 0; b < ev.size(); ++b)
                occ_k[static_cast<std::size_t>(ik)][b] =
                    smear_f_band(ev[b], mu_ha, degauss_ha, kind, mp_order);
            n_occ += fill.n_occ;
            entropy_sum += fill.entropy;
        }

        // Pass 2: density + band energy with fractional occupations.
        density.begin(ws);
        AccReal e_band = 0;
        for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
            auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
            ws.gvec = &gvk;
            ws.n_g = gvk.count();
            ws.wfc = wfc_k[static_cast<std::size_t>(ik)];
            ws.eigenvalues.resize(
                {evals_k[static_cast<std::size_t>(ik)].size()});
            for (std::size_t b = 0;
                 b < evals_k[static_cast<std::size_t>(ik)].size(); ++b)
                ws.eigenvalues[b] =
                    evals_k[static_cast<std::size_t>(ik)][b];

            const AccReal weight =
                prob.kg.weights[static_cast<std::size_t>(ik)];
            const auto& occ = occ_k[static_cast<std::size_t>(ik)];
            const int nb = static_cast<int>(occ.size());
            for (int b = 0; b < nb; ++b)
                e_band += weight * ws.eigenvalues[b]
                          * occ[static_cast<std::size_t>(b)];
            density.accumulate_kpoint(ws, weight, nb, occ.data());

            ws.gvec = saved_gvec;
            ws.n_g = gv0.count();
        }

#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, density.accumulation(), ws.nr,
                          mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(), MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif

        density.finish(ws);
        result.iterations = iter + 1;

        const auto terms =
            iteration_energy(ws, e_band, prob.ew_total, vol_bohr);
        // Mermin free energy: F = E_KS − σ · S  (σ = degauss in Ha).
        const AccReal e_ent_ha = -degauss_ha * entropy_sum;
        const AccReal e_free_ha = terms.total + e_ent_ha;
        result.energy_ha = static_cast<double>(e_free_ha);
        result.energy = result.energy_ha / qe_units::rydberg_to_hartree;
        result.e_fermi =
            static_cast<double>(mu_ha) / qe_units::rydberg_to_hartree;
        result.e_entropy =
            static_cast<double>(e_ent_ha) / qe_units::rydberg_to_hartree;
        result.n_elec_occ = static_cast<double>(n_occ);

        if (convergence.step(ws)) {
            result.converged = true;
            if (layout.rank == 0)
                ATV_INFO("[pw] {}-smear-scf converged iter={} F_ry={:.8f} "
                         "mu_ry={:.6f} n_occ={:.6f}",
                         tag, iter + 1, result.energy, result.e_fermi,
                         result.n_elec_occ);
            break;
        }
    }
    if (!result.converged && layout.rank == 0)
        ATV_WARN("[pw] {}-smear-scf not converged after {} iters", tag,
                 method.max_iter);
    return result;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
