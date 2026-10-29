#include "atomiverse/cpu/planewave/internal/pw_final_state.h"

#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/electronic/smearing.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV::cpu::pw::internal {
namespace {
using Complex = std::complex<AccReal>;
struct Layout {
    int rank = 0, size = 1;
};
Layout layout() {
    Layout value;
#ifdef ATV_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &value.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &value.size);
    }
#endif
    return value;
}
void reduce_sum(AccReal* values, int n, const Layout& mpi) {
#ifdef ATV_HAS_MPI
    if (mpi.size > 1)
        MPI_Allreduce(MPI_IN_PLACE, values, n,
                      sizeof(AccReal) == sizeof(double) ? MPI_DOUBLE
                                                        : MPI_FLOAT,
                      MPI_SUM, MPI_COMM_WORLD);
#endif
}
std::vector<Complex> copy(const xt::xarray<Complex>& values) {
    return {values.begin(), values.end()};
}
void density_binding(PWWorkspace& ws, const PWProblem& problem) {
    ws.gvec = &problem.gv0;
    ws.n_g = problem.gv0.count();
}
void band_binding(PWWorkspace& ws, const PWProblem& problem, int ik) {
    const auto& gv = problem.gv_k.at(static_cast<std::size_t>(ik));
    ws.gvec = &gv;
    ws.n_g = gv.count();
    ws.T_G.resize({gv.g.size()});
    std::copy(gv.ekin.begin(), gv.ekin.end(), ws.T_G.begin());
    ws.kb_proj = problem.kb_proj_k.empty()
                     ? std::vector<std::vector<PWWorkspace::KBProjG>>{}
                     : problem.kb_proj_k.at(static_cast<std::size_t>(ik));
    ws.atom_wfc = problem.atom_wfc_k.empty()
                      ? std::vector<std::vector<PWWorkspace::AtomWfc>>{}
                      : problem.atom_wfc_k.at(static_cast<std::size_t>(ik));
}
// Derivative scratch owns its FFT and writable arrays; no pointer in the
// accepted workspace is rebound and no FFT cache is shared concurrently.
void scratch_from_reference(PWWorkspace& out, const PWProblem& problem,
                            bool spin) {
    const auto& in = problem.ws;
    density_binding(out, problem);
    out.nr = in.nr;
    out.nc = in.nc;
    out.fft_nx = in.fft_nx;
    out.fft_ny = in.fft_ny;
    out.fft_nz = in.fft_nz;
    out.omega = in.omega;
    out.atom_pos_bohr = in.atom_pos_bohr;
    out.V_loc_G = in.V_loc_G;
    out.rho_G = in.rho_G;
    // Preserve the existing force epilogue's optional first-block trial pool.
    // init_subspace admits it only when its band/G shape matches; this owned
    // copy never replaces the accepted reference orbitals.
    out.wfc = in.wfc;
    out.nspin = spin ? 2 : 1;
    out.fft = std::make_unique<FFTPlan3D>(out.fft_nx, out.fft_ny, out.fft_nz);
    const auto n = static_cast<std::size_t>(out.n_g);
    out.V_H_G = xt::zeros<Complex>({n});
    out.V_xc_G = xt::zeros<Complex>({n});
    out.V_eff_G = xt::zeros<Complex>({n});
    if (spin) {
        out.rho_spin_G = in.rho_spin_G;
        out.V_xc_spin_G.resize(2);
        for (auto& channel : out.V_xc_spin_G)
            channel = xt::zeros<Complex>({n});
    }
}
} // namespace

FinalBandState extract_final_bands(const PWProblem& problem, const PW& method,
                                   const PWResult& reference,
                                   int final_band_count) {
    const AccReal target = AccReal(method.tol) * AccReal(.1);
    if (!(target > 0) || !std::isfinite(target))
        throw std::invalid_argument(
            "PW final bands: positive finite tolerance required");
    const bool spin = method.spin_polarized;
    const bool smear = method.smearing != electronic::Smearing::none;
    const int nk = static_cast<int>(problem.kg.kpoints.size());
    if (nk < 1 || problem.gv_k.size() != static_cast<std::size_t>(nk)
        || problem.kg.weights.size() != static_cast<std::size_t>(nk)
        || !problem.potential || problem.ws.rho_G.size() != problem.gv0.g.size()
        || problem.ws.nr < 1 || !(problem.ws.omega > 0))
        throw std::invalid_argument(
            "PW final bands: incomplete accepted geometry");
    if (smear && (!(method.degauss > 0) || !std::isfinite(method.degauss)))
        throw std::invalid_argument(
            "PW final bands: positive finite width required");
    int na = problem.n_elec / 2, nb = na;
    if (spin) {
        const int magnetization = method.multiplicity - 1;
        if (method.multiplicity < 1 || magnetization > problem.n_elec
            || (problem.n_elec + magnetization) % 2 != 0
            || problem.ws.rho_spin_G.size() != 2)
            throw std::invalid_argument(
                "PW final bands: incompatible spin state");
        na = (problem.n_elec + magnetization) / 2;
        nb = problem.n_elec - na;
        for (const auto& channel : problem.ws.rho_spin_G)
            if (channel.size() != problem.gv0.g.size())
                throw std::invalid_argument(
                    "PW final bands: spin density shape");
    } else if (!smear) {
        require_closed_shell(problem.n_elec, "PW final bands");
    }
    int bands = std::max(spin ? std::max(na, nb) : problem.n_elec / 2, 1);
    if (smear)
        bands = std::max(bands, std::max((problem.n_elec + 1) / 2, 1)
                                    + std::max(method.extra_bands, 0));
    if (final_band_count < 0 || (final_band_count && final_band_count < bands))
        throw std::invalid_argument(
            "PW final bands: requested bands omit filling states");
    bands = std::max(bands, final_band_count);
    for (const auto& gv : problem.gv_k)
        if (gv.count() < bands)
            throw std::invalid_argument(
                "PW final bands: bands exceed full G basis");
    const auto mpi = layout();
    const int ns = spin ? 2 : 1;
    const int occupied[2] = {spin ? na : problem.n_elec / 2, nb};
    FinalBandState result;
    if (!smear && spin)
        result.fixed_spin_occupations = FixedSpinOccupations{na, nb};
    result.reference_result = reference;
    result.reference_density_G = copy(problem.ws.rho_G);
    if (spin)
        for (const auto& channel : problem.ws.rho_spin_G)
            result.reference_spin_density_G.push_back(copy(channel));
    PWWorkspace ws;
    PoissonSolver poisson;
    XCBuilder xc;
    VeffBuilder veff;
    DavidsonSolver davidson(bands, target, 40);
    DensityBuilder density;
    std::vector<AccReal> spectrum(smear ? ns * nk * bands : 0, AccReal(0));
    std::exception_ptr failure;
    try {
        scratch_from_reference(ws, problem, spin);
        xc.prepare(ws);
        veff.prepare(ws);
        davidson.prepare(ws);
        density.prepare(ws);
        poisson.step(ws);
        if (spin)
            xc.step_spin(ws, method.xc);
        else
            xc.step(ws, method.xc);
        FFTWBuffer<fftw_complex> residual_work(static_cast<std::size_t>(ws.nr));
        for (int s = 0; s < ns; ++s) {
            density_binding(ws, problem);
            if (spin)
                ws.V_xc_G = ws.V_xc_spin_G[s];
            veff.step(ws);
            for (int ik = mpi.rank; ik < nk; ik += mpi.size) {
                FinalBandBlock block;
                block.spin = s;
                block.ik = ik;
                block.k_weight = problem.kg.weights[ik];
                if (!smear && occupied[s] == 0) {
                    block.coefficients = xt::zeros<Complex>(
                        {std::size_t(0), problem.gv_k[ik].g.size()});
                    result.local_blocks.push_back(std::move(block));
                    continue;
                }
                band_binding(ws, problem, ik);
                davidson.step(ws);
                ++result.local_band_solves;
                if (ws.eigenvalues.size() != static_cast<std::size_t>(bands)
                    || ws.wfc.dimension() != 2
                    || ws.wfc.shape(0) != static_cast<std::size_t>(bands)
                    || ws.wfc.shape(1) != static_cast<std::size_t>(ws.n_g))
                    throw std::runtime_error(
                        "PW final bands: incomplete eigenspectrum");
                block.coefficients = ws.wfc;
                block.epsilon_ha.assign(ws.eigenvalues.begin(),
                                        ws.eigenvalues.end());
                for (const auto eigenvalue : block.epsilon_ha)
                    if (!std::isfinite(eigenvalue))
                        throw std::runtime_error(
                            "PW final bands: nonfinite eigenvalue");
                for (const auto coefficient : block.coefficients)
                    if (!std::isfinite(coefficient.real())
                        || !std::isfinite(coefficient.imag()))
                        throw std::runtime_error(
                            "PW final bands: nonfinite coefficient");
                if (smear)
                    std::copy(block.epsilon_ha.begin(), block.epsilon_ha.end(),
                              spectrum.begin() + (s * nk + ik) * bands);
                else {
                    block.occupations.assign(bands, 0);
                    std::fill_n(block.occupations.begin(), occupied[s],
                                spin ? 1 : 2);
                }
                std::vector<Complex> hpsi(ws.n_g);
                for (int b = 0; b < bands; ++b) {
                    const auto* c =
                        ws.wfc.data() + static_cast<std::size_t>(b) * ws.n_g;
                    pw_apply_h(ws, c, hpsi.data(), residual_work.data());
                    AccReal norm2 = 0, coefficient_norm2 = 0;
                    for (int g = 0; g < ws.n_g; ++g) {
                        norm2 +=
                            std::norm(hpsi[g] - block.epsilon_ha[b] * c[g]);
                        coefficient_norm2 += std::norm(c[g]);
                    }
                    if (!std::isfinite(coefficient_norm2)
                        || std::abs(coefficient_norm2 - AccReal(1))
                               > AccReal(1e-8))
                        throw std::runtime_error(
                            "PW final bands: orbital normalization incomplete");
                    if (!std::isfinite(norm2))
                        throw std::runtime_error(
                            "PW final bands: nonfinite orbital residual");
                    if (std::sqrt(norm2) > target)
                        throw std::runtime_error("PW final bands: auxiliary "
                                                 "eigenproblem incomplete");
                    result.max_local_eigen_residual_ha = std::max(
                        result.max_local_eigen_residual_ha, std::sqrt(norm2));
                }
                result.local_blocks.push_back(std::move(block));
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    int failed = failure ? 1 : 0;
#ifdef ATV_HAS_MPI
    if (mpi.size > 1)
        MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
#endif
    if (failed) {
        if (mpi.size == 1 && failure)
            std::rethrow_exception(failure);
        throw std::runtime_error(
            "PW final bands: auxiliary completion failed on at least one rank");
    }
    if (smear) {
        reduce_sum(spectrum.data(), static_cast<int>(spectrum.size()), mpi);
        std::vector<AccReal> weights(spectrum.size());
        for (int s = 0; s < ns; ++s)
            for (int ik = 0; ik < nk; ++ik)
                std::fill_n(weights.begin() + (s * nk + ik) * bands, bands,
                            problem.kg.weights[ik]);
        const AccReal width = method.degauss * qe_units::rydberg_to_hartree;
        const AccReal degeneracy = spin ? 1 : 2;
        result.auxiliary_mu_ha = electronic::find_fermi_level(
            spectrum, weights, AccReal(problem.n_elec), width, method.smearing,
            method.mp_order, degeneracy);
        const auto fill = electronic::fill_occupations(
            spectrum, weights, result.auxiliary_mu_ha, width, method.smearing,
            method.mp_order, degeneracy);
        result.auxiliary_entropy = fill.entropy;
        result.auxiliary_electron_count = fill.n_occ;
        for (auto& block : result.local_blocks) {
            const auto begin =
                fill.occ.begin() + (block.spin * nk + block.ik) * bands;
            block.occupations.assign(begin, begin + bands);
        }
        if (spin)
            for (int ik = 0; ik < nk; ++ik)
                for (int b = 0; b < bands; ++b)
                    result.auxiliary_magnetization +=
                        problem.kg.weights[ik]
                        * (fill.occ[ik * bands + b]
                           - fill.occ[(nk + ik) * bands + b]);
    } else {
        result.auxiliary_electron_count = problem.n_elec;
        result.auxiliary_magnetization = spin ? na - nb : 0;
    }
    result.orbital_density_G.assign(problem.gv0.g.size(), Complex{});
    for (int s = 0; s < ns; ++s) {
        density_binding(ws, problem);
        density.begin(ws);
        for (const auto& block : result.local_blocks) {
            if (block.spin != s || block.occupations.empty())
                continue;
            band_binding(ws, problem, block.ik);
            ws.wfc = block.coefficients;
            ws.eigenvalues.resize({block.epsilon_ha.size()});
            std::copy(block.epsilon_ha.begin(), block.epsilon_ha.end(),
                      ws.eigenvalues.begin());
            density.accumulate_kpoint(
                ws, block.k_weight, static_cast<int>(block.occupations.size()),
                block.occupations.data());
        }
        reduce_sum(density.accumulation(), ws.nr, mpi);
        density_binding(ws, problem);
        xt::xarray<Complex> rho;
        density.extract(ws, rho);
        if (spin)
            result.orbital_spin_density_G.push_back(copy(rho));
        for (std::size_t g = 0; g < result.orbital_density_G.size(); ++g)
            result.orbital_density_G[g] += rho[g];
    }
    AccReal mismatch2 = 0;
    if (spin)
        for (int s = 0; s < ns; ++s)
            for (std::size_t g = 0; g < problem.gv0.g.size(); ++g)
                mismatch2 += std::norm(result.orbital_spin_density_G[s][g]
                                       - result.reference_spin_density_G[s][g]);
    else
        for (std::size_t g = 0; g < problem.gv0.g.size(); ++g)
            mismatch2 += std::norm(result.orbital_density_G[g]
                                   - result.reference_density_G[g]);
    result.density_mismatch_l2 = std::sqrt(mismatch2);
    return result;
}
} // namespace ATV::cpu::pw::internal
