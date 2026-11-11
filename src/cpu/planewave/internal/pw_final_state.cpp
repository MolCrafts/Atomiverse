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
// Only the frozen input density and trial orbitals are copied into scratch.
void scratch_from_reference(PWWorkspace& out, PWGridView grid,
                            std::span<const Vec3> positions,
                            PWDensityView reference,
                            const xt::xarray<Complex>& trial, bool spin) {
    out.bind({grid.modes});
    out.fft_nx = grid.modes.nx;
    out.fft_ny = grid.modes.ny;
    out.fft_nz = grid.modes.nz;
    out.omega = grid.volume;
    out.atom_pos_bohr = positions;
    out.V_loc_G = grid.local_potential;
    out.wfc = trial;
    out.nspin = spin ? 2 : 1;
    out.fft = std::make_unique<FFTPlan3D>(out.fft_nx, out.fft_ny, out.fft_nz);
    out.nr = out.fft->nr();
    out.nc = out.fft->nc();
    const auto n = static_cast<std::size_t>(out.n_g);
    out.rho_G.resize({n});
    std::copy(reference.total.begin(), reference.total.end(),
              out.rho_G.begin());
    out.V_H_G = xt::zeros<Complex>({n});
    out.V_xc_G = xt::zeros<Complex>({n});
    out.V_eff_G = xt::zeros<Complex>({n});
    if (spin) {
        out.rho_spin_G.assign(reference.spin.begin(), reference.spin.end());
        out.V_xc_spin_G.resize(2);
        for (auto& channel : out.V_xc_spin_G)
            channel = xt::zeros<Complex>({n});
    }
}
} // namespace

FinalBandState extract_final_bands(PWGridView grid, const PWIons& ions,
                                   const PWKMesh& mesh,
                                   PWDensityView density_ref,
                                   const xt::xarray<Complex>& trial,
                                   const PW& method, const PWResult& reference,
                                   int final_band_count) {
    const AccReal target = AccReal(method.tol) * AccReal(.1);
    if (!(target > 0) || !std::isfinite(target))
        throw std::invalid_argument(
            "PW final bands: positive finite tolerance required");
    const bool spin = method.spin_polarized;
    const bool smear = method.smearing != electronic::Smearing::none;
    const int nk = static_cast<int>(mesh.kg.kpoints.size());
    if (nk < 1 || mesh.gv_k.size() != static_cast<std::size_t>(nk)
        || mesh.kg.weights.size() != static_cast<std::size_t>(nk)
        || ions.species.empty()
        || density_ref.total.size() != grid.modes.g.size() || grid.modes.nx < 1
        || grid.modes.ny < 1 || grid.modes.nz < 1
        || grid.local_potential.size() != grid.modes.g.size()
        || !(grid.volume > 0))
        throw std::invalid_argument(
            "PW final bands: incomplete accepted geometry");
    if (smear && (!(method.degauss > 0) || !std::isfinite(method.degauss)))
        throw std::invalid_argument(
            "PW final bands: positive finite width required");
    int na = ions.n_elec / 2, nb = na;
    if (spin) {
        const int magnetization = method.multiplicity - 1;
        if (method.multiplicity < 1 || magnetization > ions.n_elec
            || (ions.n_elec + magnetization) % 2 != 0
            || density_ref.spin.size() != 2)
            throw std::invalid_argument(
                "PW final bands: incompatible spin state");
        na = (ions.n_elec + magnetization) / 2;
        nb = ions.n_elec - na;
        for (const auto& channel : density_ref.spin)
            if (channel.size() != grid.modes.g.size())
                throw std::invalid_argument(
                    "PW final bands: spin density shape");
    } else if (!smear) {
        require_closed_shell(ions.n_elec, "PW final bands");
    }
    int bands = std::max(spin ? std::max(na, nb) : ions.n_elec / 2, 1);
    if (smear)
        bands = std::max(bands, std::max((ions.n_elec + 1) / 2, 1)
                                    + std::max(method.extra_bands, 0));
    if (final_band_count < 0 || (final_band_count && final_band_count < bands))
        throw std::invalid_argument(
            "PW final bands: requested bands omit filling states");
    bands = std::max(bands, final_band_count);
    for (const auto& gv : mesh.gv_k)
        if (gv.count() < bands)
            throw std::invalid_argument(
                "PW final bands: bands exceed full G basis");
    const auto mpi = layout();
    const int ns = spin ? 2 : 1;
    const int occupied[2] = {spin ? na : ions.n_elec / 2, nb};
    FinalBandState result;
    if (!smear && spin)
        result.fixed_spin_occupations = FixedSpinOccupations{na, nb};
    result.reference_result = reference;
    result.reference_density_G.assign(density_ref.total.begin(),
                                      density_ref.total.end());
    if (spin)
        for (const auto& channel : density_ref.spin)
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
        scratch_from_reference(ws, grid, ions.positions, density_ref, trial,
                               spin);
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
            ws.bind({grid.modes});
            if (spin)
                ws.V_xc_G = ws.V_xc_spin_G[s];
            veff.step(ws);
            for (int ik = mpi.rank; ik < nk; ik += mpi.size) {
                FinalBandBlock block;
                block.spin = s;
                block.ik = ik;
                block.k_weight = mesh.kg.weights[ik];
                if (!smear && occupied[s] == 0) {
                    block.coefficients = xt::zeros<Complex>(
                        {std::size_t(0), mesh.gv_k[ik].g.size()});
                    result.local_blocks.push_back(std::move(block));
                    continue;
                }
                ws.bind(mesh.point(ik));
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
                            mesh.kg.weights[ik]);
        const AccReal width = method.degauss * qe_units::rydberg_to_hartree;
        const AccReal degeneracy = spin ? 1 : 2;
        result.auxiliary_mu_ha = electronic::find_fermi_level(
            spectrum, weights, AccReal(ions.n_elec), width, method.smearing,
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
                        mesh.kg.weights[ik]
                        * (fill.occ[ik * bands + b]
                           - fill.occ[(nk + ik) * bands + b]);
    } else {
        result.auxiliary_electron_count = ions.n_elec;
        result.auxiliary_magnetization = spin ? na - nb : 0;
    }
    result.orbital_density_G.assign(grid.modes.g.size(), Complex{});
    for (int s = 0; s < ns; ++s) {
        ws.bind({grid.modes});
        density.begin(ws);
        for (const auto& block : result.local_blocks) {
            if (block.spin != s || block.occupations.empty())
                continue;
            ws.bind(mesh.point(block.ik));
            ws.wfc = block.coefficients;
            ws.eigenvalues.resize({block.epsilon_ha.size()});
            std::copy(block.epsilon_ha.begin(), block.epsilon_ha.end(),
                      ws.eigenvalues.begin());
            density.accumulate_kpoint(
                ws, block.k_weight, static_cast<int>(block.occupations.size()),
                block.occupations.data());
        }
        reduce_sum(density.accumulation(), ws.nr, mpi);
        ws.bind({grid.modes});
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
            for (std::size_t g = 0; g < grid.modes.g.size(); ++g)
                mismatch2 += std::norm(result.orbital_spin_density_G[s][g]
                                       - result.reference_spin_density_G[s][g]);
    else
        for (std::size_t g = 0; g < grid.modes.g.size(); ++g)
            mismatch2 += std::norm(result.orbital_density_G[g]
                                   - result.reference_density_G[g]);
    result.density_mismatch_l2 = std::sqrt(mismatch2);
    return result;
}
} // namespace ATV::cpu::pw::internal
