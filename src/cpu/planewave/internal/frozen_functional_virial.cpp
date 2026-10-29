#include "atomiverse/cpu/planewave/internal/frozen_functional_virial.h"

#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV::cpu::pw::internal {
namespace {
struct Layout {
    int rank = 0, size = 1;
};
Layout layout() {
    Layout out;
#ifdef ATV_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &out.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &out.size);
    }
#endif
    return out;
}
void synchronize_error(const Layout& mpi, std::exception_ptr error) {
    int failed = error ? 1 : 0;
#ifdef ATV_HAS_MPI
    if (mpi.size > 1)
        MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
#endif
    if (error)
        std::rethrow_exception(error);
    if (failed)
        throw std::invalid_argument("PW frozen functional: another rank "
                                    "rejected the input or evaluation");
}
// One existing admission MAX also checks typed declaration agreement,
// including a Gamma rank with no local blocks; no extra collective is added.
void synchronize_admission(const Layout& mpi, std::exception_ptr error,
                           const std::optional<FixedSpinOccupations>& fixed) {
    const long long kind = fixed ? 1 : 0, na = fixed ? fixed->n_alpha : -1,
                    nb = fixed ? fixed->n_beta : -1;
    std::array<long long, 7> fields{
        error ? 1LL : 0LL, kind, -kind, na, -na, nb, -nb};
#ifdef ATV_HAS_MPI
    if (mpi.size > 1)
        MPI_Allreduce(MPI_IN_PLACE, fields.data(), 7, MPI_LONG_LONG, MPI_MAX,
                      MPI_COMM_WORLD);
#endif
    if (error)
        std::rethrow_exception(error);
    if (fields[0] || fields[1] != -fields[2] || fields[3] != -fields[4]
        || fields[5] != -fields[6])
        throw std::invalid_argument("PW frozen functional: collective fixed "
                                    "declaration/input rejection");
}
void require(bool good, const char* message) {
    if (!good)
        throw std::invalid_argument(message);
}
bool finite(std::complex<AccReal> v) {
    return std::isfinite(v.real()) && std::isfinite(v.imag());
}
void validate(const PWProblem& p, const FinalBandState& bands,
              cpu::xc::XCType functional, const EwaldSummation& ionic,
              const Layout& mpi) {
    const auto& ws = p.ws;
    const auto ng = p.gv0.g.size();
    require(functional == cpu::xc::XCType::LSDA
                || cpu::xc::is_pbe_family(functional),
            "PW frozen functional: only qualified LSDA/PBE-family semilocal "
            "models");
    require(
        p.potential && !p.kg.kpoints.empty()
            && p.kg.weights.size() == p.kg.kpoints.size()
            && p.gv_k.size() == p.kg.kpoints.size(),
        "PW frozen functional: matching nonempty k-point geometry is required");
    AccReal weight_sum = 0;
    for (const auto weight : p.kg.weights) {
        require(std::isfinite(weight) && weight > 0,
                "PW frozen functional: positive finite k weight required");
        weight_sum += weight;
    }
    require(std::abs(weight_sum - 1) < 1e-12,
            "PW frozen functional: normalized k weights required");
    const bool spin = ws.nspin == 2;
    const auto* fixed =
        bands.fixed_spin_occupations ? &*bands.fixed_spin_occupations : nullptr;
    if (fixed) {
        require(spin && fixed->n_alpha > 0 && fixed->n_beta >= 0
                    && static_cast<long long>(fixed->n_alpha) + fixed->n_beta
                           == p.n_elec,
                "PW frozen functional: invalid fixed spin declaration");
        require(std::abs(bands.auxiliary_electron_count - p.n_elec) < 1e-8
                    && std::abs(bands.auxiliary_magnetization
                                - (fixed->n_alpha - fixed->n_beta))
                           < 1e-8
                    && bands.auxiliary_entropy == 0,
                "PW frozen functional: fixed declaration/aux mismatch");
    }

    require(ws.nspin == 1 || spin, "general functional: spin shape");
    require(p.n_elec >= 0 && std::isfinite(bands.auxiliary_entropy)
                && std::isfinite(bands.auxiliary_electron_count)
                && std::abs(bands.auxiliary_electron_count - p.n_elec) < 1e-8,
            "general functional: collective occupation metadata");
    require(bands.orbital_spin_density_G.size() == (spin ? 2u : 0u)
                && bands.reference_spin_density_G.size() == (spin ? 2u : 0u)
                && ws.rho_spin_G.size() == (spin ? 2u : 0u),
            "general functional: spin channels");
    if (spin)
        for (int channel = 0; channel < 2; ++channel) {
            require(bands.orbital_spin_density_G[channel].size() == ng
                        && bands.reference_spin_density_G[channel].size() == ng
                        && ws.rho_spin_G[channel].size() == ng,
                    "general functional: spin density sizes");
            require(std::equal(bands.reference_spin_density_G[channel].begin(),
                               bands.reference_spin_density_G[channel].end(),
                               ws.rho_spin_G[channel].begin()),
                    "general functional: reference spin identity");
        }
    require(ng > 0 && p.gv0.h.size() == ng && p.gv0.k.size() == ng
                && p.gv0.l.size() == ng && ws.rho_G.size() == ng
                && bands.reference_density_G.size() == ng
                && bands.orbital_density_G.size() == ng,
            "PW frozen functional: density shapes do not match the reference "
            "modes");
    require(ws.fft_nx > 0 && ws.fft_ny > 0 && ws.fft_nz > 0 && ws.nr > 0
                && ws.nc > 0 && std::isfinite(p.vol_bohr) && p.vol_bohr > 0
                && ws.omega == p.vol_bohr,
            "PW frozen functional: invalid reference FFT or volume");
    require(std::equal(bands.reference_density_G.begin(),
                       bands.reference_density_G.end(), ws.rho_G.begin()),
            "PW frozen functional: bands belong to another reference density");
    require(std::isfinite(bands.density_mismatch_l2)
                && bands.density_mismatch_l2 >= 0,
            "PW frozen functional: invalid density mismatch");
    for (const auto rho : bands.orbital_density_G)
        require(finite(rho), "PW frozen functional: nonfinite orbital density");
    const auto ionic_tolerance = AccReal(64)
                                 * std::numeric_limits<AccReal>::epsilon()
                                 * std::max(AccReal(1), std::abs(p.ew_total));
    require(ionic.positions_bohr().size() == ws.atom_pos_bohr.size()
                && std::isfinite(ionic.total_energy())
                && std::abs(ionic.total_energy() - p.ew_total)
                       <= ionic_tolerance,
            "PW frozen functional: ionic contribution does not match the "
            "reference");
    if (fixed) {
        std::size_t g0 = ng;
        for (std::size_t g = 0; g < ng; ++g)
            if (p.gv0.h[g] == 0 && p.gv0.k[g] == 0 && p.gv0.l[g] == 0)
                g0 = g;
        require(g0 < ng, "PW frozen functional: fixed density G0 missing");
        for (int channel = 0; channel < 2; ++channel) {
            const auto count = channel ? fixed->n_beta : fixed->n_alpha;
            require(std::abs(bands.orbital_spin_density_G[channel][g0].real()
                                 * p.vol_bohr
                             - count)
                        < 1e-8,
                    "PW frozen functional: fixed orbital density count");
        }
        if (fixed->n_beta == 0)
            for (std::size_t g = 0; g < ng; ++g)
                require(ws.rho_spin_G[1][g] == std::complex<AccReal>{}
                            && bands.reference_spin_density_G[1][g]
                                   == std::complex<AccReal>{}
                            && bands.orbital_spin_density_G[1][g]
                                   == std::complex<AccReal>{},
                        "PW frozen functional: nonzero payload in fixed empty "
                        "beta density");
    }
    const int nk = static_cast<int>(p.gv_k.size());
    const int ns = spin ? 2 : 1;
    const int owned = mpi.rank < nk ? (nk - 1 - mpi.rank) / mpi.size + 1 : 0;
    require(bands.local_blocks.size() == static_cast<std::size_t>(owned * ns),
            "PW frozen functional: incompatible distributed block ownership");
    std::vector<bool> seen(ns * nk, false);
    for (const auto& b : bands.local_blocks) {
        const auto nb = b.occupations.size();
        require(b.ik >= 0 && b.ik < nk && b.ik % mpi.size == mpi.rank,
                "PW frozen functional: invalid local k owner");
        const auto nw = p.gv_k[b.ik].g.size();
        require(b.spin >= 0 && b.spin < ns && b.k_weight == p.kg.weights[b.ik]
                    && b.epsilon_ha.size() == nb
                    && b.coefficients.dimension() == 2
                    && b.coefficients.shape(0) == nb
                    && b.coefficients.shape(1) == nw,
                "PW frozen functional: invalid distributed band block shape");
        if (nb == 0) {
            require(
                fixed && fixed->n_beta == 0 && b.spin == 1,
                "PW frozen functional: undeclared empty/missing band block");
        }
        if (fixed) {
            const int occupied = b.spin ? fixed->n_beta : fixed->n_alpha;
            require(nb >= static_cast<std::size_t>(occupied),
                    "PW frozen functional: fixed occupied rows missing");
            if (occupied == 0)
                require(nb == 0, "PW frozen functional: fixed beta must use "
                                 "canonical zero rows");
            for (std::size_t band = 0; band < nb; ++band)
                require(b.occupations[band]
                            == (band < static_cast<std::size_t>(occupied)
                                    ? AccReal(1)
                                    : AccReal(0)),
                        "PW frozen functional: noncanonical fixed occupations");
        }
        require(!seen[b.spin * nk + b.ik],
                "PW frozen functional: duplicate spin block");
        seen[b.spin * nk + b.ik] = true;
        for (const auto f : b.occupations) {
            require(std::isfinite(f),
                    "PW frozen functional: finite signed occupation required");
        }
        for (const auto c : b.coefficients)
            require(finite(c), "PW frozen functional: nonfinite coefficient");
        for (const auto e : b.epsilon_ha)
            require(std::isfinite(e),
                    "PW frozen functional: nonfinite eigenvalue");
    }
}
void sum_band_terms(FrozenOperatorVirial& t, FrozenOperatorVirial& nl,
                    AccReal& electrons, const Layout& mpi) {
#ifdef ATV_HAS_MPI
    if (mpi.size > 1) {
        std::array<AccReal, 21> values{};
        values[20] = electrons;
        values[0] = t.energy_ha;
        values[10] = nl.energy_ha;
        std::copy(t.virial_ha.begin(), t.virial_ha.end(), values.begin() + 1);
        std::copy(nl.virial_ha.begin(), nl.virial_ha.end(),
                  values.begin() + 11);
        MPI_Allreduce(
            MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
            sizeof(AccReal) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM,
            MPI_COMM_WORLD);
        electrons = values[20];
        t.energy_ha = values[0];
        nl.energy_ha = values[10];
        std::copy_n(values.begin() + 1, 9, t.virial_ha.begin());
        std::copy_n(values.begin() + 11, 9, nl.virial_ha.begin());
    }
#endif
}
void add(FrozenOperatorVirial& out, const FrozenOperatorVirial& term) {
    out.energy_ha += term.energy_ha;
    for (int ab = 0; ab < 9; ++ab)
        out.virial_ha[ab] += term.virial_ha[ab];
}
void unpack(const std::array<AccReal, 6>& v, FrozenOperatorVirial& out) {
    constexpr int a[6] = {0, 1, 2, 0, 0, 1}, b[6] = {0, 1, 2, 1, 2, 2};
    for (int k = 0; k < 6; ++k) {
        out.virial_ha[3 * a[k] + b[k]] = v[k];
        out.virial_ha[3 * b[k] + a[k]] = v[k];
    }
}
} // namespace
FrozenFunctionalVirial assemble_frozen_functional_virial(
    const PWProblem& problem, const FinalBandState& bands,
    cpu::xc::XCType functional, const EwaldSummation& ionic,
    XCBuilder& xc_scratch) {
    const auto mpi = layout();
    std::exception_ptr error;
    try {
        validate(problem, bands, functional, ionic, mpi);
    } catch (...) {
        error = std::current_exception();
    }
    synchronize_admission(mpi, error, bands.fixed_spin_occupations);
    FrozenFunctionalVirial out;
    out.density_mismatch_l2 = bands.density_mismatch_l2;
    AccReal electrons = 0;
    try {
        for (const auto& block : bands.local_blocks) {
            for (const auto f : block.occupations)
                electrons += block.k_weight * f;
            const FrozenBandView view{
                {block.coefficients.data(), block.coefficients.size()},
                block.occupations,
                block.k_weight};
            add(out.kinetic,
                kinetic_frozen_virial(problem.gv_k[block.ik], view));
            add(out.nonlocal, nonlocal_frozen_virial(*problem.potential,
                                                     problem.gv_k[block.ik],
                                                     problem.ws.atom_pos_bohr,
                                                     view, problem.vol_bohr));
        }
    } catch (...) {
        error = std::current_exception();
    }
    synchronize_error(mpi, error);
    sum_band_terms(out.kinetic, out.nonlocal, electrons, mpi);
    require(
        std::isfinite(electrons) && std::abs(electrons - problem.n_elec) < 1e-8,
        "PW frozen functional: collective occupations disagree with reference");
    try {
        out.hartree = hartree_frozen_virial(
            problem.gv0, bands.orbital_density_G, problem.vol_bohr);
        out.local = local_potential_frozen_virial(
            *problem.potential, problem.gv0, problem.ws.atom_pos_bohr,
            bands.orbital_density_G, problem.vol_bohr);
        PWWorkspace scratch;
        scratch.gvec = &problem.gv0;
        scratch.n_g = problem.gv0.count();
        scratch.fft_nx = problem.ws.fft_nx;
        scratch.fft_ny = problem.ws.fft_ny;
        scratch.fft_nz = problem.ws.fft_nz;
        scratch.omega = problem.vol_bohr;
        scratch.fft = std::make_unique<FFTPlan3D>(
            scratch.fft_nx, scratch.fft_ny, scratch.fft_nz);
        scratch.nr = scratch.fft->nr();
        scratch.nc = scratch.fft->nc();
        require(scratch.nr == problem.ws.nr && scratch.nc == problem.ws.nc,
                "PW frozen functional: reference FFT sizes are inconsistent");
        scratch.rho_G.resize({bands.orbital_density_G.size()});
        std::copy(bands.orbital_density_G.begin(),
                  bands.orbital_density_G.end(), scratch.rho_G.begin());
        scratch.nspin = problem.ws.nspin;
        if (scratch.nspin == 2) {
            scratch.rho_spin_G.resize(2);
            for (int channel = 0; channel < 2; ++channel) {
                scratch.rho_spin_G[channel].resize(
                    {bands.orbital_density_G.size()});
                std::copy(bands.orbital_spin_density_G[channel].begin(),
                          bands.orbital_spin_density_G[channel].end(),
                          scratch.rho_spin_G[channel].begin());
            }
        }
        xc_scratch.prepare(scratch);
        if (scratch.nspin == 2)
            xc_scratch.step_spin(scratch, functional);
        else
            xc_scratch.step(scratch, functional);
        out.xc.energy_ha = scratch.E_xc;
        unpack(xc_scratch.virial_ha(), out.xc);
        out.ionic.energy_ha = ionic.total_energy();
        unpack(ionic.virial_ha(), out.ionic);
    } catch (...) {
        error = std::current_exception();
    }
    synchronize_error(mpi, error);
    for (const auto* term : {&out.kinetic, &out.hartree, &out.local,
                             &out.nonlocal, &out.xc, &out.ionic})
        add(out.total, *term);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            out.max_antisymmetric_ha =
                std::max(out.max_antisymmetric_ha,
                         std::abs(out.total.virial_ha[3 * a + b]
                                  - out.total.virial_ha[3 * b + a]));
    return out;
}
} // namespace ATV::cpu::pw::internal
