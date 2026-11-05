/// @file src/cpu/semiempirical/nddo_solver.cpp
/// @brief NDDO semiempirical inner solver scaffold.

#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/pulay_coefficients.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_derivative.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"
#include "atomiverse/cpu/semiempirical/nddo_density.h"
#include "atomiverse/cpu/semiempirical/nddo_fock.h"
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"
#include "atomiverse/cpu/semiempirical/nddo_matrix.h"
#include "atomiverse/semiempirical/am1_parameters.h"
#include "atomiverse/semiempirical/nddo_geometry.h"
#include "atomiverse/semiempirical/rm1_parameters.h"

extern "C" {
// Fortran BLAS, the vocabulary of cpu/linalg/orbital_tangent.cpp's zgemm_.
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

/// Pseudo-diagonalization starts no earlier than this SCF iteration ...
constexpr int kPseudoDiagFirstIteration = 3;
/// ... and only once max|F P − P F| [eV] of the last Fock build is this small.
constexpr double kPseudoDiagCommutatorEv = 0.05;
/// Below this HOMO–LUMO-like gap [eV] first-order mixing is unreliable.
constexpr double kPseudoDiagMinGapEv = 0.5;

std::vector<double>
packed_density_from_diagonal(const std::vector<double>& diagonal) {
    std::vector<double> packed(diagonal.size() * (diagonal.size() + 1u) / 2u,
                               0.0);
    for (std::size_t i = 0; i < diagonal.size(); ++i) {
        packed[packed_index(static_cast<int>(i), static_cast<int>(i))] =
            diagonal[i];
    }
    return packed;
}

std::vector<double> scaled_density(const std::vector<double>& density,
                                   double scale) {
    auto out = density;
    for (auto& value : out) {
        value *= scale;
    }
    return out;
}

double max_abs_delta(const std::vector<double>& a,
                     const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("NDDO density delta size mismatch");
    }
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double delta = std::abs(a[i] - b[i]);
        if (!std::isfinite(delta))
            throw std::runtime_error("NDDO non-finite density residual");
        out = std::max(out, delta);
    }
    return out;
}

} // namespace

NDDOSolver::NDDOSolver(NDDOConfig config) : config_(config) {
    set_scf_controls(config.max_iter, config.density_tol, config.energy_tol);
    result_.method = config_.method;
    result_.reference = std::string("Internal ")
                        + ATV::semiempirical::nddo_method_name(config_.method)
                        + "/NDDO reference data";
}

void NDDOSolver::set_scf_controls(int max_iter, double density_tol,
                                  double energy_tol) {
    if (max_iter <= 0 || !(density_tol > 0.0) || !std::isfinite(density_tol)
        || !(energy_tol > 0.0) || !std::isfinite(energy_tol))
        throw std::invalid_argument("NDDO requires a positive iteration budget "
                                    "and positive finite SCF tolerances");
    config_.max_iter = max_iter;
    config_.density_tol = density_tol;
    config_.energy_tol = energy_tol;
}

void NDDOSolver::build(const cpu::Frame& reference) {
    validate_supported_atoms_(reference);
    fock_builder_.release_two_center();
    system_ = NDDOSystem{reference, config_.charge, config_.multiplicity,
                         config_.method};
    state_.initial_diagonal_density = initial_diagonal_density_guess(system_);
    state_.initial_packed_density =
        packed_density_from_diagonal(state_.initial_diagonal_density);
    state_.initial_spin_density =
        scaled_density(state_.initial_packed_density, 0.5);
    const auto hcore = NDDOHCoreBuilder{}.build(system_, reference);
    state_.packed_hcore = hcore.packed_hcore;
    state_.nuclear_repulsion_ev = hcore.nuclear_repulsion_ev;
    state_.initial_one_center_fock = fock_builder_.build(
        system_, reference, state_.packed_hcore, state_.initial_packed_density,
        state_.initial_spin_density);

    diagonalizer_.build(system_.n_orbitals);
    scf_coeff_.reserve(static_cast<std::size_t>(system_.n_orbitals)
                       * system_.n_orbitals);
    n_atoms_ = reference.nlocal();
    result_.converged = false;
    scf_delta_ = std::numeric_limits<double>::infinity();
    scf_energy_delta_ = std::numeric_limits<double>::infinity();
    result_.iterations = 0;
    result_.electronic_energy_hartree = 0.0;
    result_.total_energy_ev = 0.0;
    result_.heat_of_formation_kcal_mol = 0.0;
    result_.mulliken_charges.assign(static_cast<std::size_t>(n_atoms_), 0.0);
    result_.forces_ev_a.clear();
    result_.forces_ev_a.reserve(static_cast<std::size_t>(3 * n_atoms_));
    force_scratch_.resize(static_cast<std::size_t>(3 * n_atoms_));
    density_.reserve(state_.initial_packed_density.size());
    spin_density_.reserve(state_.initial_spin_density.size());
    pending_density_.reserve(state_.initial_packed_density.size());
    pending_spin_density_.reserve(state_.initial_spin_density.size());
    reset_history();
    built_ = true;
}

void NDDOSolver::scf_prepare(const cpu::Frame& frame) {
    if (!built_) {
        throw std::runtime_error(
            "NDDOSolver::scf_prepare called before build()");
    }
    result_.converged = false;
    result_.forces_ev_a.clear();
    validate_supported_atoms_(frame);
    const auto species = frame.span(kAtom_atomic_number);
    bool same_system = frame.nlocal() == n_atoms_;
    for (int i = 0; same_system && i < n_atoms_; ++i)
        same_system = species[i] == system_.atoms[i].atomic_number;
    if (!same_system)
        build(frame);

    if (system_.n_alpha != system_.n_beta) {
        throw std::runtime_error(
            "NDDOSolver AM1 SCF currently supports closed-shell RHF only");
    }

    result_.forces_ev_a.clear();
    result_.converged = false;

    scf_x_ = frame.span(kAtom_x);
    scf_y_ = frame.span(kAtom_y);
    scf_z_ = frame.span(kAtom_z);

    const auto hcore =
        NDDOHCoreBuilder{}.build(system_, scf_x_, scf_y_, scf_z_);
    scf_hcore_ = hcore.packed_hcore;
    scf_nuc_ev_ = hcore.nuclear_repulsion_ev;
    // Warm-start from the last converged density when available (AIMD / opt).
    // Do not clear has_density_ before this branch — that would force a cold
    // start every step and break multi-step MD/Minimize SCF stability.
    scf_density_ = has_density_ ? density_ : state_.initial_packed_density;
    scf_spin_ = scaled_density(scf_density_, 0.5);
    // Two-center blocks depend only on geometry: compute them once per solve
    // (unless they exceed the builder's memory budget) and reuse them in
    // every Fock build of this SCF.
    fock_builder_.cache_two_center(system_, scf_x_, scf_y_, scf_z_);
    scf_fock_ = build_scf_fock_();
    scf_prev_energy_ev_ = std::numeric_limits<double>::infinity();
    scf_delta_ = std::numeric_limits<double>::infinity();
    scf_energy_delta_ = std::numeric_limits<double>::infinity();
    scf_iter_ = 0;
    scf_coeff_.clear();
    // The guess Fock is diagonalized as is; DIIS history starts afresh.
    diis_count_ = 0;
    diis_next_ = 0;
    scf_fock_diis_ = scf_fock_;
    scf_commutator_ = std::numeric_limits<double>::infinity();
}

void NDDOSolver::scf_diagonalize() {
    // Diagonalize until the occupied space is nearly settled, then rotate the
    // previous orbitals with BLAS-3 pseudo-diagonalization; any refusal (a
    // small orbital gap, a failed orthonormalization) diagonalizes instead.
    const bool settled = scf_iter_ >= kPseudoDiagFirstIteration
                         && scf_commutator_ <= kPseudoDiagCommutatorEv;
    if (!(settled
          && diagonalizer_.pseudo_diagonalize_packed(
              scf_fock_diis_, system_.n_beta, kPseudoDiagMinGapEv, scf_coeff_)))
        diagonalizer_.diagonalize_packed(scf_fock_diis_, scf_coeff_);
}

void NDDOSolver::scf_density() {
    NDDODensityInput density_input;
    density_input.mdim = system_.n_orbitals;
    density_input.norbs = system_.n_orbitals;
    density_input.n_occupied = system_.n_beta;
    density_input.n_fractional = system_.n_beta;
    density_input.full_occupancy = 2.0;
    density_input.fractional_occupancy = 0.0;
    // The preceding Fock build has consumed spin_density. Reuse that packed
    // buffer for the next total density, then recycle the previous density as
    // the new half-spin buffer. No third matrix or changed contraction order.
    density_builder_.build_packed(scf_coeff_, density_input, scf_spin_);
    scf_delta_ = max_abs_delta(scf_density_, scf_spin_);
    scf_density_.swap(scf_spin_);
    for (std::size_t i = 0; i < scf_density_.size(); ++i)
        scf_spin_[i] = scf_density_[i] * 0.5;
}

void NDDOSolver::scf_fock() {
    scf_fock_ = build_scf_fock_();
    extrapolate_fock_();
}

std::vector<double> NDDOSolver::build_scf_fock_() {
    if (fock_builder_.has_two_center_cache(system_))
        return fock_builder_.build_cached(system_, scf_hcore_, scf_density_,
                                          scf_spin_);
    return fock_builder_.build(system_, scf_x_, scf_y_, scf_z_, scf_hcore_,
                               scf_density_, scf_spin_);
}

void NDDOSolver::extrapolate_fock_() {
    // Pulay DIIS on the orthonormal NDDO basis: the residual is the
    // commutator e = F P - P F = F P - (F P)^T (antisymmetric, so its strict
    // lower triangle carries it). Energy and convergence use the
    // unextrapolated F(P); only the next diagonalization sees the mixture.
    const int n = system_.n_orbitals;
    const auto dim = static_cast<std::size_t>(n);
    diis_f_.resize(dim * dim);
    diis_p_.resize(dim * dim);
    diis_fp_.resize(dim * dim);
    const int n_threads = dense_thread_count_();
#pragma omp parallel for schedule(static)                                      \
    num_threads(n_threads) if (n_threads > 1)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            const auto k = packed_index(i, j);
            diis_f_[i * dim + j] = diis_f_[j * dim + i] = scf_fock_[k];
            diis_p_[i * dim + j] = diis_p_[j * dim + i] = scf_density_[k];
        }
    const char no = 'N';
    const double one = 1.0, zero = 0.0;
    dgemm_(&no, &no, &n, &n, &n, &one, diis_f_.data(), &n, diis_p_.data(), &n,
           &zero, diis_fp_.data(), &n);

    // Slot for this vector: append until full, then overwrite the oldest.
    const int slot =
        diis_count_ < kDiisVectors ? diis_count_ : diis_next_ % kDiisVectors;
    diis_next_ = (slot + 1) % kDiisVectors;
    diis_count_ = std::min(diis_count_ + 1, kDiisVectors);
    auto& error = diis_error_[static_cast<std::size_t>(slot)];
    error.resize(dim * (dim - 1) / 2);
    double largest = 0.0;
    // Column-major FP(i, j) = fp[i + j n]; strict lower triangle i > j, rows
    // stored consecutively from i(i-1)/2. The max is order independent.
#pragma omp parallel for schedule(static) num_threads(n_threads)               \
    reduction(max : largest) if (n_threads > 1)
    for (int i = 1; i < n; ++i) {
        std::size_t e = static_cast<std::size_t>(i) * (i - 1) / 2;
        for (int j = 0; j < i; ++j, ++e) {
            const double value = diis_fp_[i + j * dim] - diis_fp_[j + i * dim];
            error[e] = value;
            largest = std::max(largest, std::abs(value));
        }
    }
    if (!std::isfinite(largest))
        throw std::runtime_error("NDDO non-finite DIIS commutator");
    scf_commutator_ = largest;
    diis_fock_[static_cast<std::size_t>(slot)] = scf_fock_;
    // One serial dot per history vector: deterministic for any thread count.
    const int count = diis_count_;
#pragma omp parallel for schedule(static)                                      \
    num_threads(n_threads) if (n_threads > 1)
    for (int k = 0; k < count; ++k) {
        const auto& other = diis_error_[static_cast<std::size_t>(k)];
        double dot = 0.0;
        for (std::size_t q = 0; q < error.size(); ++q)
            dot += error[q] * other[q];
        diis_gram_[slot * kDiisVectors + k] = dot;
        diis_gram_[k * kDiisVectors + slot] = dot;
    }

    scf_fock_diis_ = scf_fock_;
    const int m = diis_count_;
    if (m < 2)
        return;
    std::array<double, kDiisVectors * kDiisVectors> gram{};
    for (int a = 0; a < m; ++a)
        for (int b = 0; b < m; ++b)
            gram[a * m + b] = diis_gram_[a * kDiisVectors + b];
    std::array<double, kDiisVectors> coefficients{};
    if (!cpu::linalg::try_pulay_coefficients(gram.data(), m,
                                             coefficients.data())) {
        // Linearly dependent history: restart from the newest vector.
        diis_fock_[0].swap(diis_fock_[static_cast<std::size_t>(slot)]);
        diis_error_[0].swap(diis_error_[static_cast<std::size_t>(slot)]);
        diis_gram_[0] = diis_gram_[slot * kDiisVectors + slot];
        diis_count_ = 1;
        diis_next_ = 1;
        return;
    }
    std::fill(scf_fock_diis_.begin(), scf_fock_diis_.end(), 0.0);
    for (int k = 0; k < m; ++k) {
        const auto& fock = diis_fock_[static_cast<std::size_t>(k)];
        const double c = coefficients[static_cast<std::size_t>(k)];
        for (std::size_t q = 0; q < fock.size(); ++q)
            scf_fock_diis_[q] += c * fock[q];
    }
}

void NDDOSolver::scf_check() {
    ++scf_iter_;
    const double electronic_ev = electronic_energy_ev(
        system_.n_orbitals, scf_density_, scf_hcore_, scf_fock_);
    const double total_ev = electronic_ev + scf_nuc_ev_;
    if (!std::isfinite(total_ev))
        throw std::runtime_error("NDDO non-finite SCF energy");
    const double energy_delta = std::abs(total_ev - scf_prev_energy_ev_);

    result_.iterations = scf_iter_;
    result_.total_energy_ev = total_ev;
    result_.electronic_energy_hartree = total_ev * units::kEvInHartree;
    result_.mulliken_charges =
        atomic_charges_from_packed_density(system_, scf_density_);
    scf_energy_delta_ = energy_delta;
    if (scf_delta_ <= config_.density_tol
        && energy_delta <= config_.energy_tol) {
        // MOPAC ΔHf: (E_elec + E_nuc − Σ eisol)·(eV→kcal) + Σ atomic ΔHf
        // (openmopac.net/Manual/calc_of_hof.html).
        double sum_eisol = 0.0;
        double sum_atheat = 0.0;
        for (const auto& atom : system_.atoms) {
            const auto* p = atom.params;
            if (p == nullptr) {
                throw std::runtime_error(
                    "NDDO parameters missing during heat-of-formation");
            }
            sum_eisol += p->eisol;
            sum_atheat += ATV::semiempirical::nddo_atomic_heat_kcal_mol(
                config_.method, atom.atomic_number);
        }
        result_.heat_of_formation_kcal_mol =
            (total_ev - sum_eisol) * ATV::semiempirical::kAm1EvToKcalMol
            + sum_atheat;
        return;
    }
    scf_prev_energy_ev_ = total_ev;
}

void NDDOSolver::stage_density() {
    const auto packed_size = static_cast<std::size_t>(system_.n_orbitals)
                             * (system_.n_orbitals + 1) / 2;
    pending_density_valid_ = false;
    if (!built_ || !result_.converged || scf_density_.size() != packed_size
        || scf_spin_.size() != packed_size)
        throw std::runtime_error("NDDO cannot accept an unconverged density");
    pending_density_ = scf_density_;
    pending_spin_density_ = scf_spin_;
    pending_density_valid_ = true;
}

void NDDOSolver::commit_staged_density() noexcept {
    if (!pending_density_valid_)
        return;
    density_.swap(pending_density_);
    spin_density_.swap(pending_spin_density_);
    has_density_ = true;
    pending_density_valid_ = false;
}

void NDDOSolver::commit_density() {
    stage_density();
    commit_staged_density();
}

void NDDOSolver::reset_history() noexcept {
    density_.clear();
    spin_density_.clear();
    has_density_ = false;
    pending_density_valid_ = false;
}

ATV::semiempirical::NDDOResult&
NDDOSolver::compute_forces(cpu::Frame& frame) try {
    if (!built_) {
        throw std::runtime_error(
            "NDDOSolver::compute_forces called before build()");
    }
    const auto packed_size = static_cast<std::size_t>(system_.n_orbitals)
                             * (system_.n_orbitals + 1) / 2;
    if (!result_.converged || scf_density_.size() != packed_size
        || scf_spin_.size() != packed_size) {
        throw std::runtime_error(
            "NDDOSolver::compute_forces requires a converged SCF density");
    }
    if (frame.nlocal() != n_atoms_) {
        throw std::runtime_error(
            "NDDOSolver::compute_forces atom count mismatch");
    }
    // At converged density the NDDO energy is stationary. Differentiate only
    // each pair's scalar integrals, then contract directly with that density.
    // Three analytic Cartesian directions are propagated together; no displaced
    // coordinates, SCF response solve, or full-system derivative matrices.
    const std::array<const StorageReal*, 3> coordinates{
        frame.span(kAtom_x), frame.span(kAtom_y), frame.span(kAtom_z)};
    const int n_threads = force_thread_count_();
    force_threads_.resize(static_cast<std::size_t>(n_threads));
    std::exception_ptr failure;
#pragma omp parallel num_threads(n_threads) if (n_threads > 1)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& scratch = force_threads_[static_cast<std::size_t>(tid)];
        scratch.forces.assign(static_cast<std::size_t>(3 * n_atoms_), 0.0);
        scratch.pair.method = system_.method;
        scratch.pair.atoms.resize(2);
#pragma omp for schedule(static, 1)
        for (int i = 1; i < n_atoms_; ++i) {
            try {
                for (int j = 0; j < i; ++j)
                    accumulate_pair_force_(i, j, coordinates, scratch.pair,
                                           scratch.forces);
            } catch (...) {
#pragma omp critical(nddo_force_failure)
                if (!failure)
                    failure = std::current_exception();
            }
        }
    }
    if (failure)
        std::rethrow_exception(failure);
    auto& forces = force_scratch_;
    forces.assign(static_cast<std::size_t>(3 * n_atoms_), 0.0);
    for (int t = 0; t < n_threads; ++t) {
        const auto& local = force_threads_[static_cast<std::size_t>(t)].forces;
        for (std::size_t k = 0; k < forces.size(); ++k)
            forces[k] += local[k];
    }
    result_.forces_ev_a.swap(forces);
    return result_;
} catch (...) {
    // Method results must not advertise a successful calculation with partial
    // derivatives, even when the caller catches and retries the failure.
    result_.forces_ev_a.clear();
    result_.converged = false;
    throw;
}

int NDDOSolver::dense_thread_count_() const noexcept {
#ifdef _OPENMP
    // n² element loops: worthwhile only for a few hundred orbitals and more.
    constexpr int kParallelOrbitals = 256;
    if (system_.n_orbitals >= kParallelOrbitals)
        return std::max(1, omp_get_max_threads());
#endif
    return 1;
}

int NDDOSolver::force_thread_count_() const noexcept {
#ifdef _OPENMP
    // Pair work is quadratic: below this the thread start-up dominates.
    constexpr int kParallelPairAtoms = 16;
    if (n_atoms_ >= kParallelPairAtoms)
        return std::max(1, omp_get_max_threads());
#endif
    return 1;
}

void NDDOSolver::accumulate_pair_force_(
    int i, int j, const std::array<const StorageReal*, 3>& coordinates,
    NDDOSystem& pair, std::vector<double>& forces) const {
    std::array<double, 36> pair_density{}, pair_spin{};
    std::array<std::array<StorageReal, 2>, 3> positions;
    const auto& ai = system_.atoms[i];
    const auto& aj = system_.atoms[j];
    pair.atoms[0] = aj;
    pair.atoms[0].first_ao = 0;
    pair.atoms[0].last_ao = aj.n_orbitals - 1;
    pair.atoms[1] = ai;
    pair.atoms[1].first_ao = aj.n_orbitals;
    pair.atoms[1].last_ao = aj.n_orbitals + ai.n_orbitals - 1;
    pair.n_orbitals = aj.n_orbitals + ai.n_orbitals;
    const int n = pair.n_orbitals;
    for (int mu = 0; mu < n; ++mu) {
        const int global_mu = mu < aj.n_orbitals
                                  ? aj.first_ao + mu
                                  : ai.first_ao + mu - aj.n_orbitals;
        for (int nu = 0; nu <= mu; ++nu) {
            const int global_nu = nu < aj.n_orbitals
                                      ? aj.first_ao + nu
                                      : ai.first_ao + nu - aj.n_orbitals;
            const auto local = packed_index(mu, nu);
            const auto global = packed_index(global_mu, global_nu);
            pair_density[local] = scf_density_[global];
            pair_spin[local] = scf_spin_[global];
        }
    }
    for (int d = 0; d < 3; ++d) {
        positions[d][0] = coordinates[d][j];
        positions[d][1] = coordinates[d][i];
    }
    const std::array<double, 3> xi{positions[0][1], positions[1][1],
                                   positions[2][1]};
    const std::array<double, 3> xj{positions[0][0], positions[1][0],
                                   positions[2][0]};
    const auto dh = internal::hcore_pair_derivative(pair, xi, xj);
    const auto dw = internal::two_center_integral_derivative(
        pair, positions[0].data(), positions[1].data(), positions[2].data(), 1,
        0);
    for (int d = 0; d < 3; ++d) {
        double gradient = dh[d].nuclear_repulsion_ev;
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu <= mu; ++nu) {
                const auto k = packed_index(mu, nu);
                gradient += (mu == nu ? 1.0 : 2.0) * pair_density[k]
                            * dh[d].packed_hcore[k];
            }
        int ip = 0;
        for (int a = 0; a < ai.n_orbitals; ++a)
            for (int b = 0; b <= a; ++b, ++ip) {
                const int aa = aj.n_orbitals + a, bb = aj.n_orbitals + b;
                const double mi = a == b ? 1.0 : 2.0;
                int jp = 0;
                for (int c = 0; c < aj.n_orbitals; ++c)
                    for (int e = 0; e <= c; ++e, ++jp) {
                        const double mj = c == e ? 1.0 : 2.0;
                        const double coulomb =
                            pair_density[packed_index(aa, bb)]
                            * pair_density[packed_index(c, e)];
                        const double exchange =
                            0.25
                            * (pair_density[packed_index(aa, c)]
                                   * pair_spin[packed_index(bb, e)]
                               + pair_density[packed_index(aa, e)]
                                     * pair_spin[packed_index(bb, c)]
                               + pair_density[packed_index(bb, c)]
                                     * pair_spin[packed_index(aa, e)]
                               + pair_density[packed_index(bb, e)]
                                     * pair_spin[packed_index(aa, c)]);
                        gradient +=
                            mi * mj * (coulomb - exchange)
                            * dw[d]
                                [ip * (aj.n_orbitals * (aj.n_orbitals + 1) / 2)
                                 + jp];
                    }
            }
        if (!std::isfinite(gradient))
            throw std::runtime_error("NDDO non-finite analytic force");
        forces[3 * i + d] -= gradient;
        forces[3 * j + d] += gradient;
    }
}

void NDDOSolver::validate_supported_atoms_(const cpu::Frame& frame) const {
    if (config_.method != ATV::semiempirical::SemiEmpiricalMethod::AM1
        && config_.method != ATV::semiempirical::SemiEmpiricalMethod::RM1
        && config_.method != ATV::semiempirical::SemiEmpiricalMethod::MNDO
        && config_.method != ATV::semiempirical::SemiEmpiricalMethod::PM3) {
        throw std::runtime_error(
            "NDDOSolver: unsupported semiempirical method");
    }
    if ((config_.method == ATV::semiempirical::SemiEmpiricalMethod::MNDO
         || config_.method == ATV::semiempirical::SemiEmpiricalMethod::PM3)
        && (config_.charge != 0 || config_.multiplicity != 1))
        throw std::invalid_argument(
            "MNDO/PM3 currently require neutral closed-shell molecules");
    if (frame.nlocal() < 1)
        throw std::invalid_argument("NDDO requires at least one atom");
    const std::array coordinates{frame.span(kAtom_x), frame.span(kAtom_y),
                                 frame.span(kAtom_z)};
    for (int i = 0; i < frame.nlocal(); ++i)
        for (const auto& axis : coordinates)
            if (!std::isfinite(axis[i]))
                throw std::invalid_argument("NDDO non-finite coordinate");
    const int* __restrict sp_type = frame.span(kAtom_atomic_number);
    for (int i = 0; i < frame.nlocal(); ++i) {
        const int z = sp_type[i];
        if (!ATV::semiempirical::nddo_has_parameter(config_.method, z)) {
            throw std::runtime_error(
                "NDDO parameters unavailable for atomic number "
                + std::to_string(z));
        }
    }
    // Coincident and overlapping nuclei have no SCF solution: refuse them by
    // name here, before any integral or Fock work.
    ATV::semiempirical::require_nddo_contacts(
        config_.method, static_cast<std::size_t>(frame.nlocal()), sp_type,
        coordinates[0], coordinates[1], coordinates[2]);
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
