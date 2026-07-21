/// @file src/cpu/semiempirical/nddo_solver.cpp
/// @brief NDDO semiempirical inner solver scaffold.

#include "atomiverse/cpu/semiempirical/nddo_solver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/semiempirical/am1_parameters.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"
#include "atomiverse/cpu/semiempirical/nddo_density.h"
#include "atomiverse/cpu/semiempirical/nddo_fock.h"
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"
#include "atomiverse/cpu/semiempirical/nddo_matrix.h"
#include "atomiverse/cpu/semiempirical/rm1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

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
        out = std::max(out, std::abs(a[i] - b[i]));
    }
    return out;
}

} // namespace

NDDOSolver::NDDOSolver(NDDOConfig config) : config_(config) {
    result_.method = config_.method;
    result_.reference = (config_.method == SemiEmpiricalMethod::RM1)
                            ? "Internal RM1/NDDO reference data (Rocha 2006)"
                            : "Internal AM1/NDDO reference data";
}

void NDDOSolver::build(const cpu::Frame& reference) {
    validate_supported_atoms_(reference);
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
    state_.initial_one_center_fock = NDDOFockBuilder{}.build(
        system_, reference, state_.packed_hcore, state_.initial_packed_density,
        state_.initial_spin_density);

    n_atoms_ = reference.nlocal();
    result_.converged = false;
    result_.iterations = 0;
    result_.electronic_energy_hartree = 0.0;
    result_.total_energy_ev = 0.0;
    result_.heat_of_formation_kcal_mol = 0.0;
    result_.mulliken_charges.assign(static_cast<std::size_t>(n_atoms_), 0.0);
    result_.forces_ev_a.clear();
    density_.clear();
    spin_density_.clear();
    has_density_ = false;
    built_ = true;
}

NDDOResult& NDDOSolver::run(const cpu::Frame& frame) {
    if (!built_) {
        throw std::runtime_error("NDDOSolver::run called before build()");
    }
    if (frame.nlocal() != n_atoms_) {
        throw std::runtime_error(
            "NDDOSolver::run requires a fixed atom count after build()");
    }
    validate_supported_atoms_(frame);

    if (system_.n_alpha != system_.n_beta) {
        throw std::runtime_error(
            "NDDOSolver AM1 SCF currently supports closed-shell RHF only");
    }

    result_.forces_ev_a.clear();

    // Hoist geometry spans once per run (SCF reuses them every iteration).
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);

    const auto hcore = NDDOHCoreBuilder{}.build(system_, sp_x, sp_y, sp_z);
    // Warm-start from the last converged density when available (AIMD / opt).
    // Note: do not clear has_density_ before this branch — that would force a
    // cold start every step and break multi-step MD/Minimize SCF stability.
    std::vector<double> density =
        has_density_ ? density_ : state_.initial_packed_density;
    std::vector<double> spin_density = scaled_density(density, 0.5);
    has_density_ = false; // cleared until this run reconverges
    std::vector<double> fock = NDDOFockBuilder{}.build(
        system_, sp_x, sp_y, sp_z, hcore.packed_hcore, density, spin_density);
    double previous_energy_ev = std::numeric_limits<double>::infinity();

    for (int iter = 1; iter <= config_.max_iter; ++iter) {
        const auto orbitals =
            NDDODiagonalizer{}.diagonalize_packed(fock, system_.n_orbitals);

        NDDODensityInput density_input;
        density_input.mdim = system_.n_orbitals;
        density_input.norbs = system_.n_orbitals;
        density_input.n_occupied = system_.n_beta;
        density_input.n_fractional = system_.n_beta;
        density_input.full_occupancy = 2.0;
        density_input.fractional_occupancy = 0.0;
        const auto next_density = NDDODensityBuilder{}.build_packed(
            orbitals.coefficients_row_major, density_input);
        const double delta = max_abs_delta(density, next_density);

        density = next_density;
        spin_density = scaled_density(density, 0.5);
        fock =
            NDDOFockBuilder{}.build(system_, sp_x, sp_y, sp_z,
                                    hcore.packed_hcore, density, spin_density);
        const double electronic_ev = electronic_energy_ev(
            system_.n_orbitals, density, hcore.packed_hcore, fock);
        const double total_ev = electronic_ev + hcore.nuclear_repulsion_ev;
        const double energy_delta = std::abs(total_ev - previous_energy_ev);

        result_.iterations = iter;
        result_.total_energy_ev = total_ev;
        result_.electronic_energy_hartree = total_ev * units::kEvInHartree;
        result_.mulliken_charges =
            atomic_charges_from_packed_density(system_, density);
        if (delta <= config_.density_tol
            || energy_delta <= config_.energy_tol) {
            result_.converged = true;
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
                sum_atheat +=
                    (config_.method == SemiEmpiricalMethod::RM1)
                        ? rm1_atomic_heat_kcal_mol(atom.atomic_number)
                        : am1_atomic_heat_kcal_mol(atom.atomic_number);
            }
            result_.heat_of_formation_kcal_mol =
                (total_ev - sum_eisol) * kAm1EvToKcalMol + sum_atheat;
            density_ = std::move(density);
            spin_density_ = std::move(spin_density);
            has_density_ = true;
            return result_;
        }
        previous_energy_ev = total_ev;
    }

    result_.converged = false;
    throw std::runtime_error("NDDOSolver AM1 SCF did not converge");
}

double NDDOSolver::energy_at_fixed_density_ev_(const StorageReal* sp_x,
                                               const StorageReal* sp_y,
                                               const StorageReal* sp_z) const {
    const auto hcore = NDDOHCoreBuilder{}.build(system_, sp_x, sp_y, sp_z);
    const auto fock = NDDOFockBuilder{}.build(
        system_, sp_x, sp_y, sp_z, hcore.packed_hcore, density_, spin_density_);
    return electronic_energy_ev(system_.n_orbitals, density_,
                                hcore.packed_hcore, fock)
           + hcore.nuclear_repulsion_ev;
}

NDDOResult& NDDOSolver::compute_forces(cpu::Frame& frame,
                                       double step_angstrom) {
    if (!built_) {
        throw std::runtime_error(
            "NDDOSolver::compute_forces called before build()");
    }
    if (!has_density_ || !result_.converged) {
        throw std::runtime_error(
            "NDDOSolver::compute_forces requires a converged run() density");
    }
    if (frame.nlocal() != n_atoms_) {
        throw std::runtime_error(
            "NDDOSolver::compute_forces atom count mismatch");
    }
    if (!(step_angstrom > 0.0) || !std::isfinite(step_angstrom)) {
        throw std::invalid_argument(
            "NDDOSolver::compute_forces step_angstrom must be positive finite");
    }

    // Local coordinate copies — displace without Frame re-resolution storms.
    const StorageReal* __restrict fx = frame.span(kAtom_x);
    const StorageReal* __restrict fy = frame.span(kAtom_y);
    const StorageReal* __restrict fz = frame.span(kAtom_z);
    std::vector<StorageReal> x(fx, fx + n_atoms_);
    std::vector<StorageReal> y(fy, fy + n_atoms_);
    std::vector<StorageReal> z(fz, fz + n_atoms_);

    result_.forces_ev_a.assign(static_cast<std::size_t>(3 * n_atoms_), 0.0);
    const double two_h = 2.0 * step_angstrom;

    auto energy_local = [&]() {
        return energy_at_fixed_density_ev_(x.data(), y.data(), z.data());
    };

    // Translational invariance: F_N = −Σ_{i<N} F_i, so only 3(N−1) independent
    // central differences when N ≥ 2 (exact for isolated-molecule AM1).
    const int n_fd = n_atoms_ >= 2 ? n_atoms_ - 1 : n_atoms_;
    for (int i = 0; i < n_fd; ++i) {
        for (int d = 0; d < 3; ++d) {
            StorageReal* coord =
                d == 0 ? x.data() : (d == 1 ? y.data() : z.data());
            const StorageReal x0 = coord[i];
            coord[i] = static_cast<StorageReal>(static_cast<double>(x0)
                                                + step_angstrom);
            const double e_plus = energy_local();
            coord[i] = static_cast<StorageReal>(static_cast<double>(x0)
                                                - step_angstrom);
            const double e_minus = energy_local();
            coord[i] = x0;
            // F = −∂E/∂R  [eV/Å]
            result_.forces_ev_a[static_cast<std::size_t>(3 * i + d)] =
                -(e_plus - e_minus) / two_h;
        }
    }
    if (n_atoms_ >= 2) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int i = 0; i < n_atoms_ - 1; ++i) {
            sx += result_.forces_ev_a[static_cast<std::size_t>(3 * i + 0)];
            sy += result_.forces_ev_a[static_cast<std::size_t>(3 * i + 1)];
            sz += result_.forces_ev_a[static_cast<std::size_t>(3 * i + 2)];
        }
        const int last = n_atoms_ - 1;
        result_.forces_ev_a[static_cast<std::size_t>(3 * last + 0)] = -sx;
        result_.forces_ev_a[static_cast<std::size_t>(3 * last + 1)] = -sy;
        result_.forces_ev_a[static_cast<std::size_t>(3 * last + 2)] = -sz;
    }
    return result_;
}

void NDDOSolver::validate_supported_atoms_(const cpu::Frame& frame) const {
    if (config_.method != SemiEmpiricalMethod::AM1
        && config_.method != SemiEmpiricalMethod::RM1) {
        throw std::runtime_error(
            "NDDOSolver: unsupported semiempirical method");
    }
    const int* __restrict sp_type = frame.span(kAtom_type);
    for (int i = 0; i < frame.nlocal(); ++i) {
        const int z = sp_type[i];
        if (!nddo_has_parameter(config_.method, z)) {
            throw std::runtime_error(
                "NDDO parameters unavailable for atomic number "
                + std::to_string(z));
        }
    }
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
