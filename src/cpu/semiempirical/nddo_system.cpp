/// @file src/cpu/semiempirical/nddo_system.cpp
/// @brief NDDO reference-shaped NDDO system indexing and population helpers.

#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/semiempirical/am1_parameters.h"
#include "atomiverse/semiempirical/rm1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

int am1_orbital_count(int z) noexcept {
    const auto* p = ATV::semiempirical::am1_parameter(z);
    return p == nullptr ? 0 : p->n_orbitals;
}

double am1_core_charge(int z) noexcept {
    const auto* p = ATV::semiempirical::am1_parameter(z);
    return p == nullptr ? 0.0 : p->core_charge;
}

NDDOSystem::NDDOSystem(const cpu::Frame& frame, int charge_in, int multiplicity,
                       ATV::semiempirical::SemiEmpiricalMethod method_in) {
    if (multiplicity < 1) {
        throw std::runtime_error("NDDO multiplicity must be positive");
    }

    const char* method_name = ATV::semiempirical::nddo_method_name(method_in);

    charge = charge_in;
    method = method_in;
    atoms.reserve(static_cast<std::size_t>(frame.nlocal()));

    int next_ao = 0;
    double electron_count = -static_cast<double>(charge_in);
    const int* __restrict sp_type = frame.span(kAtom_type);
    for (int i = 0; i < frame.nlocal(); ++i) {
        const int z = sp_type[i];
        const auto* p = ATV::semiempirical::nddo_parameter(method_in, z);
        if (p == nullptr) {
            throw std::runtime_error(
                std::string(method_name)
                + " parameters unavailable for atomic number "
                + std::to_string(z));
        }
        if (p->n_orbitals <= 0 || p->core_charge == 0.0) {
            throw std::runtime_error(
                std::string(method_name)
                + " orbital/core data unavailable for atomic number "
                + std::to_string(z));
        }
        NDDOAtom atom;
        atom.atomic_number = z;
        atom.first_ao = next_ao;
        atom.n_orbitals = p->n_orbitals;
        atom.last_ao = next_ao + p->n_orbitals - 1;
        atom.core_charge = p->core_charge;
        atom.params = p;
        atoms.push_back(atom);
        next_ao += p->n_orbitals;
        electron_count += p->core_charge;
    }

    n_orbitals = next_ao;
    n_electrons = static_cast<int>(std::lround(electron_count));
    if (n_electrons < 0 || n_electrons > 2 * n_orbitals) {
        throw std::runtime_error(
            "NDDO electron count is outside the AO capacity");
    }

    const int unpaired = multiplicity - 1;
    if ((n_electrons + unpaired) % 2 != 0) {
        throw std::runtime_error(
            "NDDO multiplicity is inconsistent with the electron count");
    }
    n_alpha = (n_electrons + unpaired) / 2;
    n_beta = n_electrons - n_alpha;
    if (n_beta < 0 || n_alpha > n_orbitals) {
        throw std::runtime_error(
            "NDDO spin electron count is outside AO capacity");
    }
}

std::vector<double> atomic_populations_from_packed_density(
    const NDDOSystem& system, const std::vector<double>& packed_density) {
    const std::size_t expected =
        static_cast<std::size_t>(system.n_orbitals)
        * static_cast<std::size_t>(system.n_orbitals + 1) / 2u;
    if (packed_density.size() != expected) {
        throw std::runtime_error(
            "packed density size does not match NDDOSystem");
    }

    std::vector<double> populations(system.atoms.size(), 0.0);
    for (std::size_t a = 0; a < system.atoms.size(); ++a) {
        const auto& atom = system.atoms[a];
        double q = 0.0;
        for (int ao = atom.first_ao; ao <= atom.last_ao; ++ao) {
            q += packed_density[packed_index(ao, ao)];
        }
        populations[a] = q;
    }
    return populations;
}

std::vector<double>
atomic_charges_from_packed_density(const NDDOSystem& system,
                                   const std::vector<double>& packed_density) {
    auto populations =
        atomic_populations_from_packed_density(system, packed_density);
    for (std::size_t i = 0; i < populations.size(); ++i) {
        populations[i] = system.atoms[i].core_charge - populations[i];
    }
    return populations;
}

std::vector<double> initial_diagonal_density_guess(const NDDOSystem& system) {
    std::vector<double> diag(static_cast<std::size_t>(system.n_orbitals), 0.0);
    const double yy = static_cast<double>(system.charge)
                      / (static_cast<double>(system.n_orbitals) + 1.0e-10);
    for (const auto& atom : system.atoms) {
        if (atom.n_orbitals == 1) {
            diag[static_cast<std::size_t>(atom.first_ao)] =
                atom.core_charge - yy;
        } else if (atom.n_orbitals == 4) {
            const double w = atom.core_charge * 0.25 - yy;
            for (int ao = atom.first_ao; ao <= atom.last_ao; ++ao) {
                diag[static_cast<std::size_t>(ao)] = w;
            }
        } else if (atom.n_orbitals == 9) {
            const double w = atom.core_charge * 0.25 - yy;
            for (int ao = atom.first_ao; ao < atom.first_ao + 4; ++ao) {
                diag[static_cast<std::size_t>(ao)] = w;
            }
            for (int ao = atom.first_ao + 4; ao <= atom.last_ao; ++ao) {
                diag[static_cast<std::size_t>(ao)] = -yy;
            }
        }
    }
    return diag;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
