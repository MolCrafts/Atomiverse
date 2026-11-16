/// @file src/cpu/dynamics/gauss_constraint.cpp
/// @brief Gauss least-constraint geometry (see the header).
#include "atomiverse/cpu/dynamics/internal/gauss_constraint.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ATV::cpu::md::internal {

GaussConstraint::GaussConstraint(GaussKind kind, int axis) {
    if (axis < 0 || axis > 2)
        throw std::invalid_argument(
            "GaussThermostat: axis must be 0, 1 or 2, got "
            + std::to_string(axis));
    if (kind == GaussKind::Flow)
        mode_[axis] = GaussAxis::Species;
    else if (kind == GaussKind::Transverse)
        mode_[axis] = GaussAxis::Free;
}

void GaussConstraint::bind(int n, const int* type) {
    int t_max = -1;
    for (int i = 0; i < n; ++i) {
        if (type[i] < 0)
            throw std::invalid_argument(
                "GaussThermostat: atom types must be >= 0");
        t_max = std::max(t_max, type[i]);
    }
    species_of_type_.assign(static_cast<std::size_t>(t_max + 1), -1);
    n_species_ = 0;
    for (int i = 0; i < n; ++i) {
        int& s = species_of_type_[static_cast<std::size_t>(type[i])];
        if (s < 0)
            s = n_species_++;
    }
    n_atoms_ = n;
}

long long GaussConstraint::dof() const {
    long long count = 0;
    for (const GaussAxis m : mode_) {
        if (m == GaussAxis::Barycentric)
            count += n_atoms_ - 1;
        else if (m == GaussAxis::Species)
            count += n_atoms_ - n_species_;
    }
    return count;
}

void GaussConstraint::peculiar(int n, const double* mass, const int* type,
                               ConstRows3 v, Rows3 w) const {
    std::vector<int> species(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int t = type[i];
        const int s = t >= 0 && t < static_cast<int>(species_of_type_.size())
                          ? species_of_type_[static_cast<std::size_t>(t)]
                          : -1;
        if (s < 0)
            throw std::invalid_argument(
                "GaussThermostat: atom type " + std::to_string(t)
                + " was not present when the thermostat was bound");
        species[static_cast<std::size_t>(i)] = s;
    }
    std::vector<double> group_mass(static_cast<std::size_t>(n_species_));
    std::vector<double> group_momentum(static_cast<std::size_t>(n_species_));
    for (int d = 0; d < 3; ++d) {
        if (mode_[d] == GaussAxis::Free) {
            std::fill_n(w[d], n, 0.0);
            continue;
        }
        const bool by_species = mode_[d] == GaussAxis::Species;
        std::fill(group_mass.begin(), group_mass.end(), 0.0);
        std::fill(group_momentum.begin(), group_momentum.end(), 0.0);
        for (int i = 0; i < n; ++i) {
            const std::size_t g =
                by_species ? static_cast<std::size_t>(species[i]) : 0;
            group_mass[g] += mass[i];
            group_momentum[g] += mass[i] * v[d][i];
        }
        for (int i = 0; i < n; ++i) {
            const std::size_t g =
                by_species ? static_cast<std::size_t>(species[i]) : 0;
            w[d][i] = v[d][i] - group_momentum[g] / group_mass[g];
        }
    }
}

double GaussConstraint::multiplier(int n, const double* mass, ConstRows3 w,
                                   ConstRows3 a) {
    double num = 0.0, den = 0.0;
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n; ++i) {
            num += mass[i] * w[d][i] * a[d][i];
            den += mass[i] * w[d][i] * w[d][i];
        }
    if (!(den > 0.0))
        throw std::runtime_error(
            "GaussThermostat: the constrained kinetic energy is zero; give "
            "the atoms peculiar velocities before the run");
    return num / den;
}

double GaussConstraint::twice_kinetic(int n, const double* mass, ConstRows3 w) {
    double sum = 0.0;
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n; ++i)
            sum += mass[i] * w[d][i] * w[d][i];
    return sum;
}

} // namespace ATV::cpu::md::internal
