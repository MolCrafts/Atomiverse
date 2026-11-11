/// @file src/semiempirical/nddo_geometry.cpp
/// @brief Covalent radii and the NDDO physical-contact refusal.
#include "atomiverse/semiempirical/nddo_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/semiempirical/nddo_parameters.h"

namespace ATV::semiempirical {

namespace {

/// Cordero 2008 single-bond covalent radii [Å], index = Z − 1, Z = 1..86.
constexpr std::array<double, 86> kCovalentRadiusAngstrom{
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, // H..Ne
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,             // Na..Ar
    2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26,       // K..Co
    1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,       // Ni..Kr
    2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42,       // Rb..Rh
    1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,       // Pd..Xe
    2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,       // Cs..Eu
    1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,             // Gd..Lu
    1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,       // Hf..Hg
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50};                        // Tl..Rn

struct Contact {
    std::size_t i, j;
    double distance, limit;
};

std::string format_contact(SemiEmpiricalMethod method, const int* z,
                           const Contact& c) {
    char buffer[160];
    std::snprintf(buffer, sizeof buffer, "%s%zu-%s%zu %.3f A (limit %.3f A)",
                  nddo_parameter(method, z[c.i])->symbol, c.i,
                  nddo_parameter(method, z[c.j])->symbol, c.j, c.distance,
                  c.limit);
    return buffer;
}

} // namespace

double covalent_radius_angstrom(int z) {
    if (z < 1 || z > static_cast<int>(kCovalentRadiusAngstrom.size()))
        throw std::invalid_argument(
            "no covalent radius tabulated for atomic number "
            + std::to_string(z) + " (table covers Z = 1..86)");
    return kCovalentRadiusAngstrom[static_cast<std::size_t>(z - 1)];
}

void require_nddo_contacts(SemiEmpiricalMethod method, std::size_t n,
                           const int* z, const StorageReal* x,
                           const StorageReal* y, const StorageReal* z_coord) {
    std::vector<double> radius(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!nddo_has_parameter(method, z[i]))
            throw std::invalid_argument(
                std::string(nddo_method_name(method))
                + " has no parameters for atomic number "
                + std::to_string(z[i]));
        radius[i] = covalent_radius_angstrom(z[i]);
    }
    std::vector<Contact> contacts;
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const double limit = kNddoContactFraction * (radius[i] + radius[j]);
            const double dx = static_cast<double>(x[i]) - x[j];
            const double dy = static_cast<double>(y[i]) - y[j];
            const double dz = static_cast<double>(z_coord[i]) - z_coord[j];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 >= limit * limit))
                contacts.push_back({j, i, std::sqrt(r2), limit});
        }
    }
    if (contacts.empty())
        return;
    std::sort(contacts.begin(), contacts.end(),
              [](const Contact& a, const Contact& b) {
                  return a.distance / a.limit < b.distance / b.limit;
              });
    char header[200];
    std::snprintf(header, sizeof header,
                  "%s: impossible geometry, refused before the SCF: %zu atom "
                  "pair(s) closer than %.2g x the sum of their covalent "
                  "radii. Closest (0-based atom index): ",
                  nddo_method_name(method), contacts.size(),
                  kNddoContactFraction);
    std::string message = header;
    const std::size_t shown =
        std::min(contacts.size(), kNddoContactReportLimit);
    for (std::size_t k = 0; k < shown; ++k) {
        if (k)
            message += ", ";
        message += format_contact(method, z, contacts[k]);
    }
    if (contacts.size() > shown)
        message +=
            ", ... (" + std::to_string(contacts.size() - shown) + " more)";
    message += ". Fix the input coordinates; no SCF can converge on "
               "overlapping nuclei.";
    throw std::invalid_argument(message);
}

} // namespace ATV::semiempirical
