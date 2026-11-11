/// @file core/periodic_torsion.cpp
/// @brief Validated torsion term tables and the per-style term builders.

#include "atomiverse/core/periodic_torsion.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ATV {

namespace {

/// A phase within one ulp-scale of 0 or π gets an exact zero sine, so
/// `cvff` (d = ±1) is a pure cosine series like LAMMPS's integer sign.
double clean_sin(double delta) {
    const double s = std::sin(delta);
    return std::abs(s) < 1.0e-14 ? 0.0 : s;
}

} // namespace

TorsionTable::TorsionTable(const std::vector<std::vector<TorsionTerm>>& terms,
                           const char* style) {
    if (terms.empty())
        throw std::invalid_argument(std::string(style)
                                    + ": at least one type is required");
    offset.reserve(terms.size() + 1);
    offset.push_back(0);
    active.assign(terms.size(), 0);
    for (std::size_t t = 0; t < terms.size(); ++t) {
        for (const TorsionTerm& term : terms[t]) {
            if (!std::isfinite(term.K) || !std::isfinite(term.delta))
                throw std::invalid_argument(std::string(style)
                                            + ": non-finite K or phase at type "
                                            + std::to_string(t));
            if (term.n < 0 || term.n > kMaxTorsionMultiplicity)
                throw std::invalid_argument(
                    std::string(style) + ": multiplicity "
                    + std::to_string(term.n) + " at type " + std::to_string(t)
                    + " is outside [0, 6]");
            K.push_back(term.K);
            n.push_back(term.n);
            cos_delta.push_back(std::cos(term.delta));
            sin_delta.push_back(clean_sin(term.delta));
            if (term.K != 0.0)
                active[t] = 1;
        }
        offset.push_back(static_cast<int>(K.size()));
    }
}

std::vector<std::vector<TorsionTerm>>
charmm_torsion_terms(const std::vector<double>& K, const std::vector<int>& n,
                     const std::vector<double>& delta) {
    if (K.empty() || K.size() != n.size() || K.size() != delta.size())
        throw std::invalid_argument(
            "DihedralCharmm: K, n, delta must be non-empty and same size");
    std::vector<std::vector<TorsionTerm>> terms(K.size());
    for (std::size_t t = 0; t < K.size(); ++t) {
        if (n[t] < 1)
            throw std::invalid_argument(
                "DihedralCharmm: multiplicity n must be >= 1 (type "
                + std::to_string(t) + ")");
        terms[t].push_back(TorsionTerm{K[t], n[t], delta[t]});
    }
    return terms;
}

std::vector<std::vector<TorsionTerm>>
cvff_torsion_terms(const std::vector<double>& K, const std::vector<int>& d,
                   const std::vector<int>& n) {
    if (K.empty() || K.size() != d.size() || K.size() != n.size())
        throw std::invalid_argument(
            "ImproperCvff: K, d, n must be non-empty and same size");
    std::vector<std::vector<TorsionTerm>> terms(K.size());
    for (std::size_t t = 0; t < K.size(); ++t) {
        if (d[t] != 1 && d[t] != -1)
            throw std::invalid_argument("ImproperCvff: d must be +1 or -1 "
                                        "(type "
                                        + std::to_string(t) + ")");
        terms[t].push_back(TorsionTerm{K[t], n[t], d[t] == 1 ? 0.0 : M_PI});
    }
    return terms;
}

} // namespace ATV
