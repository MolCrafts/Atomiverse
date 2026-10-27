// Compare one production smearing translation unit with actual QE pins.
#include "atomiverse/electronic/smearing.h"
#include "tests/cpu/planewave/smearing_qe_pins.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace ATV::electronic;

void number(double value) {
    if (std::isfinite(value))
        std::cout << value;
    else
        std::cout << "null";
}

int main() {
    std::cout << std::setprecision(17) << "{\"points\":[";
    bool first = true;
    for (const auto& p : qe_smearing_pin::points) {
        if (!first)
            std::cout << ',';
        first = false;
        const auto kind = static_cast<ATV::electronic::Smearing>(p.kind);
        std::cout << "{\"f\":";
        number(smear_f_spinless(p.x, kind, p.order));
        std::cout << ",\"s\":";
        number(smear_entropy_spinor(p.x, 0, 1, kind, p.order));
        std::cout << '}';
    }
    std::cout << "],\"spectra\":[";
    first = true;
    for (const auto& p : qe_smearing_pin::spectra) {
        if (!first)
            std::cout << ',';
        first = false;
        const auto kind = static_cast<ATV::electronic::Smearing>(p.kind);
        const std::vector<ATV::AccReal> eig(p.evals.begin(), p.evals.end());
        const std::vector<ATV::AccReal> weights(eig.size(), 1);
        const auto fixed = fill_occupations(eig, weights, p.mu, p.width, kind,
                                            p.order, p.degen);
        const auto mu = find_fermi_level(eig, weights, p.target, p.width, kind,
                                         p.order, p.degen);
        const auto fill =
            fill_occupations(eig, weights, mu, p.width, kind, p.order, p.degen);
        std::cout << "{\"at_qe_mu\":{\"n\":";
        number(fixed.n_occ);
        std::cout << ",\"s\":";
        number(fixed.entropy);
        std::cout << "},\"native_root\":{\"mu\":";
        number(mu);
        std::cout << ",\"n\":";
        number(fill.n_occ);
        std::cout << ",\"s\":";
        number(fill.entropy);
        std::cout << "}}";
    }
    std::cout << "]}\n";
}
