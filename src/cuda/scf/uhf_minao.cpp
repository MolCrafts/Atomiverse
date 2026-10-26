/// @file cuda/scf/uhf_minao.cpp
/// @brief Pack geometry-independent MINAO element metadata for device setup.
///
/// No geometry, overlap integrals, linear solve, or density is evaluated here.
#include "atomiverse/cuda/scf/internal/uhf_guess_data.h"

#include "atomiverse/cpu/scf/minao_data.h"

#include <limits>
#include <stdexcept>

namespace ATV::cuda::scf::internal {

UhfMinaoData::UhfMinaoData(const int* atomic_numbers, int natm) {
    if (!atomic_numbers || natm <= 0)
        throw std::invalid_argument("CUDA UHF MINAO: missing atomic numbers");
    for (int a = 0; a < natm; ++a)
        if (!cpu::scf::minao::has_minao(atomic_numbers[a]))
            return;
    for (int a = 0; a < natm; ++a) {
        const auto& element =
            *cpu::scf::minao::element_basis(atomic_numbers[a]);
        int ao_in_element = 0;
        for (int sh = 0; sh < element.n_shells; ++sh) {
            const auto& source = element.shells[sh];
            if (source.l < 0 || source.l > 1 || source.n_prim <= 0
                || source.n_contract <= 0
                || source.n_prim
                       > std::numeric_limits<int>::max() / source.n_prim)
                throw std::invalid_argument("CUDA UHF MINAO: table requires an "
                                            "unsupported contraction");
            for (int c = 0; c < source.n_contract; ++c) {
                if (alpha.size() > static_cast<std::size_t>(
                        std::numeric_limits<int>::max() - source.n_prim)
                    || angular.size() >= static_cast<std::size_t>(
                           std::numeric_limits<int>::max()))
                    throw std::overflow_error(
                        "CUDA UHF MINAO: table exceeds device index range");
                const int shell = static_cast<int>(angular.size());
                angular.push_back(source.l);
                atom.push_back(a);
                for (int p = 0; p < source.n_prim; ++p) {
                    alpha.push_back(source.alpha[p]);
                    coefficient.push_back(
                        source.coeff[p * source.n_contract + c]);
                }
                primitive_offset.push_back(static_cast<int>(alpha.size()));
                for (int cart = 0; cart < 2 * source.l + 1; ++cart) {
                    ao_shell.push_back(shell);
                    ao_cart.push_back(cart);
                    occupation.push_back(element.occupations[ao_in_element++]);
                }
            }
        }
        if (ao_in_element != element.n_ao)
            throw std::logic_error("CUDA UHF MINAO: inconsistent AO metadata");
    }
}

} // namespace ATV::cuda::scf::internal
