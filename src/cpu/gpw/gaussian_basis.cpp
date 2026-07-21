/// @file src/cpu/gpw/gaussian_basis.cpp
/// @brief GaussianBasis lookup constructor over the in-tree static tables.

#include "atomiverse/cpu/gpw/gaussian_basis.h"

#include <stdexcept>

#include "atomiverse/cpu/gpw/basis_data/gpw_tables.h"

namespace ATV {
namespace cpu {
namespace gpw {

GaussianBasis::GaussianBasis(molrs::Element element, std::string_view name)
    : element(element), name(name) {
    for (const auto* row : basis_data::kGaussianBases) {
        if (row->z != static_cast<int>(element) || name != row->name)
            continue;
        this->sets.reserve(row->n_sets);
        for (unsigned long s = 0; s < row->n_sets; ++s) {
            const auto& in = row->sets[s];
            GaussianBasisSet set;
            set.n = in.n;
            set.lmin = in.lmin;
            set.lmax = in.lmax;
            set.ncoef = in.ncoef;
            set.exps.assign(in.exps, in.exps + in.n_exps);
            set.nshell.assign(in.nshell, in.nshell + in.n_nshell);
            set.coef.assign(in.coef, in.coef + in.n_coef);
            sets.push_back(std::move(set));
        }
        return;
    }
    std::string supported;
    for (const auto* row : basis_data::kGaussianBases)
        supported += std::string(" ") + row->symbol + ":" + row->name;
    throw std::runtime_error("GaussianBasis: no in-tree entry for Z="
                             + std::to_string(static_cast<int>(element))
                             + " name='" + std::string(name)
                             + "'. Supported:" + supported);
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
