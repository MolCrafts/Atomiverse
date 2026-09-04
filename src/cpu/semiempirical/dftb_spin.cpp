/// @file src/cpu/semiempirical/dftb_spin.cpp
/// HOAO collapse of shell-resolved collinear DFTB spin constants (private).
#include "atomiverse/cpu/semiempirical/internal/dftb_spin.h"

#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

std::vector<double>
collapse_spin_w_ev(const std::vector<basis_data::DftbSpinConstant>& spin_w_ha,
                   const std::vector<int>& z_per_atom) {
    if (spin_w_ha.empty())
        throw std::invalid_argument(
            "collapse_spin_w_ev: empty spin constant table");

    std::vector<double> out(z_per_atom.size(), 0.0);
    for (std::size_t a = 0; a < z_per_atom.size(); ++a) {
        const int z = z_per_atom[a];
        const basis_data::DftbSpinConstant* row = nullptr;
        for (const auto& r : spin_w_ha) {
            if (r.z == z) {
                row = &r;
                break;
            }
        }
        if (row == nullptr)
            throw std::invalid_argument(
                "collapse_spin_w_ev: no spin constant for Z="
                + std::to_string(z));
        double w_ha = 0.0;
        if (row->n_shell == 1 || z == 1)
            w_ha = row->w_ha[0][0];
        else if (row->n_shell == 2)
            w_ha = row->w_ha[1][1];
        else
            throw std::invalid_argument(
                "collapse_spin_w_ev: unsupported n_shell="
                + std::to_string(row->n_shell) + " for Z=" + std::to_string(z));
        out[a] = w_ha * units::kHartreeInEv;
    }
    return out;
}

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
