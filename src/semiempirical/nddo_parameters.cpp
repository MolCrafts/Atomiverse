#include "atomiverse/semiempirical/nddo_parameters.h"
#include "atomiverse/semiempirical/mndo_parameters.h"
#include "atomiverse/semiempirical/pm3_parameters.h"
#include "atomiverse/semiempirical/rm1_parameters.h"
#include <stdexcept>
#include <string>

namespace ATV::semiempirical {
const AM1Parameter* nddo_parameter(SemiEmpiricalMethod method, int z) noexcept {
    switch (method) {
    case SemiEmpiricalMethod::AM1:
        return am1_parameter(z);
    case SemiEmpiricalMethod::RM1:
        return rm1_parameter(z);
    case SemiEmpiricalMethod::MNDO:
        return mndo_parameter(z);
    case SemiEmpiricalMethod::PM3:
        return pm3_parameter(z);
    }
    return nullptr;
}
bool nddo_has_parameter(SemiEmpiricalMethod method, int z) noexcept {
    return nddo_parameter(method, z) != nullptr;
}
double nddo_atomic_heat_kcal_mol(SemiEmpiricalMethod method, int z) {
    if (!nddo_parameter(method, z))
        throw std::invalid_argument(std::string(nddo_method_name(method))
                                    + " has no parameters for atomic number "
                                    + std::to_string(z));
    // Experimental atomic heats are independent of the fitted Hamiltonian.
    return am1_atomic_heat_kcal_mol(z);
}
const char* nddo_method_name(SemiEmpiricalMethod method) noexcept {
    switch (method) {
    case SemiEmpiricalMethod::AM1:
        return "AM1";
    case SemiEmpiricalMethod::RM1:
        return "RM1";
    case SemiEmpiricalMethod::MNDO:
        return "MNDO";
    case SemiEmpiricalMethod::PM3:
        return "PM3";
    }
    return "unknown";
}
} // namespace ATV::semiempirical
