#include "ensemble_diagnostic.h"
#include "fixed_k_pins.h"
int main(int argc, char** argv) {
    return pw_ensemble_diagnostic::entry(
        argc, argv, pw_fixed_k_qe::pins.size(),
        [](std::size_t i) {
            pw_analytic_diagnostic::Recipe recipe;
            recipe.smearing = ATV::electronic::Smearing::none;
            recipe.width_ry = 0;
            pw_ensemble_diagnostic::run(pw_fixed_k_qe::pins.at(i), recipe);
        },
        "PW_FIXED_K");
}
