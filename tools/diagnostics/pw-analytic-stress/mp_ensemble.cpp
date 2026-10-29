#include "ensemble_diagnostic.h"
#include "mp_ensemble_pins.h"
int main(int argc, char** argv) {
    pw_analytic_diagnostic::Recipe r;
    r.smearing = ATV::electronic::Smearing::methfessel_paxton;
    r.width_ry = .04;
    r.kmesh = 1;
    r.isolated_h3 = false;
    return pw_ensemble_diagnostic::entry(
        argc, argv, 1,
        [&](std::size_t) { pw_ensemble_diagnostic::run(pw_mp_qe::pin, r); },
        "PW_MP");
}
