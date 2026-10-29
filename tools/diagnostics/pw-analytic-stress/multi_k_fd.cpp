#include "ensemble_diagnostic.h"
int main(int argc, char** argv) {
    return pw_ensemble_diagnostic::entry(
        argc, argv, pw_multi_k_fd_qe::pins.size(),
        [](std::size_t i) {
            pw_ensemble_diagnostic::run(pw_multi_k_fd_qe::pins.at(i));
        },
        "PW_MULTI_K_FD");
}
