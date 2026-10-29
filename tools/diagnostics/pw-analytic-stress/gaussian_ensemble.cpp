#include "ensemble_diagnostic.h"
#include "gaussian_ensemble_pins.h"
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--control") {
        std::vector<char*> args;
        args.push_back(argv[0]);
        for (int i = 2; i < argc; ++i)
            args.push_back(argv[i]);
        const int forwarded_argc = int(args.size());
        args.push_back(nullptr);
        return pw_ensemble_diagnostic::entry(
            forwarded_argc, args.data(), 1,
            [](std::size_t) {
                const auto& c = pw_gaussian_qe::near_integer_control;
                pw_ensemble_diagnostic::run(c.pin, c.recipe);
            },
            "PW_GAUSSIAN");
    }
    return pw_ensemble_diagnostic::entry(
        argc, argv, pw_gaussian_qe::cases.size(),
        [](std::size_t i) {
            const auto& c = pw_gaussian_qe::cases.at(i);
            pw_ensemble_diagnostic::run(c.pin, c.recipe);
        },
        "PW_GAUSSIAN");
}
