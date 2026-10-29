#include "empty_beta_fixture.h"
#include "ensemble_diagnostic.h"
int main(int argc, char** argv) {
    auto recipe = empty_beta_fixture::recipe();
    std::vector<char*> args{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--rotate-z90")
            recipe.rotate_z90 = true;
        else
            args.push_back(argv[i]);
    }
    int n = args.size();
    args.push_back(nullptr);
    return pw_ensemble_diagnostic::entry(
        n, args.data(), 1,
        [&](std::size_t) {
            auto p = empty_beta_qe::pin;
            if (recipe.rotate_z90) {
                for (int i = 0; i < p.nat; ++i) {
                    auto f = p.forces_ha_bohr[i];
                    p.forces_ha_bohr[i] = {-f[1], f[0], f[2]};
                }
                auto s = p.stress_ha_bohr3;
                p.stress_ha_bohr3 = {s[1], s[0], s[2], -s[3], -s[5], s[4]};
            }
            std::printf("EMPTY_ROTATION z90=%d\n", recipe.rotate_z90);
            pw_ensemble_diagnostic::run(p, recipe);
        },
        "PW_EMPTY_BETA");
}
