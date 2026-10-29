#pragma once
#include "atomiverse/cpu/planewave/pw_solver.h"
#include <complex>
#include <stdexcept>
#include <vector>
namespace trace {
struct Counts {
    int davidson = 0, extract = 0, assembly = 0, legacy_stress = 0;
};
extern Counts counts;
struct SharedBands {
    const void* state = nullptr;
    const void* ionic = nullptr;
    int force_calls = 0;
    bool same_inputs = true;
    double fcounts[2]{}, rho_counts[2]{},
        aux_ne = 0, aux_m = 0, aux_entropy = 0, completion_residual = 0;
    std::vector<std::complex<ATV::AccReal>> payload;
    int fixed_na = -1, fixed_nb = -1, empty_beta_blocks = 0;
    bool beta_layout_zero = false, beta_density_exact_zero = false;
    // Diagnostic context only. Reset on every extract and trace::reset().
    bool mp_active = false;
    int mp_order = 0;
    double mp_width_ha = 0, mp_target_ha = 0, mp_reference_energy_ha = 0;
    double mp_min_f = 0, mp_max_f = 0, mp_mu = 0;
};
extern SharedBands shared_bands;
inline void reset() {
    counts = Counts{};
    shared_bands = SharedBands{};
}
class Counter final : public ATV::cpu::Component {
    const ATV::cpu::pw::PW& m_;

  public:
    int iterations = 0, builds = 0, destroys = 0;
    explicit Counter(const ATV::cpu::pw::PW& m) : m_(m) {}
    std::string name() const override { return "pw_general_actual_scf"; }
    void build(const ATV::cpu::BuildContext&) override {
        iterations = 0;
        ++builds;
    }
    void launch(const ATV::cpu::LaunchContext& c) override {
        if (m_.last_result().iterations != c.step + 1 || iterations != c.step)
            throw std::runtime_error("one actual SCF iteration measurement");
        ++iterations;
    }
    void destroy(const ATV::cpu::DestroyContext&) override { ++destroys; }
};
inline void closed_budget(const Counter& stage, int spin) {
    if (stage.builds != 1 || stage.destroys != 1 || counts.extract != 1
        || counts.assembly != 1 || counts.legacy_stress != 0
        || counts.davidson != spin * (stage.iterations + 1))
        throw std::runtime_error(
            "closed public one-SCF one-allspin-sweep budget");
}
} // namespace trace
