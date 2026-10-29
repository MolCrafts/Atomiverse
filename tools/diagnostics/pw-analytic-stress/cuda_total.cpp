#include "cuda_total_common.h"
#include "qe-2026-10-28/pins.h"
int main(int argc, char** argv) {
    const pw_cuda_total::Reference ref{"LSDA",
                                       ATV::cpu::xc::XCType::LSDA,
                                       pw_analytic_qe::energy_ha,
                                       pw_analytic_qe::forces_ha_bohr,
                                       pw_analytic_qe::stress_ev_ang3,
                                       {16, 18, 16},
                                       {24, 28, 24}};
    return run_cuda_total(argc, argv, ref);
}
