/// @file cpu/task/internal/minimize_builders.cpp
/// @brief Typed Minimize params -> the components and criterion they mirror.

#include "atomiverse/cpu/task/internal/minimize_builders.h"

#include <type_traits>
#include <variant>

#include "atomiverse/cpu/driver/criteria.h"

namespace ATV::cpu::task::internal {

cpu::minimizer::FIRE tuned_fire(const FIREParams& p) {
    cpu::minimizer::FIRE fire;
    fire.dt0 = p.dt0;
    fire.dt_max = p.dt_max;
    fire.n_min = p.n_min;
    fire.f_inc = p.f_inc;
    fire.f_dec = p.f_dec;
    fire.alpha0 = p.alpha0;
    fire.f_alpha = p.f_alpha;
    fire.max_step = p.max_step;
    return fire;
}

cpu::minimizer::IsotropicCellRelax
tuned_isotropic_cell_relax(const IsotropicCellRelaxParams& p) {
    cpu::minimizer::IsotropicCellRelax cell(p.P_target, p.alpha0);
    cell.max_scale = p.max_scale;
    cell.bulk_mod = p.bulk_mod;
    return cell;
}

cpu::minimizer::AnisotropicCellRelax
tuned_anisotropic_cell_relax(const AnisotropicCellRelaxParams& p) {
    cpu::minimizer::AnisotropicCellRelax cell(p.P_target, p.alpha0);
    cell.max_scale = p.max_scale;
    cell.bulk_mod = p.bulk_mod;
    cell.relax_x = p.relax_x;
    cell.relax_y = p.relax_y;
    cell.relax_z = p.relax_z;
    return cell;
}

cpu::minimizer::TriclinicCellRelax
tuned_triclinic_cell_relax(const TriclinicCellRelaxParams& p) {
    cpu::minimizer::TriclinicCellRelax cell(p.alpha0, p.bulk_mod);
    cell.max_strain = p.max_strain;
    return cell;
}

cpu::minimizer::JointIsotropicRelax
tuned_joint_isotropic_relax(const JointIsotropicRelaxParams& p) {
    cpu::minimizer::JointIsotropicRelax cell(p.P_target, p.atom_step);
    cell.alpha0 = p.alpha0;
    cell.max_scale = p.max_scale;
    cell.bulk_mod = p.bulk_mod;
    return cell;
}

void add_minimize_criterion(minimizer::Minimize& driver,
                            const MinimizeCriterionParams& criterion) {
    std::visit(
        [&](const auto& c) {
            using C = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<C, ForceConvergedParams>)
                driver.add_criterion(cpu::ForceConverged{
                    c.ftol, c.rms_ftol, c.max_dtol, c.rms_dtol, c.etol});
            else if constexpr (std::is_same_v<C, PressureConvergedParams>)
                driver.add_criterion(
                    cpu::PressureConverged{c.P_target, c.ptol});
            else
                driver.add_criterion(cpu::StressConverged{
                    c.sigma_target, c.stol, c.check_x, c.check_y, c.check_z});
        },
        criterion);
}

} // namespace ATV::cpu::task::internal
