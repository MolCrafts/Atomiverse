/// @file regressions/semiempirical/nddo_mopac_fixtures.h
/// @brief Executed OpenMOPAC 23.2.5 pins shared by CPU/CUDA MNDO and PM3 cases.
#pragma once
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/semiempirical/nddo_result.h"
#include <cmath>
#include <cstdio>
namespace atv_reg_nddo_mopac {
struct Pin {
    const char* name;
    int count;
    int species[4];
    double xyz[4][3];
    double energy_ev, hof_kcal;
    double charges[4];
    double forces[4][3];
};
inline constexpr Pin kMndoPins[] = {
    {"mndo-ammonia",
     4,
     {7, 1, 1, 1},
     {{0.1, -0.1, 0.05},
      {1.05, 0.05, 0.27},
      {-0.34, 0.78, 0.36},
      {-0.25, -0.81, 0.59}},
     -250.1432261006765,
     -4.15051786050844,
     {-0.24601301893293, 0.08032557734292, 0.07839923989017, 0.08728820169984},
     {{-0.3893254286536477, 2.8642262764690773, -1.558163720618963},
      {0.9487510330107841, -0.3140030038918734, 0.31499727425572216},
      {0.09078564514636256, -0.9953186669965581, -0.2401754853112102},
      {-0.6502112489768419, -1.5549046041649854, 1.4833419308510316}}},
    {"mndo-co",
     2,
     {6, 8},
     {{0.1, -0.2, 0.3}, {0.95, 0.4, 0.9}},
     -448.52399630622097,
     -3.7281980939988,
     {0.24436525391919, -0.24436525391919},
     {{3.4588021891153007, 2.441507460914359, 2.441507462051227},
      {-3.458802187195281, -2.441507460914359, -2.441507460695917}}},
    {"mndo-water",
     3,
     {8, 1, 1},
     {{0.0, 0.0, 0.0}, {0.96, 0.0, 0.0}, {-0.24, 0.93, 0.1}},
     -351.4007012981908,
     -60.3773032042127,
     {-0.31799186774882, 0.15878645017323, 0.1592054175756},
     {{0.7720679063538695, 1.3609478705234312, 0.1463384933231273},
      {-0.8863284806490582, -0.18363764541007102, -0.019745983141918128},
      {0.11426057476499982, -1.1773102244936742, -0.12659251018485174}}},
};
inline constexpr Pin kPm3Pins[] = {
    {"pm3-ammonia",
     4,
     {7, 1, 1, 1},
     {{0.1, -0.1, 0.05},
      {1.05, 0.05, 0.27},
      {-0.34, 0.78, 0.36},
      {-0.25, -0.81, 0.59}},
     -208.5656409954807,
     -1.23810136503744,
     {0.01672172443229, -0.00672073723683, -0.00902733453112,
      -0.00097365266434},
     {{-0.21030597992571257, 2.638867550861898, -0.565712484233598},
      {0.8151003909658433, -0.3001912474062969, 0.0006633171789826204},
      {0.040281209764450106, -0.9151242656170633, -0.4511719284257919},
      {-0.6450756203521067, -1.4235520365403167, 1.0162210949162924}}},
    {"pm3-co",
     2,
     {6, 8},
     {{0.1, -0.2, 0.3}, {0.95, 0.4, 0.9}},
     -411.1597382526485,
     -13.7071959042387,
     {0.22224971968821, -0.22224971968821},
     {{5.277013408582973, 3.724950669088933, 3.724950669088933},
      {-5.277013405653645, -3.724950669088933, -3.724950667021168}}},
    {"pm3-water",
     3,
     {8, 1, 1},
     {{0.0, 0.0, 0.0}, {0.96, 0.0, 0.0}, {-0.24, 0.93, 0.1}},
     -324.89413759945705,
     -53.1281142549024,
     {-0.35058582359021, 0.17531187496139, 0.17527394862881},
     {{0.45019980305483115, 0.9266817693287539, 0.09964321634470112},
      {-0.38127617413867737, -0.23875219378411033, -0.025672278960443416},
      {-0.06892362871745074, -0.6879295751825489, -0.07397093738638688}}},
};

inline ATV::cpu::Frame make_frame(const Pin& pin) {
    ATV::cpu::Frame frame(pin.count);
    for (int i = 0; i < pin.count; ++i) {
        frame.span(ATV::kAtom_atomic_number)[i] = pin.species[i];
        frame.span(ATV::kAtom_tag)[i] = i + 1;
        frame.span(ATV::kAtom_mass)[i] = pin.species[i] == 1   ? 1
                                         : pin.species[i] == 6 ? 12
                                         : pin.species[i] == 7 ? 14
                                                               : 16;
        frame.span(ATV::kAtom_x)[i] = pin.xyz[i][0];
        frame.span(ATV::kAtom_y)[i] = pin.xyz[i][1];
        frame.span(ATV::kAtom_z)[i] = pin.xyz[i][2];
    }
    return frame;
}
inline bool near(const char* label, const char* field, double got, double want,
                 double tolerance) {
    if (std::isfinite(got) && std::abs(got - want) <= tolerance)
        return true;
    std::fprintf(stderr, "FAIL %s %s got=%.17g want=%.17g tol=%g\n", label,
                 field, got, want, tolerance);
    return false;
}
inline bool check_result(const ATV::semiempirical::NDDOResult& result,
                         const Pin& pin, const char* label,
                         bool with_forces = true) {
    if (!result.converged
        || result.mulliken_charges.size() != static_cast<std::size_t>(pin.count)
        || (with_forces
            && result.forces_ev_a.size()
                   != static_cast<std::size_t>(3 * pin.count))) {
        std::fprintf(stderr, "FAIL %s %s result convergence/shape\n", label,
                     pin.name);
        return false;
    }
    bool ok =
        near(label, "energy [eV]", result.total_energy_ev, pin.energy_ev, 1e-6);
    ok &= near(label, "HOF [kcal/mol]", result.heat_of_formation_kcal_mol,
               pin.hof_kcal, 1e-6);
    for (int i = 0; i < pin.count; ++i) {
        char field[80];
        std::snprintf(field, sizeof(field), "%s charge[%d]", pin.name, i);
        ok &= near(label, field, result.mulliken_charges[i], pin.charges[i],
                   1e-7);
        if (with_forces)
            for (int d = 0; d < 3; ++d) {
                std::snprintf(field, sizeof(field), "%s force[%d,%d]", pin.name,
                              i, d);
                ok &= near(label, field, result.forces_ev_a[3 * i + d],
                           pin.forces[i][d], 1e-5);
            }
    }
    return ok;
}
} // namespace atv_reg_nddo_mopac
