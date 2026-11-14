/// @file core/velocity_create.h
/// @brief The random draw of LAMMPS `velocity create`, shared by the CPU and
///        CUDA `InitVelocity` so one seed gives one set of velocities on
///        either device.
#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

namespace ATV::md {

/// @brief Velocity-draw distribution for `InitVelocity` (LAMMPS `dist`).
///
/// Mirrors LAMMPS `velocity create ... dist` (`velocity.cpp:38`,
/// `enum { UNIFORM, GAUSSIAN }`):
///  - `uniform`  — each component drawn from `U(-0.5, 0.5)`
///                 (`random->uniform() - 0.5`, velocity.cpp:276), then
///                 scaled by `1/sqrt(m)`.
///  - `gaussian` — each component drawn from `N(0, 1)`
///                 (`random->gaussian()`, velocity.cpp:280), then scaled
///                 by `1/sqrt(m)`.
///
/// After either draw, the velocities are COM-corrected and rescaled so the
/// measured temperature exactly equals the requested `T`, so the choice of
/// `dist` only affects the *shape* of the per-atom distribution, never the
/// aggregate temperature.
enum class VelocityDist {
    uniform,  ///< U(-0.5, 0.5) per component (LAMMPS `dist uniform`).
    gaussian, ///< N(0, 1) per component (LAMMPS `dist gaussian`).
};

/// @brief Raw `velocity create` draws for `n` atoms, keyed by tag.
///
/// `rng` feeds `dist` three numbers per atom (x, y, z), atoms taken in
/// ascending tag order, so the velocity an atom gets depends on its tag
/// and not on the row it sits in (a device may order rows differently).
/// The draws are unscaled: the caller multiplies by `1/sqrt(m)`, removes
/// the COM velocity and rescales to the target temperature.
///
/// @param tag Tags of rows `[0, n)`, unique.
/// @return `3 n` values, row `i`'s x, y, z at `3 i`, `3 i + 1`, `3 i + 2`.
template <typename Tag>
std::vector<double> velocity_create_draws(std::mt19937& rng, VelocityDist dist,
                                          const Tag* tag, int n) {
    std::vector<int> rows(static_cast<std::size_t>(n));
    std::iota(rows.begin(), rows.end(), 0);
    std::sort(rows.begin(), rows.end(),
              [tag](int a, int b) { return tag[a] < tag[b]; });
    std::vector<double> draws(3 * rows.size());
    auto fill = [&](auto&& draw) {
        for (const int row : rows)
            for (int axis = 0; axis < 3; ++axis)
                draws[3 * static_cast<std::size_t>(row) + axis] = draw(rng);
    };
    if (dist == VelocityDist::gaussian)
        fill(std::normal_distribution<double>(0.0, 1.0));
    else
        fill(std::uniform_real_distribution<double>(-0.5, 0.5));
    return draws;
}

} // namespace ATV::md
