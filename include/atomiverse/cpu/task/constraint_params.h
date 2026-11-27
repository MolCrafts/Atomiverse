#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ATV::cpu::task {
/// Solver a distance-constraint parameter set selects.
enum class DistanceConstraintAlgorithm { Shake, Rattle, Lincs };

/// Backend-neutral, serializable distance-constraint configuration.
template <DistanceConstraintAlgorithm Algorithm>
struct DistanceConstraintParams {
    double dt, tolerance;
    int max_iter;
    std::vector<std::pair<int, double>> bonds, angles;
    explicit DistanceConstraintParams(double step = 1.0, double tol = 1e-10,
                                      int iterations = 500)
        : dt(step), tolerance(tol), max_iter(iterations) {
        if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(tolerance)
            || tolerance <= 0 || max_iter < 1)
            throw std::invalid_argument("distance constraints: positive finite "
                                        "dt/tolerance and iterations required");
    }
    void constrain_bond(int type, double length) {
        append(bonds, type, length);
    }
    void constrain_angle(int type, double distance) {
        append(angles, type, distance);
    }
    /// @brief The native component: `Native(dt, tolerance, max_iter,
    ///        extra...)` with every rule applied.
    template <class Native, class... Extra>
    Native component(Extra... extra) const {
        Native result(dt, tolerance, max_iter, extra...);
        for (auto [type, length] : bonds)
            result.constrain_bond(type, length);
        for (auto [type, distance] : angles)
            result.constrain_angle(type, distance);
        return result;
    }
    bool operator==(const DistanceConstraintParams& other) const {
        return dt == other.dt && tolerance == other.tolerance
               && max_iter == other.max_iter && bonds == other.bonds
               && angles == other.angles;
    }

  private:
    static void append(std::vector<std::pair<int, double>>& rules, int type,
                       double length) {
        if (type < 0 || !std::isfinite(length) || length <= 0)
            throw std::invalid_argument(
                "distance constraints: nonnegative type and positive finite "
                "distance required");
        for (auto rule : rules)
            if (rule.first == type)
                throw std::invalid_argument(
                    "distance constraints: duplicate type rule");
        rules.emplace_back(type, length);
    }
};
using FixShakeParams =
    DistanceConstraintParams<DistanceConstraintAlgorithm::Shake>;
using FixRattleParams =
    DistanceConstraintParams<DistanceConstraintAlgorithm::Rattle>;

/// LINCS: the shared rule set plus the matrix-expansion order.
struct FixLincsParams
    : DistanceConstraintParams<DistanceConstraintAlgorithm::Lincs> {
    int order; ///< expansion order [dimensionless], >= 1
    explicit FixLincsParams(double step = 1.0, double tol = 1e-10,
                            int iterations = 500, int expansion_order = 4)
        : DistanceConstraintParams(step, tol, iterations),
          order(expansion_order) {
        if (order < 1)
            throw std::invalid_argument(
                "LINCS: the expansion order must be >= 1");
    }
    template <class Native>
    Native component() const {
        return DistanceConstraintParams::component<Native>(order);
    }
    bool operator==(const FixLincsParams& other) const {
        return DistanceConstraintParams::operator==(other)
               && order == other.order;
    }
};
} // namespace ATV::cpu::task
