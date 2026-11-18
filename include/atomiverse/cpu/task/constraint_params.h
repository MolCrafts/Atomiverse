#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ATV::cpu::task {
/// Backend-neutral, serializable distance-constraint configuration.
template <bool VelocityProjection>
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
    template <class Native>
    Native component() const {
        Native result(dt, tolerance, max_iter);
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
using FixShakeParams = DistanceConstraintParams<false>;
using FixRattleParams = DistanceConstraintParams<true>;
} // namespace ATV::cpu::task
