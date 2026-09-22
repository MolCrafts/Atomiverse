/// @file regressions/architecture/arch-cuda-gpw-public-shape-freeze/
///       reg_arch-cuda-gpw-public-shape-freeze.cpp
/// @brief Architecture gate: CUDA GPW product public shape = CPU GpwDft.
///
/// Freezes the **target** public contract for `cuda-gpw-full-product` by
/// asserting the CPU oracle surface that the CUDA method must mirror:
///   - concrete `GpwDft : cpu::Driver<gpw::Phase>` (multi-path DIIS stages)
///   - members + `build` / `run` / `last_result` / `is_uks` (no *Config)
/// Stage footholds under `cuda/gpw/internal/` are not the product API.
/// OT free-loop residual on CPU does not change this contract.
///
/// Public headers only. Shape-only — no multi-iter SCF (iron law: e2e stays
/// in regressions that pin energy, not this harness).

#include <cstdio>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"

namespace {

// Compile-time freeze: method is a Driver node, not a freestanding solver bag.
static_assert(
    std::is_base_of_v<ATV::cpu::Driver<ATV::gpw::Phase>, ATV::cpu::gpw::GpwDft>,
    "CUDA product must mirror GpwDft as Driver<gpw::Phase>, not a "
    "free-loop SCFSolver bag");

// Detect the frozen entry points (members + build/run/last_result).
template <typename T>
using build_t = decltype(std::declval<T&>().build(
    std::declval<const std::vector<ATV::cpu::gpw::GpwAtom>&>()));

template <typename T>
using run_t = decltype(std::declval<T&>().run(
    std::declval<const std::vector<ATV::cpu::gpw::GpwAtom>&>()));

template <typename T>
using last_result_t = decltype(std::declval<const T&>().last_result());

template <typename T>
using is_uks_t = decltype(std::declval<const T&>().is_uks());

template <typename, typename = void>
struct has_build : std::false_type {};
template <typename T>
struct has_build<T, std::void_t<build_t<T>>> : std::true_type {};

template <typename, typename = void>
struct has_run : std::false_type {};
template <typename T>
struct has_run<T, std::void_t<run_t<T>>> : std::true_type {};

template <typename, typename = void>
struct has_last_result : std::false_type {};
template <typename T>
struct has_last_result<T, std::void_t<last_result_t<T>>> : std::true_type {};

template <typename, typename = void>
struct has_is_uks : std::false_type {};
template <typename T>
struct has_is_uks<T, std::void_t<is_uks_t<T>>> : std::true_type {};

static_assert(has_build<ATV::cpu::gpw::GpwDft>::value,
              "frozen public shape requires build(atoms)");
static_assert(has_run<ATV::cpu::gpw::GpwDft>::value,
              "frozen public shape requires run(atoms, …)");
static_assert(has_last_result<ATV::cpu::gpw::GpwDft>::value,
              "frozen public shape requires last_result()");
static_assert(has_is_uks<ATV::cpu::gpw::GpwDft>::value,
              "frozen public shape requires is_uks()");

// last_result returns a const GpwResult reference (owned buffer on the method).
static_assert(std::is_same_v<last_result_t<ATV::cpu::gpw::GpwDft>,
                             const ATV::cpu::gpw::GpwResult&>,
              "last_result must expose const GpwResult& (Frame/evaluator write "
              "is separate)");

} // namespace

int main() {
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GpwResult;

    if (ATV::slot_count<ATV::gpw::Phase>() != 5) {
        std::fprintf(stderr,
                     "FAIL: gpw::Phase slot_count != 5 (CUDA product must keep "
                     "the same stage spine for DIIS)\n");
        return 1;
    }
    if (ATV::gpw::kind_of(ATV::gpw::Phase{}) != ATV::PhaseKind::SCF) {
        std::fprintf(stderr, "FAIL: gpw::Phase kind is not SCF\n");
        return 1;
    }

    // Default-constructed method: members (not Config), UKS selection,
    // last_result pre-run is not converged — no SCF launched.
    GpwDft method;
    method.n_occ = 4;
    method.cutoff_ry = 300.0;
    method.tol = 1e-8;
    method.max_iter = 100;
    method.use_ot =
        false; // DIIS path is the product spine; OT residual ≠ block

    if (method.is_uks()) {
        std::fprintf(stderr, "FAIL: default n_alpha/n_beta should be RKS\n");
        return 1;
    }
    method.n_alpha = 7;
    method.n_beta = 5;
    if (!method.is_uks()) {
        std::fprintf(stderr, "FAIL: n_alpha/n_beta should select UKS\n");
        return 1;
    }

    const GpwResult& pre = method.last_result();
    if (pre.converged) {
        std::fprintf(
            stderr, "FAIL: last_result before run must not report converged\n");
        return 1;
    }

    // Documented freeze: stage footholds are not this type; product composes
    // them under a GpwDft-shaped public surface (see note).
    std::printf(
        "PASS arch-cuda-gpw-public-shape-freeze: GpwDft public shape frozen "
        "for CUDA product (Driver + build/run/last_result; no SCF)\n");
    return 0;
}
