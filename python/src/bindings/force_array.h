/// @file python/src/bindings/force_array.h
/// @brief The one numpy currency for nuclear forces returned to Python.
///
/// Every binding that publishes forces (`RHF/UHF/RKS/UKS.forces`,
/// `AM1Evaluator.forces`) hands back the same thing: a freshly allocated,
/// numpy-owned `(n_atoms, 3)` double array [eV/Å]. The engine's own force
/// storage is not exposed — it is either reused by the next solve
/// (`AM1Result::forces_ev_a`) or a temporary (`GradientResult::forces`), so
/// the array Python keeps must own its buffer.
///
/// `to_force_array()` is that copy: it allocates, copies, and attaches an
/// `nb::capsule` deleter so the buffer's lifetime follows the numpy array and
/// nothing in the engine has to outlive the call.
///
/// This is deliberately *not* in `molrs_frame_bridge.h`, whose charter is the
/// molrs.Frame capsule boundary; a numpy return currency is a different
/// concern with a different dependency (`nanobind/ndarray.h`).
#pragma once

#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

/// @brief Nuclear forces as returned to Python: a fresh, numpy-owned
///        (n_atoms, 3) C-contiguous double array [eV/Å].
using ForceArray = nb::ndarray<nb::numpy, double, nb::shape<-1, 3>>;

/// @brief Copy a contiguous force buffer into a fresh numpy-owned array.
///
/// The source is read as row-major (fx0, fy0, fz0, fx1, …) — the layout of
/// both an `xt::xtensor<double, 2>` of shape [n_atoms, 3] and the flat
/// `std::vector<double>` an NDDO result publishes — and is copied element for
/// element, so the returned array shares no storage with the engine and is
/// unaffected by a subsequent solve.
///
/// @param forces_ev_a Nuclear forces F = −∂E/∂R [eV/Å], contiguous row-major,
///                    length exactly 3·n_atoms (three Cartesian components per
///                    atom). An empty span yields an empty (0, 3) array.
/// @return An (n_atoms, 3) numpy array [eV/Å] owning its buffer; the buffer is
///         released by the array's capsule deleter, not by the caller.
/// @throws std::invalid_argument (Python `ValueError`) when the length is not a
///         multiple of 3 — that is a mis-sized force buffer, never a shape to
///         round down to.
inline ForceArray to_force_array(std::span<const double> forces_ev_a) {
    if (forces_ev_a.size() % 3u != 0u)
        throw std::invalid_argument(
            "force buffer length " + std::to_string(forces_ev_a.size())
            + " is not a multiple of 3; expected 3 Cartesian components "
              "[eV/Å] per atom");
    const std::size_t n_atoms = forces_ev_a.size() / 3u;
    auto buffer = std::make_unique<double[]>(forces_ev_a.size());
    std::copy(forces_ev_a.begin(), forces_ev_a.end(), buffer.get());
    nb::capsule owner(buffer.get(), [](void* p) noexcept {
        delete[] static_cast<double*>(p);
    });
    return ForceArray(buffer.release(), {n_atoms, std::size_t{3}}, owner);
}
