/// @file cpu/frame/molrs_exchange.h
/// @brief cpu::from_molrs / cpu::to_molrs -- molrs.Frame <-> cpu::Frame.
///
/// `molrs.Frame` is the canonical exchange layer; `cpu::Frame` is the CPU
/// backend's execution frame. The materializer belongs to the CPU `frame/`
/// pillar and runs once per `run()` -- never inside a step loop.
///
/// `from_molrs()` accepts only the schema-v2
/// atoms/bonds/angles/dihedrals/impropers blocks and their declared columns.
/// Unknown blocks/columns, wrong dtypes, and inconsistent row counts fail
/// immediately. Required fields (positions, `atomic_number`) fail with a
/// named exception; optional state defaults to zero/empty.
/// `to_molrs()` pushes mutable atom state, the simulation box, and every
/// serializable exact-dtype metadata entry back into the same molrs.Frame.
/// The box is in the write-back set because a barostat or a cell relaxation
/// makes the cell an *output* of the run, not just an input.
///
/// @see cpu::Frame for the CPU execution frame
/// @see molrs.hpp for the molrs CXX FFI bridge
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {

/// @brief Materialize a molrs.Frame into a new cpu::Frame.
///
/// Reads the `atoms` block (`x, y, z, vx, vy, vz, fx, fy, fz, mass,
/// atomic_number, tag, charge`, plus per-axis image columns `ix, iy, iz`),
/// the `bonds`, `angles`, `dihedrals` and `impropers` topology blocks, and
/// the simulation box.
///
/// Required fields are the atom positions (`x`, `y`, `z`) and
/// `atomic_number` — the atomic number Z, which molrs 0.13 split out of the
/// old `atoms.type` column and which lands in `cpu::Frame`'s `kAtom_type`.
/// Their absence throws `std::runtime_error` naming the missing field.
/// Every other field is optional and defaults to zero / empty when absent.
/// The String columns `type` (force-field label) and `element` (symbol) are
/// accepted so that a frame written by a molrs tool still validates, but
/// they are not imported: Atomiverse keys species on Z.
///
/// @param src molrs.Frame to read (introspected by block/column name).
/// @return Newly constructed cpu::Frame with all available fields filled.
/// @throws std::runtime_error if a required field is missing.
Frame from_molrs(const molrs::FrameRef& src);

/// @brief Export a cpu::Frame into an existing molrs.Frame.
///
/// Round-trip field contract (arch-refactor-06-frame; the golden test in
/// tests/cpu/test_molrs_exchange.cpp asserts both directions):
///
/// | field | from_molrs (molrs→cpu) | to_molrs (cpu→molrs) |
/// |---|---|---|
/// | x/y/z [Å]              | required (throws)       | written |
/// | vx/vy/vz, fx/fy/fz     | optional (zero default) | written |
/// | charge [e]             | optional (zero default) | written |
/// | image (ix/iy/iz)       | optional (i32→int8)     | written (int8→i32) |
/// | tag                    | optional (i32→TagInt)   | written (checked i32 narrowing; overflow throws, no silent truncation) |
/// | atomic_number (Z)      | required (u32 → `kAtom_type`) | NOT written (per-run immutable input) |
/// | type, element          | allow-listed, not imported | NOT written |
/// | mass [amu]             | optional                | NOT written (per-run immutable input) |
/// | bonds/angles/dihedrals/impropers | optional → per-atom | NOT written (per-run static) |
/// | box                    | optional                | written (row-major H; a zero-volume cell is skipped) |
/// | typed metadata         | read (all 19 molrs dtypes) | written (all serializable entries) |
///
/// image evolves during MD (unwrapped-coordinate reconstruction) and tag
/// is the identity anchor after multi-rank migration — both round-trip.
/// The box is an output too: a barostat or cell relaxation rewrites the
/// cell, so it crosses back in molrs' row-major order (`h[i*3+j]` ↔
/// `cell(i, j)`), symmetric with the read side. The one exception is a
/// cell with no volume (`box.volume() <= 0`, i.e. `|det H| == 0` — what a
/// default-constructed `Box` has): it is skipped, because molrs rejects a
/// singular cell matrix and a frame that never carried a box must not
/// acquire one from the export.
/// Metadata is part of the Frame boundary itself.  Energy, kinetic energy,
/// conserved energy, step, and the two six-component stress tensors retain
/// their exact typed values in `Frame::meta`; there is no string side channel
/// and no synthetic one-row scalar/stress block.
///
/// @param src cpu::Frame whose state is exported.
/// @param dst molrs.Frame to update in place (must be the same frame that
///            was materialized, or any frame with a compatible atom count).
/// @throws std::runtime_error when an atom tag does not fit i32.
void to_molrs(const Frame& src, molrs::FrameRef& dst);

} // namespace cpu
} // namespace ATV
