/// @file cpu/communication/bcast_frame.h
/// @brief `bcast_frame` -- replicate the root rank's whole Frame onto every
///        rank of `MPI_COMM_WORLD`.
///
/// The rank-0-reads / every-rank-computes primitive. Before payload traffic,
/// every rank collectively validates the atoms-Block column schema
/// (`comm::column_schema_fingerprint` over `Block::resolve_columns`) and the
/// typed metadata schema. The root then broadcasts an explicit fixed-width
/// header, nine cell doubles, typed metadata bytes, and **one Bcast per
/// atoms column** — the native bytes of `[0, n)` rows, column-major, no
/// per-row packing and no borrowed wire plan. Every non-root rank
/// reconstructs a full replica: all rows owned (`nlocal == n`,
/// `nghost == 0`), tag→local map rebuilt, cell and serializable metadata
/// overwritten. The root's Frame is never modified.
///
/// A variable-length topology column's reserve stride is per rank, so the
/// root's stride travels in the header: a receiver adopts it where it is
/// wider (`Frame::grow_group`) and stages + zero-pads where its own is
/// wider.
///
/// The payload is pure byte transit -- no arithmetic touches any value in
/// flight, so every replicated field (positions [Angstrom], velocities
/// [Angstrom/fs], forces [eV/Angstrom], charges [e], scalars [eV], ...) is
/// bit-identical to the root's.
///
/// **Collective contract (lockstep):** under multi-rank every rank must call
/// this function in lockstep *before* any rank gate -- it enters MPI
/// collectives internally (same discipline as `gather_for_root_io`).
///
/// **Precondition:** runtime column and serializable metadata registration
/// must be identical on every rank. A divergent dtype, width, flag, key, or
/// order fails collectively before any cell/meta/atom payload is sent.
///
/// Degrade path: on a non-MPI build, or when MPI is not initialized, or at
/// world size 1, the call returns immediately and the Frame is untouched
/// (`MPI_Initialized` probe, same as `scatter_atoms_from_root`).
///
/// @see cpu/comm/atom_pack.h    `column_schema_fingerprint`, the rank
///      agreement value this primitive checks before any payload
/// @see cpu/communication/comm.h  `Comm::exchange`, the
///      distribute-by-ownership sibling of this replicate-everywhere call
#pragma once

namespace ATV {
namespace cpu {

class Frame;

namespace comm {

/// @brief Broadcast the root rank's Frame to every rank of MPI_COMM_WORLD.
///
/// @param frame On the root rank: the source Frame (owned rows `[0, nlocal)`;
///              read-only, never modified). On every other rank: the
///              destination Frame, overwritten with a full replica (atom
///              rows, bonded topology, runtime exchange columns, cell, and
///              serializable typed metadata).
/// @param root  Rank holding the source Frame (default 0).
/// @throws std::runtime_error collectively on schema disagreement.
/// @throws std::length_error if a native byte payload exceeds MPI's
///         `INT_MAX` count limit (no silent chunking).
void bcast_frame(Frame& frame, int root = 0);

} // namespace comm
} // namespace cpu
} // namespace ATV
