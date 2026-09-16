/// @file src/cpu/io/readable_path.h
/// @brief Path pre-flight and molrs error translation for the FFI boundary.
///
/// **molrs 0.13** made the bridge readers and writers fallible —
/// `fn xyz_read_first_frame(path: &str) -> Result<Box<FrameRef>>` and the
/// same for `read_first_frame`, `write_frame_xyz`, `write_frame` (named
/// `read_frame_zarr_first` / `write_frame_zarr` before molrs 0.14),
/// `frame_set_box` and the four `frame_set_column_*`. cxx maps `Result<T>` to
/// a C++ function returning `T` that throws `rust::Error`, which derives from
/// `std::exception`, so a caller's `catch` finally sees these failures. Before
/// 0.13 they were `panic!` inside a non-`Result` extern fn, which cxx turns
/// into an uncatchable `abort()`.
///
/// Two things are still wanted on this side, and both live here:
///   * `require_readable_file` / `require_readable_store` — a pre-flight check
///     so the overwhelmingly common "wrong filename" case reports the path in
///     our own wording instead of whatever the Rust I/O error happens to say;
///   * `with_molrs_context` — attach the operation and path to a `rust::Error`
///     that does cross, because the bridge messages do not name the file.
///
/// Internal to `src/cpu/io/` and its neighbours (same status as
/// `src/cpu/io/molrs_meta.h`) — not a public header, not part of the API.
#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

// `rust::Error` is not emitted into the generated bridge header (Result<T>
// maps to T + throw, so Error never appears in a signature). This is a
// private header, so including the cxx runtime here is allowed — the ban
// in CLAUDE.md is on `rust/cxx.h` in PUBLIC headers.
#include "rust/cxx.h"

namespace ATV {
namespace cpu {
namespace internal {

/// @brief Compose the message for an unreadable path.
/// @param who  Public function name used as the message prefix.
/// @param path Filesystem path that could not be read.
/// @param why  Reason clause appended after the path.
/// @return Formatted `"<who>: cannot read '<path>': <why>"`.
inline std::string unreadable_message(const char* who, const std::string& path,
                                      const char* why) {
    return std::string(who) + ": cannot read '" + path + "': " + why;
}

/// @brief Throw unless `path` is an existing regular file this process can read.
///
/// @param who  Public function name used as the message prefix.
/// @param path Filesystem path about to be handed to a molrs reader.
/// @throws std::runtime_error if the path does not exist, names a directory, or
///         cannot be opened for reading.
inline void require_readable_file(const char* who, const std::string& path) {
    const std::filesystem::path fs_path{path};
    std::error_code ec;
    const auto status = std::filesystem::status(fs_path, ec);
    if (ec || !std::filesystem::exists(status))
        throw std::runtime_error(
            unreadable_message(who, path, "no such file or directory"));
    if (std::filesystem::is_directory(status))
        throw std::runtime_error(
            unreadable_message(who, path, "is a directory, not a file"));
    std::ifstream probe(path, std::ios::binary);
    if (!probe)
        throw std::runtime_error(unreadable_message(
            who, path, "not a readable file (permission denied?)"));
}

/// @brief Run a molrs bridge call, naming the operation and path on failure.
///
/// molrs 0.13 reports I/O and schema errors as `rust::Error`, whose message
/// does not include the file it was working on. Wrapping restores that
/// context without swallowing the underlying reason.
///
/// @tparam F  Callable invoking exactly one molrs bridge function.
/// @param who Public function name used as the message prefix.
/// @param path Filesystem path the call operates on.
/// @param f   The bridge call.
/// @return Whatever @p f returns.
/// @throws std::runtime_error wrapping any `rust::Error` @p f raised.
template <typename F>
auto with_molrs_context(const char* who, const std::string& path, F&& f)
    -> decltype(f()) {
    try {
        return f();
    } catch (const ::rust::Error& e) {
        throw std::runtime_error(std::string(who) + "('" + path
                                 + "'): " + e.what());
    }
}

/// @brief Throw unless `path` is an existing readable store (directory or file).
///
/// A Zarr store is a directory tree, but may also be a single archive file, so
/// both shapes are accepted; only unreadability is rejected.
///
/// @param who  Public function name used as the message prefix.
/// @param path Filesystem path about to be handed to a molrs reader.
/// @throws std::runtime_error if the path does not exist or cannot be read.
inline void require_readable_store(const char* who, const std::string& path) {
    const std::filesystem::path fs_path{path};
    std::error_code ec;
    const auto status = std::filesystem::status(fs_path, ec);
    if (ec || !std::filesystem::exists(status))
        throw std::runtime_error(
            unreadable_message(who, path, "no such file or directory"));
    if (std::filesystem::is_directory(status)) {
        std::filesystem::directory_iterator entries(fs_path, ec);
        if (ec)
            throw std::runtime_error(unreadable_message(
                who, path, "directory is not listable (permission denied?)"));
        return;
    }
    std::ifstream probe(path, std::ios::binary);
    if (!probe)
        throw std::runtime_error(unreadable_message(
            who, path, "not a readable file (permission denied?)"));
}

} // namespace internal
} // namespace cpu
} // namespace ATV
