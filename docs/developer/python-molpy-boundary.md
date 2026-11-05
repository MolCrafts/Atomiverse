# Python Frame boundary (acyclic)

The MolCrafts/molpy repository owns the native Rust core, the `molcxx`
C++ bridge and the Python `molpy._core` extension. Atomiverse links `molcxx`
and accepts `molpy.Frame` objects through their shared FFI handle.

Both builds use the same local checkout selected by `ATV_MOLPY_PATH`
(default `../molpy`). No revision is pinned. The CXX bridge exports the
`molrs::` namespace of molpy's own `crates/molrs` core; it is not a separate
repository or Python package dependency.

The nanobind bridge obtains the capsule name and ABI line from
`molpy._core._ffi_abi_token()`. It checks the ABI against the linked
`crates/molrs-ffi` source, then lets `molcxx` validate the shared handle.
Capsule names are engine-private and must not be inferred from package names.

The dependency direction stays acyclic: molpy owns the Frame and never
imports Atomiverse; Atomiverse consumes that Frame and provides computation.
