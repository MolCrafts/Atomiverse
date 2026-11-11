# Python Frame boundary (acyclic)

The MolCrafts/molpy repository owns the native Rust core, the `molcxx`
C++ bridge and the Python `molpy._core` extension. Atomiverse links `molcxx`
and accepts `molpy.Frame` objects through their shared FFI handle.

Both builds use the same local checkout selected by `ATV_MOLPY_PATH`
(default `../molpy`). No revision is pinned. The CXX bridge exports the
`molrs::` namespace of molpy's own `crates/molrs` core; it is not a separate
repository or Python package dependency.

The nanobind bridge takes the capsule name from the linked `molcxx` bridge
(`molrs::frameref_capsule_name()`). The name carries molpy's handle-layout
fingerprint, so a frame from a separately built molpy fails the capsule-name
check before its pointer is read. Capsule names are engine-private and must
not be inferred from package names.

The dependency direction stays acyclic: molpy owns the Frame and never
imports Atomiverse; Atomiverse consumes that Frame and provides computation.
