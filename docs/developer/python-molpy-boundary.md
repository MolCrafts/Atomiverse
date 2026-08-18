# Python Frame boundary (acyclic)

```
molrs (C++/Rust schema + FFI)     ← atomiverse C++ only
molpy (Python toolkit)            ← depends on molrs; never on atomiverse
atomiverse C++ library            ← depends on molrs; never on molpy or app/
atomiverse CLI (app/)             ← depends on the library; never the reverse
atomiverse Python package         ← user Frame is molpy.Frame
```

There is **no** `atomiverse → molpy → atomiverse` cycle. Users `import molpy`
only. The nanobind bridge still reads the engine-private `molrs.FrameRef`
capsule that molpy.Frame carries — that is not a user import.
