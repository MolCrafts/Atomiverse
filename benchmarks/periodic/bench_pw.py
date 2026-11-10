"""Periodic plane-wave PBE-family accuracy against live Quantum ESPRESSO."""

from benchmarks.periodic.cases import method_spec

SPEC = method_spec("pw")

if __name__ == "__main__":
    from benchmarks.__main__ import method_main

    raise SystemExit(method_main(SPEC))
