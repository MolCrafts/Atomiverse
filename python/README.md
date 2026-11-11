# atomiverse

Python front end of [Atomiverse](https://github.com/MolCrafts/Atomiverse), an
engine for multiscale, multiphysics atomistic simulation: classical MD, Monte
Carlo, geometry optimization, vibrational analysis, and electronic structure
(Gaussian SCF, plane-wave DFT, GPW, NDDO/AM1). Simulation state crosses the
boundary as a `molpy.Frame`.

## Install

This package is the pure-Python layer: the API and the `python -m
atomiverse` CLI, with no compiled code. The C++ extension is built for your
machine from the matching source tag:

```bash
pip install atomiverse
V=$(python -m atomiverse --version)
git clone --recursive --branch v$V https://github.com/MolCrafts/Atomiverse
pip install 'atomiverse[native]'

python -m atomiverse config md minimize --src Atomiverse
python -m atomiverse compile -j 8
python -m atomiverse install
```

`compile` builds a native wheel with exactly the listed modules, plus molpy
from the sibling `../molpy` checkout (`ATV_MOLPY_PATH`) so both sides of the
frame exchange are the same tree; `install` installs both with pip, so `pip uninstall atomiverse`
removes the extension.

Toolchain, module list and the full API:
[Python package guide](https://github.com/MolCrafts/Atomiverse/blob/master/docs/user-guide/python.md).

## License

GPL-2.0-only.
