"""PM3 native Python gate against this case's existing MOPAC fixtures.

Run with the built extension on PYTHONPATH: python python_nddo.py --device both.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python_nddo_checks import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[3] /
         "tools/tb_oracle/inputs/mopac/mndo-pm3-23.2.5/pins.json",
         ('PM3',))
