# Short MTK volume parity

oracle: literature

G. J. Martyna, D. J. Tobias, and M. L. Klein, “Constant pressure molecular
dynamics algorithms,” J. Chem. Phys. **101**, 4177 (1994),
[doi:10.1063/1.467468](https://doi.org/10.1063/1.467468).

A deterministic 216-atom LJ lattice undergoes one or two MTK steps on each
backend. The CPU and CUDA volumes must agree to 1e-8 and 1e-6 relative,
respectively, and the first step must actually change the cell. These
thresholds are retained from the former MD pipelines under tests/; the
program is now a standalone public-API regression. Each case starts from
its own frame, uses 0.005 fs, and finishes within ten seconds on Arrhenius.
