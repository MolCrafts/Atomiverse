# MetalWalls twin — voronoi-volume-tf

Templates: substitute `@LTF@` (bohr) and `@VORONOI@` (bohr³) in `runtime.inpt`.

| case | LTF | VORONOI | Q_top [e] (c8ad175d) |
|------|-----|---------|----------------------|
| perfect | 0.0 | 113.741732075823890 | 4.354124119406114e-3 |
| TF bulk v | 3.779452249251540 | 113.741732075823890 | 3.707555065364343e-3 |
| TF half v | 3.779452249251540 | 56.870866037911945 | 3.321360199637453e-3 |

```bash
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
MW=…/work/opt/metalwalls/mw
sed -e 's/@LTF@/3.779452249251540/' -e 's/@VORONOI@/113.741732075823890/' \
  runtime.inpt > /tmp/rt.inpt
cp data.inpt /tmp/ && cp /tmp/rt.inpt /tmp/runtime.inpt
cd /tmp && mpirun -n 1 "$MW" && cat total_charges.out
```
