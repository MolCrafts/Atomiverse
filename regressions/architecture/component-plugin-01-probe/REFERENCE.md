# component-plugin-01-probe

oracle: harness (architecture probes; no literature energy)

Expected stdout (ATV_HAS_MD on):

```
component energy=0 force=0
potential energy=1 force=1
bias energy=1 force=1
path_coupling energy=1 force=1
semiempirical energy=1 force=1
plane_wave energy=1 force=1
gpw energy=1 force=1
hf energy=1 force=1
langevin energy=0 force=1
```

Without `ATV_HAS_MD` the `langevin` row is omitted. Exit 0 on match.
