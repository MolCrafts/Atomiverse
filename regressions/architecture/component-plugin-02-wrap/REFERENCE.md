# component-plugin-02-wrap

oracle: harness

Expected:

```
ok: wrap launches=3 slot_is_driver=0
```

Inner component launches == 3 (budget 3, one outer step). The nested slot is a
Component, not a DriverNode. The inner loop is physical, so it is mounted at the
outer `end_of_step`; mounting it inside the parent's force pipeline throws.
