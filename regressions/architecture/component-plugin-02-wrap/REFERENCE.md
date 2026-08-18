# component-plugin-02-wrap

oracle: harness

Expected:

```
ok: wrap launches=3 adapter_is_driver=0
```

Inner component launches == 3. Adapter is not a DriverNode. Outer plan op `is_driver() == false`.
