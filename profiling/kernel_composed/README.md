# kernel_composed probes — tables for QuettaSim

Writes the `ncu/` vs `cuda_event/` layout that QuettaSim `kernel_composed`
interpolates. Not the live-server / serving-wall probes in `profiling/probes/`.

```bash
# from a QuettaSim checkout (QuettaBench as submodule):
cd QuettaBench/profiling/kernel_composed
just gpu=A100 profile-gpu
just gpu=A100 flash tag=tp1 nh=32 nkv=8 hd=128

# standalone Bench: point at the Sim tree
QUETTASIM=/path/to/QuettaSim just gpu=A100 profile-gpu
```

Tables land in `$QUETTASIM/data/kernel_data/` (or `KERNEL_DATA`). YAML
derivers (`util_flops` / `frontend:` / `serving:`) stay in QuettaSim:

```bash
just -f $QUETTASIM/justfile derive-device A100
just -f $QUETTASIM/justfile serving-fit A100 Qwen3.5-9B
```

See `$QUETTASIM/device_spec/README.md` for the full "add a GPU" workflow. Schema contract: filenames
`{gpu}[_tpN].csv`, method split `ncu/` (graphed decode) vs `cuda_event/` (eager
prefill), columns as the current H100/A100 tables.
