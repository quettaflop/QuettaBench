# QuettaBench systemd units

Neutral, repo-versioned systemd units for the QuettaBench profiling/benchmark
pipeline. These are **authored here, not installed**. They mirror the live
agentic-serve orchestration units but retarget everything at this repo
(`WorkingDirectory=/root/QuettaBench`, `ExecStart=/root/QuettaBench/scripts/...`)
and route every dashboard-JSON artifact into the neutral
`BENCH_ARTIFACT_DIR=/mnt/100g/agent-bench/artifacts` (the dashboard frontend now
lives in the separate QuettaBoard repo, referenced via
`DASHBOARD_DIR=/root/QuettaBoard`).

## Units

Each service is `Type=oneshot` and is driven by its paired timer.

| Service | Timer cadence | What it does |
| --- | --- | --- |
| `quetta-bench-orchestrator.service` | boot+5min, then every 10min | One orchestration tick: compiles `sweep.yaml` into a scoped job manifest, syncs GPU code, runs coverage-reconcile + GPU-reclaim preflights, dispatches the next pending job per idle host, detects completions/OOMs, and publishes `sweep-state.json`. Runs `scripts/run-bench-orchestrator-service.sh` (which wraps `sync-gpu-code.sh` + `bench_orchestrator.sh`). **Dispatches work and writes state.** |
| `quetta-bench-dashboard-refresh.service` | boot+2min, then every 15min | Rebuilds dashboard-JSON artifacts (`sweep-state.json`, `data.json` + scoped sidecars via QuettaBoard's `build:data`, `coverage-blockers.*.json`, `gpu-state.json`) into `BENCH_ARTIFACT_DIR`, then validates them. Runs `scripts/rebuild-local-dashboard.sh`. Does **not** build the frontend bundle (QuettaBoard owns that via `deploy:tailscale`); the local `vite build` + promote is guarded behind `BENCH_BUILD_BUNDLE=1`, default off. |
| `quetta-bench-gpu-state-refresh.service` | boot+1min, then every 1min | Regenerates just the private `gpu-state.json` (per-host GPU occupancy) into `BENCH_ARTIFACT_DIR`. Runs `scripts/refresh-gpu-state.sh`. Read-only w.r.t. sweep state. |
| `quetta-bench-gpu-orphan-cleaner.service` | boot+3min, then every 5min | Reclaims orphaned GPU processes via `scripts/clean_orphan_gpus.py --execute`. **Kills remote processes.** |

## Install (run manually; nothing here installs itself)

```sh
# Option A: copy into /etc
sudo cp /root/QuettaBench/deploy/systemd/quetta-bench-*.{service,timer} /etc/systemd/system/

# Option B: symlink the repo copies (keeps them versioned in-tree)
sudo systemctl link /root/QuettaBench/deploy/systemd/quetta-bench-*.service
sudo systemctl link /root/QuettaBench/deploy/systemd/quetta-bench-*.timer

sudo systemctl daemon-reload

# Enable the TIMERS (the services are triggered by their timers; do not enable
# the services directly).
sudo systemctl enable --now quetta-bench-orchestrator.timer
sudo systemctl enable --now quetta-bench-dashboard-refresh.timer
sudo systemctl enable --now quetta-bench-gpu-state-refresh.timer
sudo systemctl enable --now quetta-bench-gpu-orphan-cleaner.timer
```

To validate a unit before installing: `systemd-analyze verify quetta-bench-*.service`.

## WARNINGS

**Do NOT enable `quetta-bench-orchestrator.timer` while the live
`agentic-serve-bench-orchestrator.timer` is still enabled.** Both dispatch onto
the same GPU hosts and share `/mnt/100g/agent-bench/state` +
`/mnt/100g/agent-bench/results`, so running both causes double-dispatch and
corrupted sweep state. Disable the agentic-serve orchestrator timer first, or
keep this one off until the migration cutover.

The same double-execute hazard applies to
`quetta-bench-gpu-orphan-cleaner.timer` (it `--execute`s kills against the shared
hosts): do not run it alongside `agentic-serve-gpu-orphan-cleaner.timer`.

The read-only refreshers (`quetta-bench-gpu-state-refresh.timer` and
`quetta-bench-dashboard-refresh.timer`) are safe to shadow-run alongside the
live agentic-serve refreshers **only if** `BENCH_ARTIFACT_DIR` is pointed at a
scratch directory distinct from the live artifact dir, so their JSON writes do
not clobber the live dashboard's inputs. Override it per-unit, e.g.:

```sh
sudo systemctl edit quetta-bench-dashboard-refresh.service
# [Service]
# Environment=BENCH_ARTIFACT_DIR=/mnt/100g/agent-bench/artifacts-shadow
```

Note that R2 mirroring is **off by default** in these units (the
`rebuild-local-dashboard.sh` `--mirror-r2` block only runs when `MIRROR_R2=1`,
which is not set here). Leave it off for shadow runs so scratch artifacts are
never uploaded to the shared `agent-bench` R2 bucket.
