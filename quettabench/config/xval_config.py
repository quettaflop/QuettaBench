"""What crossval/table.py and crossval/logit_agreement.py still import from
the retired config loader; removed in Phase 3, when crossval.py reads
results files instead. The validators live in config/validate.py; the
workload records come from configs/, not workloads.json.
"""

from pathlib import Path

import yaml

from quettabench.config.validate import (  # noqa: F401  (re-exported for table.py)
    check_ep_legal,
    disagg_gpu_sets,
    disagg_kv_config,
    parse_serving_style,
    serving_style_compatible,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def workloads():
    """{run name: {name, dtype[, comm_bound_bs]}} per configs/runs/*.yaml,
    its model yaml under the run's engine overrides: the three keys the two
    crossval scripts read."""
    out = {}
    for run in sorted((CONFIGS / "runs").glob("*.yaml")):
        raw = yaml.safe_load(run.read_text())
        model = yaml.safe_load((CONFIGS / "models" / f"{raw['model']}.yaml").read_text())
        engine = raw.get("engine") or {}
        out[run.stem] = {"name": model["name"], "dtype": engine.get("dtype", model["dtype"])}
        crossval = model.get("crossval") or {}
        if "comm_bound_bs" in crossval:
            out[run.stem]["comm_bound_bs"] = crossval["comm_bound_bs"]
    return out
