"""One idempotent provisioning pass for a GPU box. It replaced bootstrap.sh
and oneshot.sh, retired in e34d433.

  python3 -m quettabench.provision <run.yaml>... [--set [<run>:]k=v ...]
                                   [--backend local|docker] [--dry-run] [--env-file FILE]

Inputs are the hardware yaml's provision: block and --set; a `<run>:`
prefix scopes a --set to the run yaml of that stem. Every step checks
before it acts, so a second pass changes nothing:

  1. cache dirs under provision.cache_root, off the small container root
  2. a python with vLLM: provision.py, else pip install provision.vllm_spec
  3. NCCL: provision.nccl_lib must exist when given; without it a
     Blackwell GPU (compute capability >= 10) needs the python's NCCL
     >= 2.28
  4. a QuettaServe checkout at provision.qs_dir (default
     <cache_root>/QuettaServe), cloned from provision.qs_repo at qs_commit
  5. per run: the bench binary (provision.bench_bin, else built in the
     checkout), the engine checkpoint (qs_ckpt), the moe config (qs_cfg)
     and the vLLM weights (weights, else hf_id downloaded to
     <cache_root>/models/<run>)

--backend docker (hardware.launcher=docker) replaces steps 2 and 3, since
vLLM and its NCCL live in the image: docker must be on PATH and each run's
engine.image must exist locally (`docker image inspect`). A missing image
prints `MISSING: docker image <image>; pull it with: docker pull <image>`;
pulling is the user's action, never provision's.

What it cannot resolve prints `MISSING: <what>`; after listing them all
the pass exits 1, nothing guessed. --dry-run prints the plan and writes
nothing. A complete pass writes the env file (the justfile loads it as
.env) and prints each run's `qbench run` lines for both engines.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from quettabench.config import loader
from quettabench.config.schema import ConfigError

REPO = Path(__file__).resolve().parents[1]
CACHE_DIRS = ("vllm", "torch", "hf", "tmp", "models")
NCCL_FLOOR = (2, 28)
NCCL_VERSION = "import torch; print(*torch.cuda.nccl.version(), sep='.')"


class Plan:
    """Runs or, on a dry run, prints each action; counts what is missing."""

    def __init__(self, dry):
        self.dry, self.missing, self.env = dry, 0, {}

    def log(self, text):
        print(f"[provision] {text}", file=sys.stderr, flush=True)

    def need(self, what):
        self.log(f"MISSING: {what}")
        self.missing += 1

    def act(self, argv, what, cwd=None):
        """Run argv (or say it would); a failure is MISSING `what`."""
        if self.dry:
            return self.log(f"would: {shlex.join(argv)}")
        self.log(f"+ {shlex.join(argv)}")
        if subprocess.run(argv, cwd=cwd, check=False).returncode != 0:
            self.need(what)


def succeeds(argv):
    try:
        return subprocess.run(argv, capture_output=True, check=False).returncode == 0
    except OSError:
        return False


def compute_capability():
    """Major compute capability of GPU 0, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False).stdout
    except OSError:
        return None
    return int(float(out.split()[0])) if out.strip() else None


def nccl_version(py):
    out = subprocess.run([py, "-c", NCCL_VERSION], capture_output=True, text=True,
                         check=False).stdout.strip()
    return tuple(int(p) for p in out.split(".")[:2]) if out else None


def docker_images(plan, configs):
    """Docker on PATH and every run's engine.image present locally; nothing pulled."""
    if not shutil.which("docker"):
        return plan.need("docker is not on PATH; hardware.launcher=docker runs the server with it")
    for name, cfg in configs.items():
        image = cfg.engine.image
        if not image:
            plan.need(f"{name}: hardware.launcher=docker needs engine.image")
        elif plan.dry:
            plan.log(f"would: docker image inspect {image}")
        elif succeeds(["docker", "image", "inspect", image]):
            plan.log(f"{name}: docker image ok: {image}")
        else:
            plan.need(f"docker image {image}; pull it with: docker pull {image}")


def machine(plan, prov, docker=False):
    """Steps 1-4 (2 and 3 skipped for docker); returns (python, QuettaServe checkout)."""
    cache = Path(prov["cache_root"])
    if plan.dry:
        plan.log("would: mkdir -p " + " ".join(str(cache / d) for d in CACHE_DIRS))
    else:
        for d in CACHE_DIRS:
            (cache / d).mkdir(parents=True, exist_ok=True)
    plan.env.update({"VLLM_CACHE_ROOT": cache / "vllm", "XDG_CACHE_HOME": cache / "vllm",
                     "TORCHINDUCTOR_CACHE_DIR": cache / "torch", "HF_HOME": cache / "hf",
                     "TMPDIR": cache / "tmp"})

    py = prov.get("py") or "python3"
    plan.env["PYTHON_BIN"] = py
    if docker:
        plan.log("vllm: in the docker image")
    elif succeeds([py, "-m", "pip", "show", "vllm"]):
        plan.log(f"vllm python ok: {py}")
    elif prov.get("vllm_spec"):
        plan.act([py, "-m", "pip", "install", *shlex.split(prov["vllm_spec"])],
                 f"pip install {prov['vllm_spec']!r} failed")
    else:
        plan.need(f"no vllm in {py!r}; set hardware.provision.py to a vllm python or "
                  "hardware.provision.vllm_spec to install one")

    lib = prov.get("nccl_lib")
    if docker:
        plan.log("nccl: in the docker image")
    elif lib:
        if not Path(lib).is_dir():
            plan.need(f"hardware.provision.nccl_lib {lib!r} does not exist")
        plan.env["LD_LIBRARY_PATH"] = lib
    elif (compute_capability() or 0) >= 10 and (nccl_version(py) or (0,)) < NCCL_FLOOR:
        plan.need(f"Blackwell needs NCCL >= 2.28 and {py} has {nccl_version(py)}; set "
                  "hardware.provision.nccl_lib to a newer NCCL's lib dir")

    qs = Path(prov.get("qs_dir") or cache / "QuettaServe")
    if not (qs / ".git").is_dir() and prov.get("qs_repo"):
        plan.act(["git", "clone", prov["qs_repo"], str(qs)], f"git clone {prov['qs_repo']} failed")
        if prov.get("qs_commit"):
            plan.act(["git", "-C", str(qs), "checkout", prov["qs_commit"]],
                     f"git checkout {prov['qs_commit']} failed")
    return py, qs


def bench_binary(plan, name, bench, prov, qs):
    """The prebuilt binary, or the newest test binary cargo builds in qs."""
    if prov.get("bench_bin") and os.access(prov["bench_bin"], os.X_OK):
        plan.log(f"{name}: bench bin {prov['bench_bin']}")
        return prov["bench_bin"]
    crate, test = bench["crate"], bench["test"]
    if plan.dry:
        return plan.log(f"would: build {crate}/{test} in {qs}")
    if not (qs / crate / "Cargo.toml").exists():
        return plan.need(f"{name} engine: set {name}:hardware.provision.bench_bin, or "
                         "hardware.provision.qs_repo, or hardware.provision.qs_dir to a "
                         "QuettaServe checkout")
    plan.act(["cargo", "test", "-p", crate, "--test", test, "--release", "--no-run"],
             f"{name}: cargo build failed", cwd=qs)
    built = sorted((p for p in (qs / "target/release/deps").glob(f"{test}-*")
                    if p.suffix not in (".d", ".o")), key=lambda p: p.stat().st_mtime)
    if not built:
        return plan.need(f"{name}: no {test} binary after build")
    plan.log(f"{name}: built {built[-1]}")
    return str(built[-1])


def run(plan, name, cfg, py, qs):
    """Step 5 for one run; returns its `qbench run` lines, or none on a gap."""
    prov, bench = cfg.hardware.provision, cfg.engine.extra.get("quettaserve")
    if not bench:
        return plan.need(f"{name} has no quettaserve: block in its model yaml")
    binary = bench_binary(plan, name, bench, prov, qs)
    if not prov.get("qs_ckpt"):
        plan.need(f"{name} engine checkpoint: set {name}:hardware.provision.qs_ckpt")
    if bench["family"] == "moe" and not prov.get("qs_cfg"):
        plan.need(f"{name} moe config: set {name}:hardware.provision.qs_cfg")
    weights = prov.get("weights")
    if not weights and prov.get("hf_id"):
        weights = Path(prov["cache_root"]) / "models" / name
        if weights.is_dir() and any(weights.iterdir()):
            plan.log(f"{name}: weights {weights}")
        else:
            plan.act([py, "-m", "huggingface_hub.commands.huggingface_cli", "download",
                      prov["hf_id"], "--local-dir", str(weights)],
                     f"{name}: hf download {prov['hf_id']} failed (HF_TOKEN set?)")
    elif not weights:
        plan.need(f"{name} weights: set {name}:hardware.provision.weights=<dir> or "
                  f"{name}:hardware.provision.hf_id=<hf-repo-id>")
    qs_sets = [("engine.name", "quettaserve"), ("engine.model", prov.get("qs_ckpt")),
               ("engine.extra.bench_bin", binary), ("engine.extra.qs_cfg", prov.get("qs_cfg"))]
    yaml_path = f"configs/runs/{name}.yaml"
    return [shlex.join(["qbench", "run", yaml_path, "--set", f"engine.model={weights}"]),
            shlex.join(["qbench", "run", yaml_path, *[a for k, v in qs_sets if v
                                                      for a in ("--set", f"{k}={v}")]])]


def split_sets(items):
    """(sets for every run, {run stem: its own sets}) from `[<run>:]k=v` items."""
    common, scoped = [], {}
    for item in items:
        key, sep, value = item.partition("=")
        stem, colon, path = key.partition(":")
        if colon:
            scoped.setdefault(stem, []).append(f"{path}{sep}{value}")
        else:
            common.append(item)
    return common, scoped


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qbench provision", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_yaml", nargs="+", help="configs/runs/<name>.yaml to provision for")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="[RUN:]K=V")
    ap.add_argument("--backend", choices=("local", "docker"),
                    help="sets hardware.launcher for every run")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    ap.add_argument("--env-file", default=str(REPO / ".env"))
    a = ap.parse_args(argv)
    common, scoped = split_sets(a.overrides)
    if a.backend:
        common.append(f"hardware.launcher={a.backend}")
    plan = Plan(a.dry_run)
    configs = {}
    for path in a.run_yaml:
        name = Path(path).stem
        try:
            points = loader.load(path, ["allow_unverified=true", *common, *scoped.get(name, [])])
        except (ConfigError, ValueError) as e:
            plan.need(str(e))
            continue
        if len(points) > 1:
            plan.need(f"{path}: a matrix of {len(points)} points; provision a run yaml "
                      "without matrix: for the same model")
            continue
        configs[name] = points[0]
    if not configs:
        return 1
    docker = {n: c for n, c in configs.items() if c.hardware.launcher == "docker"}
    if docker:
        docker_images(plan, docker)
    py, qs = machine(plan, next(iter(configs.values())).hardware.provision,
                     len(docker) == len(configs))
    lines = [line for name, cfg in configs.items() for line in run(plan, name, cfg, py, qs) or ()]
    if plan.missing:
        plan.log("provisioning incomplete; resolve the MISSING items above")
        return 1
    if plan.dry:
        plan.log("dry run: plan only, nothing written")
        return 0
    Path(a.env_file).write_text("".join(f"{k}={v}\n" for k, v in plan.env.items()))
    plan.log(f"wrote {a.env_file}; next:")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
