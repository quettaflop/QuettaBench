#!/usr/bin/env python3
"""Coverage audit — compares actual R2 results against the target grid.

Run after sweep ticks to identify profile×conc gaps. Outputs a compact
report showing:
  1. Overall coverage % (filled / expected)
  2. Per-host gap summary
  3. Specific missing profile×conc combos per model/hw

Designed to be called from the orchestrator cron or sweep sentry.

Usage:
    python3 coverage_audit.py [--data-url URL] [--json]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from urllib.request import urlopen

SINGLE_PROFILES = [
    'chat-singleturn',
    'coding-singleturn',
]
MULTI_PROFILES = [
    'chat-multiturn',
    'swebench-multiturn',
    'terminalbench-multiturn',
    'osworld-multiturn',
]
SINGLE_CONCS = [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500]
MULTI_CONCS = [5, 10, 20, 40, 80, 120, 160, 200, 256, 320]

# TODO(phase-1): the dashboard's data.json is produced by the separate QuettaBoard
# repo. Override with COVERAGE_AUDIT_DATA_URL, or point DASHBOARD_DIR at the
# QuettaBoard checkout, once its public/ layout is finalized.
_DASHBOARD_DIR = os.environ.get("DASHBOARD_DIR", "/root/QuettaBoard")
DATA_URL = os.environ.get(
    "COVERAGE_AUDIT_DATA_URL", f"file://{_DASHBOARD_DIR}/public/data.json"
)


def load_data(url):
    with urlopen(url) as r:
        return json.loads(r.read())


def audit(data):
    filled = defaultdict(set)
    for b in data:
        cfg = b.get('config', {})
        hw = b.get('hardware', '')
        model = b.get('modelShort') or b.get('model_short', '') or ''
        backend = cfg.get('backend', 'vllm')
        mode = cfg.get('mode', '')
        profile = cfg.get('profile', '')
        conc = cfg.get('concurrency', 0)
        if not hw or not profile or not conc:
            continue
        key = (hw, model, backend)
        filled[key].add((mode, profile, conc))

    results = []
    for key, present in sorted(filled.items()):
        hw, model, backend = key
        expected_single = {('single-turn', p, c) for p in SINGLE_PROFILES for c in SINGLE_CONCS}
        expected_multi = {('multi-turn', p, c) for p in MULTI_PROFILES for c in MULTI_CONCS}
        expected = expected_single | expected_multi
        have = present & expected
        missing = expected - present

        missing_single_profiles = defaultdict(list)
        for m, p, c in sorted(missing):
            if m == 'single-turn':
                missing_single_profiles[p].append(c)
        missing_multi_profiles = defaultdict(list)
        for m, p, c in sorted(missing):
            if m == 'multi-turn':
                missing_multi_profiles[p].append(c)

        results.append({
            'hardware': hw,
            'model': model,
            'backend': backend,
            'have': len(have),
            'expected': len(expected),
            'pct': round(100 * len(have) / len(expected), 1) if expected else 0,
            'missing_single': dict(missing_single_profiles),
            'missing_multi': dict(missing_multi_profiles),
        })

    return results


def print_report(results):
    total_have = sum(r['have'] for r in results)
    total_expected = sum(r['expected'] for r in results)
    pct = round(100 * total_have / total_expected, 1) if total_expected else 0

    print(f"=== Coverage Audit ===")
    print(f"Overall: {total_have}/{total_expected} ({pct}%)")
    print(f"Combos tracked: {len(results)}")
    print()

    by_hw = defaultdict(list)
    for r in results:
        by_hw[r['hardware']].append(r)

    for hw, entries in sorted(by_hw.items()):
        hw_have = sum(e['have'] for e in entries)
        hw_exp = sum(e['expected'] for e in entries)
        hw_pct = round(100 * hw_have / hw_exp, 1) if hw_exp else 0
        print(f"--- {hw}: {hw_have}/{hw_exp} ({hw_pct}%) ---")
        for e in sorted(entries, key=lambda x: x['pct']):
            n_miss_s = sum(len(v) for v in e['missing_single'].values())
            n_miss_m = sum(len(v) for v in e['missing_multi'].values())
            print(f"  {e['model']}/{e['backend']}: {e['have']}/{e['expected']} ({e['pct']}%) "
                  f"[miss: {n_miss_s} single, {n_miss_m} multi]")
            if e['missing_multi']:
                for p, concs in sorted(e['missing_multi'].items())[:3]:
                    print(f"    multi missing: {p} @ concs {concs[:5]}{'...' if len(concs) > 5 else ''}")
            if e['missing_single']:
                for p, concs in sorted(e['missing_single'].items())[:3]:
                    print(f"    single missing: {p} @ concs {concs[:5]}{'...' if len(concs) > 5 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-url', default=DATA_URL)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    data = load_data(args.data_url)
    results = audit(data)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
    else:
        print_report(results)


if __name__ == '__main__':
    main()
