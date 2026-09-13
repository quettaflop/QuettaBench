'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'omit-pr-'));
const results = path.join(tmp, 'results');
fs.mkdirSync(results);
process.env.BENCHMARK_RESULTS_DIR = results;
process.env.DASHBOARD_RESULT_PREFIX_BYTES = '4096';

const { omitTopLevelArray } = require('./omit-per-request-fs-hook.cjs');

const small = '{"config":{"model":"m"},"summary":{"median_ttft_ms":1},"per_request":[{"success":true}]}';
assert.deepStrictEqual(JSON.parse(omitTopLevelArray(small, 'per_request')), {
  config: { model: 'm' },
  summary: { median_ttft_ms: 1 },
});

const prefix = '{"config":{"model":"m"},"summary":{"median_ttft_ms":1},"per_request":[{"success":true},';
assert.deepStrictEqual(JSON.parse(omitTopLevelArray(prefix, 'per_request')), {
  config: { model: 'm' },
  summary: { median_ttft_ms: 1 },
});

const fat = {
  config: { model: 'm', backend: 'vllm', profile: 'chat', concurrency: 1 },
  summary: { median_ttft_ms: 2, concurrency: 1 },
  per_request: Array.from({ length: 2000 }, (_, i) => ({ success: true, i })),
};
const fatPath = path.join(results, 'fat.json');
fs.writeFileSync(fatPath, JSON.stringify(fat));
assert.ok(fs.statSync(fatPath).size > 20_000);

const slim = JSON.parse(fs.readFileSync(fatPath, 'utf8'));
assert.strictEqual(slim.summary.median_ttft_ms, 2);
assert.strictEqual(slim.per_request, undefined);

const outside = path.join(tmp, 'other.json');
fs.writeFileSync(outside, JSON.stringify(fat));
assert.ok(JSON.parse(fs.readFileSync(outside, 'utf8')).per_request);

fs.rmSync(tmp, { recursive: true, force: true });
console.log('omit-per-request-fs-hook.test.cjs: ok');
