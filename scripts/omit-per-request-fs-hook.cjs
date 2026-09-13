'use strict';

// Patch fs.readFileSync before QuettaBoard's build:data runs.
// Result JSON keeps a huge top-level `per_request` array (tens of MB, ~27k
// rows) that build-data.ts JSON.parse's and then discards. Reading those
// bytes also charges the oneshot cgroup's page cache — after a cold boot
// that was a 42G peak and OOMed the host. Only the results tree is patched;
// validate:data still reads the full slim data.json.

const fs = require('fs');
const path = require('path');

const PREFIX_BYTES = Number(process.env.DASHBOARD_RESULT_PREFIX_BYTES || 262144);
const resultsRoot = process.env.BENCHMARK_RESULTS_DIR
  ? path.resolve(process.env.BENCHMARK_RESULTS_DIR)
  : '';

function underResults(filePath) {
  if (!resultsRoot) return false;
  const abs = path.resolve(String(filePath));
  return abs === resultsRoot || abs.startsWith(resultsRoot + path.sep);
}

function omitTopLevelArray(text, key) {
  const re = new RegExp(`,\\s*"${key}"\\s*:`);
  const match = re.exec(text);
  if (!match) return text;
  let i = match.index + match[0].length;
  while (i < text.length && /\s/.test(text[i])) i += 1;
  if (text[i] !== '[') return text;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (; i < text.length; i += 1) {
    const c = text[i];
    if (inStr) {
      if (esc) {
        esc = false;
        continue;
      }
      if (c === '\\') {
        esc = true;
        continue;
      }
      if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') {
      inStr = true;
      continue;
    }
    if (c === '[') depth += 1;
    else if (c === ']') {
      depth -= 1;
      if (depth === 0) {
        return text.slice(0, match.index) + text.slice(i + 1);
      }
    }
  }
  // Prefix read stopped inside per_request: drop the incomplete key.
  return `${text.slice(0, match.index).trimEnd().replace(/,\s*$/, '')}}`;
}

function slimResultPrefix(filePath) {
  const fd = fs.openSync(filePath, 'r');
  try {
    const buf = Buffer.alloc(PREFIX_BYTES);
    const n = fs.readSync(fd, buf, 0, buf.length, 0);
    let text = buf.toString('utf8', 0, n);
    if (!text.includes('"per_request"')) return null;
    text = omitTopLevelArray(text, 'per_request');
    if (!text.trimEnd().endsWith('}')) text = `${text.trimEnd().replace(/,\s*$/, '')}}`;
    return text;
  } finally {
    fs.closeSync(fd);
  }
}

const origReadFileSync = fs.readFileSync;
fs.readFileSync = function patchedReadFileSync(filePath, options) {
  if (
    underResults(filePath) &&
    String(filePath).endsWith('.json') &&
    !String(filePath).endsWith('_per_turn.json')
  ) {
    try {
      const st = fs.statSync(filePath);
      if (st.size > PREFIX_BYTES) {
        const slim = slimResultPrefix(filePath);
        if (slim != null) {
          const enc = typeof options === 'string' ? options : options && options.encoding;
          if (enc || options == null || typeof options === 'string') return slim;
          return Buffer.from(slim, 'utf8');
        }
      }
    } catch {
      // Fall through to the original read.
    }
  }
  return origReadFileSync.apply(this, arguments);
};

module.exports = { omitTopLevelArray, slimResultPrefix, underResults };
