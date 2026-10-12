"use strict";

// Run with: node tests/js/test_long_rl_dashboard.js
// Exercise the same dependency-free helpers used by the browser frontend.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "../../web/long_rl_dashboard/app.js"), "utf8"), context);
const math = context.dashboardMath;

for (const missing of [null, undefined, NaN, Infinity, "1", false]) assert.equal(math.valid(missing), false);
assert.equal(math.avg([null, undefined, 2, 4]), 3);
assert.equal(math.avg([null, undefined]), null);
assert.equal(math.percentile([-17, 101, null], .5), 42);
assert.equal(math.percentile([], .5), null);

// Net healing is a real negative observation, and the highest endpoint belongs
// in the final histogram bin rather than being dropped by rounding.
const histogram = math.histogram([-17, -1, 0, 101, null], 24);
assert.equal(histogram.count, 4);
assert.equal(histogram.missing, 1);
assert.equal(histogram.bins.reduce((sum, bin) => sum + bin.count, 0), 4);
assert.equal(histogram.bins[0].low, -17);
assert.equal(histogram.bins.at(-1).count, 1);

// Rolling means weight logged round summaries equally, retain absent current
// values as gaps, and expose the number of actual contributing observations.
const rolled = math.rolling([{y: 2}, {y: null}, {y: 4}], 3);
assert.equal(rolled[1].y, null);
assert.equal(rolled[2].y, 3);
assert.equal(rolled[2].contributors, 2);
assert.equal(math.rolling([{y: 1}, {y: 9}, {y: 5}], 2)[2].y, 7);

assert.equal(math.outcomeClass({outcome: "missing", censored: true}), "other");
assert.equal(math.numericEligible({outcome: "missing", censored: true, coverage_valid: true}, true), false);
assert.equal(math.numericEligible({truncated: true, coverage_valid: true, numeric_eligible: false}, true), true);
assert.equal(math.numericEligible({truncated: true, coverage_valid: false}, true), false);
assert.equal(math.numericEligible({fault: true, numeric_eligible: true}, true), false);
assert.equal(math.numericEligible({requested_occurrence: false, numeric_eligible: true}, true), false);

// A failed requested rollout stays in the rate denominator; an unexpected
// duplicate observation contributes neither an extra request nor numeric data.
const rows = [
  {round: 3, outcome: "victory", reward: 1, floor_progress: 12, requested_occurrence: true, coverage_valid: true, numeric_eligible: true},
  {round: 3, fault: true, requested_occurrence: true, numeric_eligible: false},
  {round: 3, outcome: "victory", reward: 999, requested_occurrence: false, numeric_eligible: false}
];
const grouped = math.perRoundEpisodes(rows, false)[0];
assert.equal(grouped.n, 2);
assert.equal(grouped.numeric_n, 1);
assert.equal(grouped.episode_return, 1);
assert.equal(grouped.episode_floor_progress, 12);
assert.equal(grouped.episode_success, .5);
assert.equal(grouped.episode_fault_rate, .5);

// Reproduce the restore lifecycle: startup consumes saved choices, then the user
// clears a filter while options are loading. A later callback must not reapply
// the old saved choice and silently alter the user's selected cohort.
let saved = {source: "fresh_scratch", range: [0, 200]};
const controls = {source: ""};
saved = math.applySavedOnce(saved, value => { controls.source = value.source; });
assert.equal(controls.source, "fresh_scratch");
assert.equal(saved, null);
controls.source = "";
saved = math.applySavedOnce(saved, value => { controls.source = value.source; });
assert.equal(controls.source, "");
assert.equal(saved, null);

console.log("Dashboard JS regressions passed: missing values, signed histograms, rolling means, requested denominators, numeric eligibility and restore lifecycle.");
