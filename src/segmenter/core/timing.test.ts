import { describe, it, expect } from 'vitest';
import { PHASE_ORDER, FILTER_SUBSTEP_ORDER } from './types';
import { createTimingAccumulator, summarizePhase } from './timing';

describe('summarizePhase', () => {
  it('is all zeros for no samples', () => {
    expect(summarizePhase([])).toEqual({ p50: 0, p95: 0, max: 0, count: 0, total: 0 });
  });

  it('reports percentiles, max, count and total', () => {
    const summary = summarizePhase([5, 1, 4, 2, 3]);
    expect(summary.p50).toBe(3);
    expect(summary.max).toBe(5);
    expect(summary.count).toBe(5);
    expect(summary.total).toBe(15);
  });

  it('pins p95 to the largest sample rather than reading past the end', () => {
    expect(summarizePhase([1, 2]).p95).toBe(2);
    expect(summarizePhase([7]).p95).toBe(7);
  });

  it('does not mutate the caller array', () => {
    const samples = [3, 1, 2];
    summarizePhase(samples);
    expect(samples).toEqual([3, 1, 2]);
  });
});

describe('createTimingAccumulator', () => {
  it('reports every phase in PHASE_ORDER, zero-filled when never recorded', () => {
    const report = createTimingAccumulator().report(0);
    expect(Object.keys(report.phases)).toEqual([...PHASE_ORDER]);
    for (const phase of PHASE_ORDER) {
      expect(report.phases[phase].count).toBe(0);
    }
  });

  it('accumulates repeated samples for one phase', () => {
    const timings = createTimingAccumulator();
    timings.record('decode', 10);
    timings.record('decode', 30);
    timings.record('decode', 20);
    const report = timings.report(120);
    expect(report.phases.decode.count).toBe(3);
    expect(report.phases.decode.p50).toBe(20);
    expect(report.phases.decode.max).toBe(30);
    expect(report.phases.decode.total).toBe(60);
    expect(report.phases.encode.count).toBe(0);
    expect(report.totalMs).toBe(120);
  });

  it('keeps phases independent', () => {
    const timings = createTimingAccumulator();
    timings.record('encode', 100);
    timings.record('nms', 5);
    const report = timings.report(200);
    expect(report.phases.encode.total).toBe(100);
    expect(report.phases.nms.total).toBe(5);
    expect(report.phases.decode.total).toBe(0);
  });
});

describe('createTimingAccumulator filter sub-steps', () => {
  it('reports every sub-step in FILTER_SUBSTEP_ORDER, zero-filled when never recorded', () => {
    const report = createTimingAccumulator().report(0);
    expect(Object.keys(report.filterSubPhases)).toEqual([...FILTER_SUBSTEP_ORDER]);
    for (const step of FILTER_SUBSTEP_ORDER) {
      expect(report.filterSubPhases[step].count).toBe(0);
    }
  });

  it('accumulates repeated samples for one sub-step', () => {
    const timings = createTimingAccumulator();
    timings.recordFilterSub('upscale', 10);
    timings.recordFilterSub('upscale', 30);
    timings.recordFilterSub('upscale', 20);
    const report = timings.report(60);
    expect(report.filterSubPhases.upscale.count).toBe(3);
    expect(report.filterSubPhases.upscale.p50).toBe(20);
    expect(report.filterSubPhases.upscale.max).toBe(30);
    expect(report.filterSubPhases.upscale.total).toBe(60);
    expect(report.filterSubPhases.select.count).toBe(0);
  });

  it('keeps sub-steps independent of the phases they nest inside', () => {
    const timings = createTimingAccumulator();
    timings.record('filter', 100);
    timings.recordFilterSub('select', 3);
    const report = timings.report(100);
    expect(report.phases.filter.total).toBe(100);
    expect(report.filterSubPhases.select.total).toBe(3);
    expect(report.filterSubPhases.threshold.total).toBe(0);
  });

  // The no-residual invariant: the worker's three regions are drawn to be
  // exhaustive and non-overlapping across the stage, so their totals sum to
  // the filter total. A residual is exactly what would muddy the percentage
  // breakdown this instrumentation exists to produce.
  it('has the three sub-step totals sum to the filter total with no residual', () => {
    const timings = createTimingAccumulator();
    timings.record('filter', 100);
    timings.recordFilterSub('select', 3);
    timings.recordFilterSub('upscale', 80);
    timings.recordFilterSub('threshold', 17);
    timings.record('filter', 50);
    timings.recordFilterSub('select', 2);
    timings.recordFilterSub('upscale', 40);
    timings.recordFilterSub('threshold', 8);
    const report = timings.report(150);
    const subTotal = FILTER_SUBSTEP_ORDER.reduce(
      (sum, step) => sum + report.filterSubPhases[step].total,
      0,
    );
    expect(subTotal).toBe(report.phases.filter.total);
  });
});
