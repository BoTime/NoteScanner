import {
  FILTER_SUBSTEP_ORDER,
  PHASE_ORDER,
  type FilterSubstep,
  type PhaseTiming,
  type SegmentationPhase,
  type TimingReport,
} from './types';

/**
 * Percentiles, not means — the same estimator as `playground/fixtures.ts`'s
 * `summarize`, deliberately duplicated rather than shared: that one belongs to
 * the benchmark harness and this one is shipped library code, and coupling
 * them would make a change to either a change to both.
 *
 * A mean hides the tail, and the tail is the whole question here: an
 * everything-mode pass whose median batch is 40 ms but whose p95 is 900 ms
 * reads to a user as a hang, not as a fast segmenter.
 */
export function summarizePhase(samples: readonly number[]): PhaseTiming {
  if (samples.length === 0) return { p50: 0, p95: 0, max: 0, count: 0, total: 0 };

  const sorted = [...samples].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];

  return {
    p50: at(0.5),
    p95: at(0.95),
    max: sorted[sorted.length - 1],
    count: sorted.length,
    total: sorted.reduce((sum, value) => sum + value, 0),
  };
}

export interface TimingAccumulator {
  record(phase: SegmentationPhase, ms: number): void;
  /** A region inside the `filter` stage. Nests under it; does not replace it. */
  recordFilterSub(step: FilterSubstep, ms: number): void;
  /** `totalMs` is wall clock for the whole run, measured by the caller. */
  report(totalMs: number): TimingReport;
}

export function createTimingAccumulator(): TimingAccumulator {
  const samples = new Map<SegmentationPhase, number[]>();
  const filterSamples = new Map<FilterSubstep, number[]>();

  const push = <K>(into: Map<K, number[]>, key: K, ms: number) => {
    const bucket = into.get(key);
    if (bucket) bucket.push(ms);
    else into.set(key, [ms]);
  };

  return {
    record(phase, ms) {
      push(samples, phase, ms);
    },
    recordFilterSub(step, ms) {
      push(filterSamples, step, ms);
    },
    report(totalMs) {
      // Every phase gets a row even when it never ran, so the results table
      // has a stable shape and a missing phase reads as "0 samples" rather
      // than as a hole.
      const phases = {} as Record<SegmentationPhase, PhaseTiming>;
      for (const phase of PHASE_ORDER) {
        phases[phase] = summarizePhase(samples.get(phase) ?? []);
      }
      // Same zero-fill rule for the filter breakdown.
      const filterSubPhases = {} as Record<FilterSubstep, PhaseTiming>;
      for (const step of FILTER_SUBSTEP_ORDER) {
        filterSubPhases[step] = summarizePhase(filterSamples.get(step) ?? []);
      }
      return { phases, filterSubPhases, totalMs };
    },
  };
}
