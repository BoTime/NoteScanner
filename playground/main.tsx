import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { BenchmarkView } from './BenchmarkView';
import { SegmentView } from './SegmentView';
import { CompareView } from './CompareView';
import { BoundaryView } from './BoundaryView';
import '../src/styles.css';

type View = 'benchmark' | 'segment' | 'compare' | 'boundary';

function App() {
  const [view, setView] = useState<View>('benchmark');

  return (
    <main style={{ padding: 16, fontFamily: 'system-ui' }}>
      <h1>segment-viewer playground</h1>
      <nav style={{ marginBottom: 12 }}>
        <button
          data-testid="view-benchmark"
          type="button"
          aria-pressed={view === 'benchmark'}
          onClick={() => setView('benchmark')}
        >
          Benchmark
        </button>{' '}
        <button
          data-testid="view-segment"
          type="button"
          aria-pressed={view === 'segment'}
          onClick={() => setView('segment')}
        >
          Segment
        </button>{' '}
        <button
          data-testid="view-compare"
          type="button"
          aria-pressed={view === 'compare'}
          onClick={() => setView('compare')}
        >
          Compare
        </button>{' '}
        <button
          data-testid="view-boundary"
          type="button"
          aria-pressed={view === 'boundary'}
          onClick={() => setView('boundary')}
        >
          Boundary
        </button>
      </nav>
      {view === 'benchmark' && <BenchmarkView />}
      {view === 'segment' && <SegmentView />}
      {view === 'compare' && <CompareView />}
      {view === 'boundary' && <BoundaryView />}
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
