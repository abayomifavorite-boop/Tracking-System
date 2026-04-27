const { useState, useEffect, useCallback, useMemo, useRef } = React;

// ═══════════════════════════════════════════════════════════════════
// DATA STRUCTURE: BINARY MIN-HEAP (Priority Queue)
// Used by Dijkstra's algorithm for O(log n) extract-min operations.
// A naive array scan would give O(n²) — unacceptable for real-time use.
// ═══════════════════════════════════════════════════════════════════

class MinHeap {
  constructor() { this.h = []; }

  push(item) {
    this.h.push(item);
    this._up(this.h.length - 1);
  }

  pop() {
    if (!this.h.length) return null;
    const top = this.h[0];
    const last = this.h.pop();
    if (this.h.length) { this.h[0] = last; this._down(0); }
    return top;
  }

  get size() { return this.h.length; }

  _up(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.h[p].cost <= this.h[i].cost) break;
      [this.h[p], this.h[i]] = [this.h[i], this.h[p]];
      i = p;
    }
  }

  _down(i) {
    const n = this.h.length;
    for (;;) {
      let s = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.h[l].cost < this.h[s].cost) s = l;
      if (r < n && this.h[r].cost < this.h[s].cost) s = r;
      if (s === i) break;
      [this.h[s], this.h[i]] = [this.h[i], this.h[s]];
      i = s;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// ALGORITHM: DIJKSTRA'S SHORTEST PATH
// Finds the shortest path from `start` to the nearest node in `targetSet`.
// Complexity: O((V + E) log V) with binary heap.
// Returns: { path: string[], distance: number, target: string|null }
// ═══════════════════════════════════════════════════════════════════

function dijkstra(graph, start, targetSet) {
  if (!graph[start]) return { path: [], distance: Infinity, target: null };

  const dist = {};
  const prev = {};
  const heap = new MinHeap();

  // Initialise all distances to ∞
  for (const n of Object.keys(graph)) { dist[n] = Infinity; prev[n] = null; }

  dist[start] = 0;
  heap.push({ node: start, cost: 0 });

  while (heap.size > 0) {
    const { node, cost } = heap.pop();

    // Early exit: we've reached a safe zone node
    if (targetSet.has(node) && node !== start) {
      const path = [];
      for (let c = node; c !== null; c = prev[c]) path.unshift(c);
      return { path, distance: cost, target: node };
    }

    // Stale heap entry — skip
    if (cost > dist[node]) continue;

    for (const { neighbor, weight } of graph[node]) {
      const nc = cost + weight;
      if (nc < dist[neighbor]) {
        dist[neighbor] = nc;
        prev[neighbor] = node;
        heap.push({ node: neighbor, cost: nc });
      }
    }
  }

  return { path: [], distance: Infinity, target: null };
}

// ═══════════════════════════════════════════════════════════════════
// GRAPH CONSTRUCTION
// Models the geographic zone as an 8-directional weighted grid graph.
// Cardinal edges weight = 1.0, diagonal edges weight = √2 ≈ 1.414.
// ═══════════════════════════════════════════════════════════════════

const COLS = 20;
const ROWS = 14;
const CELL = 34;
const MAP_W = COLS * CELL;
const MAP_H = ROWS * CELL;

// Safe zone: bounded rectangular region
const SZ = { x1: 5, y1: 3, x2: 13, y2: 10 };

const DIRS = [
  [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
  [1, 1, 1.4142], [-1, 1, 1.4142], [1, -1, 1.4142], [-1, -1, 1.4142],
];

function buildGraph() {
  const g = {};
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const id = `${x},${y}`;
      g[id] = [];
      for (const [dx, dy, w] of DIRS) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < COLS && ny >= 0 && ny < ROWS)
          g[id].push({ neighbor: `${nx},${ny}`, weight: w });
      }
    }
  }
  return g;
}

function buildSafeZone() {
  const z = new Set();
  for (let y = SZ.y1; y <= SZ.y2; y++)
    for (let x = SZ.x1; x <= SZ.x2; x++)
      z.add(`${x},${y}`);
  return z;
}

// ═══════════════════════════════════════════════════════════════════
// SUBJECT DEFINITIONS
// ═══════════════════════════════════════════════════════════════════

const SUBJECTS_DEF = [
  {
    id: "C1",
    label: "Dara",
    type: "Child",
    short: "D",
    color: "#00ff66",
    alertColor: "#66ffaa",
    initPos: "9,6",
  },
  {
    id: "E1",
    label: "Grace",
    type: "Elderly",
    short: "E",
    color: "#22d3ee",
    alertColor: "#67e8f9",
    initPos: "9,7",
  },
];

// ═══════════════════════════════════════════════════════════════════
// GPS SIMULATION — MOVEMENT ENGINE
// Simulates realistic movement: 20% chance of wandering (can exit
// safe zone), 80% chance of preferring safe zone neighbors.
// ═══════════════════════════════════════════════════════════════════

function simulateGPS(subject, graph, safeZone) {
  const neighbors = graph[subject.pos]?.map((e) => e.neighbor) || [];
  if (!neighbors.length) return subject.pos;

  const WANDER_PROB = 0.20;

  if (Math.random() < WANDER_PROB) {
    // Pure random step — can exit safe zone
    return neighbors[Math.floor(Math.random() * neighbors.length)];
  }

  // Biased walk — prefer safe zone
  const safeNeighbors = neighbors.filter((n) => safeZone.has(n));
  const pool = safeNeighbors.length > 0 ? safeNeighbors : neighbors;
  return pool[Math.floor(Math.random() * pool.length)];
}

// ═══════════════════════════════════════════════════════════════════
// NODE → PIXEL
// ═══════════════════════════════════════════════════════════════════

function nxy(nodeId) {
  const [x, y] = nodeId.split(",").map(Number);
  return { x: x * CELL + CELL / 2, y: y * CELL + CELL / 2 };
}

// ═══════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════

function TrackingSystem() {
  const graph    = useMemo(buildGraph, []);
  const safeZone = useMemo(buildSafeZone, []);

  // ── Static grid nodes (memoised — never changes) ──
  const gridDots = useMemo(() => {
    const dots = [];
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const inSZ = x >= SZ.x1 && x <= SZ.x2 && y >= SZ.y1 && y <= SZ.y2;
        dots.push(
          React.createElement('circle', {
            key: `d${x},${y}`,
            cx: x * CELL + CELL / 2,
            cy: y * CELL + CELL / 2,
            r: inSZ ? 2 : 1.2,
            fill: inSZ ? "#0d3a30" : "#0a1e2e"
          })
        );
      }
    }
    return dots;
  }, []);

  // ── Simulation state (single atomic object for consistency) ──
  const mkInitState = () => ({
    subjects: SUBJECTS_DEF.map((s) => ({
      ...s,
      pos: s.initPos,
      trail: [s.initPos],
      isBreached: false,
    })),
    paths: {},
    alerts: [],
    stats: { ticks: 0, breaches: 0, avgMs: "0.00", allMs: [] },
  });

  const [sim, setSim] = useState(mkInitState);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1200);
  const timerRef = useRef(null);

  // ── Simulation tick ──
  const tick = useCallback(() => {
    setSim((prev) => {
      const newPaths  = {};
      const newAlerts = [];
      const msTimes   = [];

      const subjects = prev.subjects.map((sub) => {
        // Simulate GPS reading
        const pos = simulateGPS(sub, graph, safeZone);
        const isBreached = !safeZone.has(pos);

        if (isBreached) {
          // Run Dijkstra to compute optimal return route
          const t0 = performance.now();
          const result = dijkstra(graph, pos, safeZone);
          const ms = performance.now() - t0;
          msTimes.push(ms);

          if (result.path.length) newPaths[sub.id] = result.path;

          newAlerts.push({
            id: `${sub.id}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            subjectId: sub.id,
            label: `${sub.label} (${sub.type})`,
            color: sub.color,
            pos,
            steps: result.path.length,
            dist: result.distance.toFixed(2),
            ms: ms.toFixed(2),
            time: new Date().toLocaleTimeString(),
          });
        }

        return {
          ...sub,
          pos,
          isBreached,
          trail: [...sub.trail, pos].slice(-30),
        };
      });

      const allMs = [...prev.stats.allMs, ...msTimes].slice(-400);
      const avg =
        allMs.length
          ? (allMs.reduce((a, b) => a + b, 0) / allMs.length).toFixed(2)
          : "0.00";

      return {
        subjects,
        paths: newPaths,
        alerts: newAlerts.length
          ? [...newAlerts, ...prev.alerts].slice(0, 80)
          : prev.alerts,
        stats: {
          ticks:   prev.stats.ticks + 1,
          breaches: prev.stats.breaches + newAlerts.length,
          avgMs:   avg,
          allMs,
        },
      };
    });
  }, [graph, safeZone]);

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(tick, speed);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [running, tick, speed]);

  const anyBreached = sim.subjects.some((s) => s.isBreached);

  // ── RENDER ──
  return React.createElement('div', { className: 'app' },

    // HEADER
    React.createElement('div', { className: 'hdr' },
      React.createElement('div', { className: 'hdr-left' },
        React.createElement('div', { className: 'hdr-title' }, 'SafeTrack Guardian System'),
        React.createElement('div', { className: 'hdr-sub' },
          'GPS Simulation · Dijkstra Shortest Path · Geofence Detection · Real-Time Alerts'
        )
      ),
      React.createElement('div', { className: 'hdr-meta' },
        React.createElement('div', { style: { fontSize: 9, color: '#0f3a55', letterSpacing: 1 } },
          `NODES: ${COLS * ROWS} · EDGES: ${COLS * ROWS * 8}`
        ),
        React.createElement('div', { className: `sys-badge ${anyBreached ? 'sys-brk' : 'sys-ok'}` },
          React.createElement('div', { className: 'dot-pulse' }),
          anyBreached ? 'BREACH DETECTED' : 'ALL CLEAR'
        )
      )
    ),

    // MAIN
    React.createElement('div', { className: 'main' },
      // MAP SECTION
      React.createElement('div', { className: 'map-wrap' },
        React.createElement('div', { className: 'map-hdr' },
          React.createElement('div', { className: 'map-tag' }, '// Geographic Zone — 8-Directional Graph'),
          React.createElement('div', { className: 'algo-tag' },
            'Algorithm: ',
            React.createElement('span', null, "Dijkstra's SSSP"),
            ' · Heap: ',
            React.createElement('span', null, 'Binary Min-Heap')
          )
        ),

        React.createElement('svg', {
          className: 'map',
          width: MAP_W,
          height: MAP_H,
          viewBox: `0 0 ${MAP_W} ${MAP_H}`
        },
          React.createElement('defs', null,
            React.createElement('filter', { id: 'glow-teal' },
              React.createElement('feGaussianBlur', { stdDeviation: '2.5', result: 'b' }),
              React.createElement('feMerge', null,
                React.createElement('feMergeNode', { in: 'b' }),
                React.createElement('feMergeNode', { in: 'SourceGraphic' })
              )
            ),
            React.createElement('filter', { id: 'glow-red' },
              React.createElement('feGaussianBlur', { stdDeviation: '3', result: 'b' }),
              React.createElement('feMerge', null,
                React.createElement('feMergeNode', { in: 'b' }),
                React.createElement('feMergeNode', { in: 'SourceGraphic' })
              )
            )
          ),

          // Grid lines
          Array.from({ length: COLS + 1 }, (_, i) =>
            React.createElement('line', {
              key: `v${i}`,
              x1: i * CELL,
              y1: 0,
              x2: i * CELL,
              y2: MAP_H,
              stroke: '#050f1a',
              strokeWidth: 0.5
            })
          ),
          Array.from({ length: ROWS + 1 }, (_, i) =>
            React.createElement('line', {
              key: `h${i}`,
              x1: 0,
              y1: i * CELL,
              x2: MAP_W,
              y2: i * CELL,
              stroke: '#050f1a',
              strokeWidth: 0.5
            })
          ),

          // Safe zone fill
          React.createElement('rect', {
            x: SZ.x1 * CELL,
            y: SZ.y1 * CELL,
            width: (SZ.x2 - SZ.x1 + 1) * CELL,
            height: (SZ.y2 - SZ.y1 + 1) * CELL,
            fill: 'rgba(0,190,110,0.055)',
            stroke: 'rgba(0,210,120,0.3)',
            strokeWidth: 1.5,
            strokeDasharray: '7,5',
            rx: 2
          }),
          React.createElement('text', {
            x: SZ.x1 * CELL + 5,
            y: SZ.y1 * CELL + 12,
            fontSize: 8,
            fill: 'rgba(0,210,120,0.35)',
            fontFamily: 'Share Tech Mono',
            letterSpacing: 1.5
          }, 'SAFE ZONE'),

          // Coordinate labels (corner nodes)
          [
            [0, 0], [COLS - 1, 0], [0, ROWS - 1], [COLS - 1, ROWS - 1],
          ].map(([x, y]) =>
            React.createElement('text', {
              key: `lbl${x},${y}`,
              x: x * CELL + CELL / 2,
              y: y * CELL + CELL / 2 + (y === 0 ? -5 : 14),
              fontSize: 7,
              fill: '#0a2035',
              textAnchor: 'middle',
              fontFamily: 'Share Tech Mono'
            }, `${x},${y}`)
          ),

          // Static grid nodes
          gridDots,

          // Dijkstra return paths
          sim.subjects.map((sub) => {
            const path = sim.paths[sub.id];
            if (!path || path.length < 2) return null;

            const pts = path.map((n) => {
              const { x, y } = nxy(n);
              return `${x},${y}`;
            }).join(" ");

            const target = path[path.length - 1];
            const { x: tx, y: ty } = nxy(target);

            return React.createElement('g', { key: `path-${sub.id}` },
              React.createElement('polyline', {
                points: pts,
                fill: 'none',
                stroke: sub.color,
                strokeWidth: 2,
                strokeOpacity: 0.55,
                strokeDasharray: '5,4',
                strokeLinecap: 'round',
                strokeLinejoin: 'round'
              }),

              path.slice(1, -1).map((n) => {
                const { x, y } = nxy(n);
                return React.createElement('circle', {
                  key: `wp-${n}`,
                  cx: x,
                  cy: y,
                  r: 2.5,
                  fill: sub.color,
                  fillOpacity: 0.45
                });
              }),

              React.createElement('circle', {
                cx: tx,
                cy: ty,
                r: 9,
                fill: 'none',
                stroke: sub.color,
                strokeWidth: 1,
                strokeOpacity: 0.3
              }),
              React.createElement('circle', {
                cx: tx,
                cy: ty,
                r: 5,
                fill: sub.color,
                fillOpacity: 0.6
              }),
              React.createElement('text', {
                x: tx,
                y: ty + 4,
                textAnchor: 'middle',
                fontSize: 7,
                fill: '#050e18',
                fontFamily: 'Share Tech Mono',
                fontWeight: 'bold'
              }, 'T')
            );
          }),

          // Subject movement trails
          sim.subjects.map((sub) =>
            sub.trail.slice(0, -1).map((pos, i) => {
              const { x, y } = nxy(pos);
              const opacity = ((i + 1) / sub.trail.length) * 0.35;
              return React.createElement('circle', {
                key: `tr-${sub.id}-${i}`,
                cx: x,
                cy: y,
                r: 2,
                fill: sub.color,
                fillOpacity: opacity
              });
            })
          ),

          // Subject markers
          sim.subjects.map((sub) => {
            const { x, y } = nxy(sub.pos);
            return React.createElement('g', { key: `sub-${sub.id}` },
              sub.isBreached
                ? React.createElement(React.Fragment, null,
                    React.createElement('circle', {
                      cx: x,
                      cy: y,
                      r: 12,
                      fill: 'none',
                      stroke: '#ff4444',
                      strokeWidth: 1,
                      strokeOpacity: 0.4,
                      filter: 'url(#glow-red)'
                    },
                      React.createElement('animate', {
                        attributeName: 'r',
                        values: '10;20;10',
                        dur: '1.1s',
                        repeatCount: 'indefinite'
                      }),
                      React.createElement('animate', {
                        attributeName: 'stroke-opacity',
                        values: '0.5;0;0.5',
                        dur: '1.1s',
                        repeatCount: 'indefinite'
                      })
                    ),
                    React.createElement('circle', {
                      cx: x,
                      cy: y,
                      r: 8,
                      fill: 'rgba(255,50,50,0.18)',
                      stroke: '#ff4444',
                      strokeWidth: 1.5,
                      filter: 'url(#glow-red)'
                    })
                  )
                : React.createElement('circle', {
                    cx: x,
                    cy: y,
                    r: 8,
                    fill: `${sub.color}18`,
                    stroke: sub.color,
                    strokeWidth: 1.5,
                    filter: 'url(#glow-teal)'
                  }),

              React.createElement('text', {
                x: x,
                y: y + 3.5,
                textAnchor: 'middle',
                fontSize: 8,
                fontWeight: 'bold',
                fill: sub.isBreached ? '#ff6060' : sub.color,
                fontFamily: 'Share Tech Mono'
              }, sub.short),

              React.createElement('text', {
                x: x,
                y: y - 13,
                textAnchor: 'middle',
                fontSize: 7,
                fill: sub.isBreached ? '#ff4444' : sub.color,
                fillOpacity: 0.7,
                fontFamily: 'Share Tech Mono'
              }, sub.label)
            );
          }),

          // Tick counter HUD
          React.createElement('text', {
            x: MAP_W - 8,
            y: MAP_H - 8,
            textAnchor: 'end',
            fontSize: 8,
            fill: '#0a1e30',
            fontFamily: 'Share Tech Mono'
          }, `T:${sim.stats.ticks.toString().padStart(4, "0")}`)
        ),

        // Legend
        React.createElement('div', { className: 'legend' },
          React.createElement('div', { className: 'leg-item' },
            React.createElement('div', { className: 'leg-box', style: { background: 'rgba(0,190,110,0.15)', border: '1px dashed rgba(0,210,120,0.4)' } }),
            'Safe Zone'
          ),
          React.createElement('div', { className: 'leg-item' },
            React.createElement('div', { className: 'leg-box', style: { background: '#f59e0b' } }),
            'Liam (Child)'
          ),
          React.createElement('div', { className: 'leg-item' },
            React.createElement('div', { className: 'leg-box', style: { background: '#22d3ee' } }),
            'Grace (Elderly)'
          ),
          React.createElement('div', { className: 'leg-item' },
            React.createElement('div', { className: 'leg-box', style: { background: 'transparent', border: '1px solid #ff4444' } }),
            'Breach Zone'
          ),
          React.createElement('div', { className: 'leg-item' },
            React.createElement('div', { className: 'leg-line' }),
            'Dijkstra Return Path'
          ),
          React.createElement('div', { className: 'leg-item', style: { color: '#1a3a55' } },
            React.createElement('span', { style: { color: '#0a6050' } }, '●'),
            ' Trail'
          )
        )
      ),

      // SIDE PANEL
      React.createElement('div', { className: 'side' },
        // System Metrics
        React.createElement('div', { className: 'blk' },
          React.createElement('div', { className: 'blk-title' }, '// System Metrics'),
          React.createElement('div', { className: 'stats-grid' },
            React.createElement('div', { className: 'sbox' },
              React.createElement('div', { className: 'sval' }, sim.stats.ticks),
              React.createElement('div', { className: 'slbl' }, 'GPS Ticks')
            ),
            React.createElement('div', { className: 'sbox' },
              React.createElement('div', { className: `sval ${sim.stats.breaches > 0 ? 'danger' : ''}` },
                sim.stats.breaches
              ),
              React.createElement('div', { className: 'slbl' }, 'Breaches')
            ),
            React.createElement('div', { className: 'sbox wide' },
              React.createElement('div', { className: 'sval ms' },
                sim.stats.avgMs,
                React.createElement('span', { style: { fontSize: 10, color: '#1a4a60' } }, ' ms')
              ),
              React.createElement('div', { className: 'slbl' }, 'Avg. Dijkstra Compute Time')
            )
          )
        ),

        // Subjects
        React.createElement('div', { className: 'blk' },
          React.createElement('div', { className: 'blk-title' }, '// Tracked Subjects'),
          sim.subjects.map((sub) => {
            const path = sim.paths[sub.id];
            return React.createElement('div', { key: sub.id, className: `subj-card ${sub.isBreached ? 'brk' : ''}` },
              React.createElement('div', { className: 'subj-row' },
                React.createElement('div', { className: 'sdot', style: { background: sub.isBreached ? '#ff4444' : sub.color } }),
                React.createElement('div', { className: 'sname' }, `${sub.label} · ${sub.type}`),
                React.createElement('div', { className: `sbadge ${sub.isBreached ? 'brk' : 'safe'}` },
                  sub.isBreached ? 'BREACH' : 'SAFE'
                )
              ),
              React.createElement('div', { className: 'sdata' },
                'NODE: ',
                React.createElement('em', null, sub.pos),
                sub.isBreached && path
                  ? [
                      React.createElement(React.Fragment, { key: 'br1' }, React.createElement('br', null), 'PATH NODES: ', React.createElement('em', null, path.length)),
                      React.createElement(React.Fragment, { key: 'br2' }, React.createElement('br', null), 'RETURN DIST: ', React.createElement('em', null, `~${(path.length * CELL).toFixed(0)}m equiv.`)),
                      React.createElement(React.Fragment, { key: 'br3' }, React.createElement('br', null), 'TARGET NODE: ', React.createElement('em', null, path[path.length - 1] || '—'))
                    ]
                  : React.createElement(React.Fragment, null, React.createElement('br', null), 'STATUS: ', React.createElement('em', null, 'Within safe zone'))
              )
            );
          })
        ),

        // Speed selector
        React.createElement('div', { className: 'blk', style: { display: 'flex', alignItems: 'center', gap: 8, padding: '8px 13px' } },
          React.createElement('div', { style: { fontSize: 9, color: '#1a3a55', letterSpacing: 1, flex: 1 } }, 'TICK SPEED'),
          React.createElement('select', {
            className: 'spd-select',
            value: speed,
            onChange: (e) => setSpeed(Number(e.target.value))
          },
            React.createElement('option', { value: 2000 }, '0.5× Slow'),
            React.createElement('option', { value: 1200 }, '1× Normal'),
            React.createElement('option', { value: 700 }, '2× Fast'),
            React.createElement('option', { value: 350 }, '4× Turbo')
          )
        ),

        // Alert Log
        React.createElement('div', { className: 'blk-title', style: { padding: '9px 13px 0', flexShrink: 0 } }, '// Alert Log'),
        React.createElement('div', { className: 'alerts-scroll' },
          sim.alerts.length === 0
            ? React.createElement('div', { className: 'no-alerts' }, '// NO BREACH EVENTS')
            : sim.alerts.map((a) =>
                React.createElement('div', { key: a.id, className: 'alert-item', style: { borderColor: a.color } },
                  React.createElement('div', { className: 'al-hdr' },
                    React.createElement('div', { className: 'al-name', style: { color: a.color } }, a.label),
                    React.createElement('div', { className: 'al-time' }, a.time)
                  ),
                  React.createElement('div', { className: 'al-body' },
                    'POS: ',
                    React.createElement('em', null, a.pos),
                    ' · STEPS: ',
                    React.createElement('em', null, a.steps),
                    React.createElement('br', null),
                    'DIST: ',
                    React.createElement('em', null, a.dist),
                    ' · COMPUTE: ',
                    React.createElement('em', null, `${a.ms}ms`)
                  )
                )
              )
        ),

        // Controls
        React.createElement('div', { className: 'ctrl' },
          React.createElement('button', {
            className: `btn btn-go ${running ? 'live' : ''}`,
            onClick: () => setRunning(true),
            disabled: running
          }, running ? 'LIVE' : 'START'),
          React.createElement('button', {
            className: 'btn btn-pz',
            onClick: () => setRunning(false),
            disabled: !running
          }, 'PAUSE'),
          React.createElement('button', {
            className: 'btn btn-rst',
            onClick: () => { setRunning(false); setSim(mkInitState()); }
          }, 'RESET')
        )
      )
    )
  );
}

// Mount the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(TrackingSystem));
