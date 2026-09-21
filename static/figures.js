/*
 * figures.js: turns ```plot and ```svg blocks in Claude's answers into real figures.
 *   ```plot  -> JSON spec drawn with Plotly (expressions evaluated with math.js)
 *   ```svg   -> sanitised inline SVG diagram
 * Requires: Plotly, math (math.js), DOMPurify, loaded before this file.
 */
(function () {
  // ---------------------------------------------------- R-style functions ---
  const SQ2PI = Math.sqrt(2 * Math.PI);
  function lgamma(z) {                       // Lanczos approximation
    if (z < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * z))) - lgamma(1 - z);
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    z -= 1; let a = c[0]; const t = z + g + 0.5;
    for (let i = 1; i < 9; i++) a += c[i] / (z + i);
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
  }
  function erf(x) {                          // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const lchoose = (n, k) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  const isInt = k => Math.abs(k - Math.round(k)) < 1e-9;
  function integrate(f, a, b, n = 400) {     // Simpson's rule
    if (b <= a) return 0; const h = (b - a) / n; let s = f(a) + f(b);
    for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * f(a + i * h);
    return s * h / 3;
  }
  function randn() { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }

  const F = {
    dnorm: (x, m = 0, s = 1) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQ2PI),
    pnorm: (x, m = 0, s = 1) => 0.5 * (1 + erf((x - m) / (s * Math.SQRT2))),
    qnorm: (p, m = 0, s = 1) => {           // bisection on pnorm
      if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
      let lo = -40, hi = 40;
      for (let i = 0; i < 100; i++) { const mid = (lo + hi) / 2; (F.pnorm(mid) < p) ? lo = mid : hi = mid; }
      return m + s * (lo + hi) / 2;
    },
    dt: (x, d) => Math.exp(lgamma((d + 1) / 2) - lgamma(d / 2)) / Math.sqrt(d * Math.PI) * (1 + x * x / d) ** (-(d + 1) / 2),
    pt: (x, d) => x >= 0 ? 0.5 + integrate(t => F.dt(t, d), 0, x) : 0.5 - integrate(t => F.dt(t, d), 0, -x),
    dchisq: (x, k) => x < 0 ? 0 : (x === 0 ? (k === 2 ? 0.5 : (k < 2 ? Infinity : 0)) :
      Math.exp((k / 2 - 1) * Math.log(x) - x / 2 - (k / 2) * Math.LN2 - lgamma(k / 2))),
    df: (x, d1, d2) => x <= 0 ? 0 : Math.exp(0.5 * (d1 * Math.log(d1 * x) + d2 * Math.log(d2) - (d1 + d2) * Math.log(d1 * x + d2))
      - Math.log(x) - (lgamma(d1 / 2) + lgamma(d2 / 2) - lgamma((d1 + d2) / 2))),
    dbinom: (k, n, p) => (!isInt(k) || k < 0 || k > n) ? 0 : Math.exp(lchoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p)),
    pbinom: (k, n, p) => { let s = 0; for (let i = 0; i <= Math.floor(k); i++) s += F.dbinom(i, n, p); return Math.min(1, s); },
    dpois: (k, l) => (!isInt(k) || k < 0) ? 0 : Math.exp(k * Math.log(l) - l - lgamma(k + 1)),
    dunif: (x, a = 0, b = 1) => (x >= a && x <= b) ? 1 / (b - a) : 0,
    dexp: (x, r = 1) => x < 0 ? 0 : r * Math.exp(-r * x),
    choose: (n, k) => Math.round(Math.exp(lchoose(n, k))),
    rnorm: (n, m = 0, s = 1) => Array.from({ length: n }, () => m + s * randn()),
    runif: (n, a = 0, b = 1) => Array.from({ length: n }, () => a + (b - a) * Math.random()),
    rexp: (n, r = 1) => Array.from({ length: n }, () => -Math.log(1 - Math.random()) / r),
    rbinom: (n, size, p) => Array.from({ length: n }, () => { let c = 0; for (let i = 0; i < size; i++) c += Math.random() < p; return c; }),
    rpois: (n, l) => Array.from({ length: n }, () => { let k = 0, t = Math.exp(-l), s = t, u = Math.random(); while (u > s && k < 1000) { k++; t *= l / k; s += t; } return k; }),
  };
  // raw functions: math.js passes plain numbers through unchanged
  const mathjs = math.create(math.all);
  mathjs.import(F, { override: true });

  const toArr = v => (v && v.toArray) ? v.toArray() : (Array.isArray(v) ? v : [v]);
  const compile = expr => { const c = mathjs.compile(String(expr)); return x => { const y = Number(c.evaluate({ x })); return isFinite(y) ? y : null; }; };
  const sample = expr => toArr(mathjs.evaluate(String(expr))).flat().map(Number);
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));

  // ------------------------------------------------------------- plotting ---
  const PALETTE = ['#4f46e5', '#e8590c', '#0ca678', '#d6336c', '#1c7ed6', '#f59f00'];

  function renderPlot(spec, el) {
    const css = getComputedStyle(document.documentElement);
    const text = css.getPropertyValue('--text').trim() || '#222';
    const grid = css.getPropertyValue('--border').trim() || '#ddd';
    const traces = [], shapes = [], annotations = [];
    let ci = 0; const nextColor = L => L.color || PALETTE[ci++ % PALETTE.length];

    for (const L of spec.layers || []) {
      const name = L.label, showlegend = !!L.label;
      switch (L.type) {
        case 'function':
        case 'shade': {
          const f = compile(L.expr), xs = linspace(+L.from, +L.to, L.n || 400), ys = xs.map(f);
          const color = nextColor(L);
          traces.push(L.type === 'function'
            ? { x: xs, y: ys, mode: 'lines', name, showlegend, line: { color, width: 2.5, dash: L.dash } }
            : { x: xs, y: ys, mode: 'lines', name, showlegend, fill: 'tozeroy', line: { color, width: 0 }, fillcolor: /^#[0-9a-f]{6}$/i.test(color) ? color + '55' : color, opacity: /^#[0-9a-f]{6}$/i.test(color) ? 1 : 0.4 });
          break;
        }
        case 'bars': {
          let x = L.x, y = L.y;
          if (L.expr) { const f = compile(L.expr); x = []; for (let k = Math.ceil(+L.from); k <= +L.to; k++) x.push(k); y = x.map(f); }
          traces.push({ type: 'bar', x, y, name, showlegend, marker: { color: nextColor(L) } });
          break;
        }
        case 'line':
        case 'points':
          traces.push({ x: L.x, y: L.y, mode: L.type === 'line' ? 'lines+markers' : 'markers', name, showlegend,
            marker: { color: nextColor(L), size: 7 }, line: { width: 2 } });
          break;
        case 'histogram': {
          const data = L.data || sample(L.sample);
          traces.push({ type: 'histogram', x: data, name, showlegend, nbinsx: L.bins,
            histnorm: L.density ? 'probability density' : '', marker: { color: nextColor(L), line: { color: 'white', width: 1 } }, opacity: 0.85 });
          break;
        }
        case 'boxplot': {
          const data = L.data || sample(L.sample);
          traces.push({ type: 'box', y: data, name: L.label || ' ', showlegend: false, marker: { color: nextColor(L) }, boxmean: true });
          break;
        }
        case 'vline':
        case 'hline': {
          const v = L.type === 'vline';
          shapes.push(v ? { type: 'line', x0: L.x, x1: L.x, yref: 'paper', y0: 0, y1: 1, line: { color: L.color || text, dash: 'dash', width: 1.5 } }
                        : { type: 'line', y0: L.y, y1: L.y, xref: 'paper', x0: 0, x1: 1, line: { color: L.color || text, dash: 'dash', width: 1.5 } });
          if (L.label) annotations.push(v ? { x: L.x, yref: 'paper', y: 1, text: L.label, showarrow: false, yanchor: 'bottom' }
                                          : { y: L.y, xref: 'paper', x: 1, text: L.label, showarrow: false, xanchor: 'right', yanchor: 'bottom' });
          break;
        }
        case 'text':
          annotations.push({ x: L.x, y: L.y, text: L.text, showarrow: false });
          break;
      }
    }
    const axis = t => ({ title: { text: t || '' }, gridcolor: grid, zerolinecolor: grid, color: text });
    Plotly.newPlot(el, traces, {
      title: { text: spec.title || '', font: { size: 16 } },
      xaxis: axis(spec.xlabel), yaxis: axis(spec.ylabel),
      barmode: 'overlay', bargap: 0.1, shapes, annotations,
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: text, family: 'system-ui, sans-serif' },
      margin: { t: spec.title ? 50 : 20, r: 20, b: 50, l: 60 }, height: 380,
      legend: { orientation: 'h', y: -0.2 },
    }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d'] });
  }

  // ------------------------------------------------ find & replace blocks ---
  window.renderFigures = function (container) {
    container.querySelectorAll('pre > code.language-plot, pre > code.language-svg').forEach(code => {
      const pre = code.parentElement, src = code.textContent;
      const fig = document.createElement('div'); fig.className = 'figure';
      pre.before(fig);                       // draw first; keep the source if drawing fails
      try {
        if (code.classList.contains('language-plot')) {
          renderPlot(JSON.parse(src), fig);
        } else {
          const clean = DOMPurify.sanitize(src, { USE_PROFILES: { svg: true, svgFilters: true } });
          if (!/<svg/i.test(clean)) throw new Error('empty SVG');
          fig.classList.add('diagram');
          fig.innerHTML = clean;
        }
        pre.remove();
      } catch (err) {
        console.error(err);
        fig.className = 'fig-error';
        fig.textContent = "Couldn't draw this figure (" + err.message + "). The raw figure code is below; try asking again.";
      }
    });
  };
})();
