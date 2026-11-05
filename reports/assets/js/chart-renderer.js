/* Module 4: Plotly Chart Renderer & Responsive Viewport Engine */
(function (global) {
  'use strict';

  const PLOT_CONFIG = { responsive: true, displaylogo: false };

  function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  function stripCardDuplicateTitle(layout) {
    const copy = deepClone(layout || {});
    copy.title = { text: '' };
    return copy;
  }

  function showPlotFallback(containerId, message) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '<div class="plot-empty-state">' + message + '</div>';
  }

  function renderStaticPlots(figures) {
    const mapping = [
      ['plot-eng-lat', figures.engLat],
      ['plot-eng-sto', figures.engSto],
      ['plot-ml-clf', figures.mlClf],
      ['plot-ml-imp', figures.mlImp],
      ['plot-trends', figures.trends],
      ['plot-obs-law', figures.obsLaw],
      ['plot-corridors-tr', figures.corridors],
      ['plot-corridors-act', figures.corridors],
      ['plot-actuarial-mat', figures.actuarialMat]
    ];

    if (!global.NewSpaceDataLoader.isPlotlyReady()) {
      mapping.forEach(function (pair) {
        showPlotFallback(pair[0], 'Biblioteca gráfica Plotly.js não pôde ser carregada.');
      });
      return;
    }

    mapping.forEach(function (pair) {
      const id = pair[0];
      const spec = pair[1];
      const el = document.getElementById(id);
      if (!el) return;
      if (!spec || !Array.isArray(spec.data)) {
        showPlotFallback(id, 'Dados do gráfico indisponíveis no payload auditado.');
        return;
      }
      try {
        global.Plotly.newPlot(id, deepClone(spec.data), stripCardDuplicateTitle(spec.layout), PLOT_CONFIG);
      } catch (err) {
        showPlotFallback(id, 'Erro ao desenhar gráfico: ' + err.message);
      }
    });
  }

  function renderSimScatter(ranked) {
    const el = document.getElementById('plot-sim-scatter');
    if (!el) return;

    if (!global.NewSpaceDataLoader.isPlotlyReady()) {
      showPlotFallback('plot-sim-scatter', 'Biblioteca gráfica Plotly.js indisponível.');
      return;
    }

    if (!ranked || ranked.length === 0) {
      global.Plotly.purge(el);
      showPlotFallback(
        'plot-sim-scatter',
        'Nenhum alvo qualificado para os filtros atuais. Ajuste os controles para visualizar o mapa astrodinâmico.'
      );
      return;
    }

    const vw = global.innerWidth || document.documentElement.clientWidth || 1280;
    const isMobile = vw <= 640;
    const isTablet = vw > 640 && vw <= 1024;
    const topPlot = ranked.slice(0, 300);
    const scores = topPlot.map(function (r) { return r.topsis; });
    const minScore = Math.min.apply(null, scores);
    const maxScore = Math.max.apply(null, scores);
    const equalScores = Math.abs(maxScore - minScore) < 1e-6;

    const trace = {
      x: topPlot.map(function (r) { return r.dv; }),
      y: topPlot.map(function (r) { return r.dur; }),
      mode: 'markers',
      type: 'scatter',
      text: topPlot.map(function (r, idx) {
        return (
          '<b>#' + (idx + 1) + ' ' + r.name + '</b><br>' +
          'Espectro: ' + r.spec + '<br>' +
          'Diâmetro Ref.: ' + r.diam + ' m<br>' +
          'Observações: ' + r.obs + ' (OCC ' + r.occ + ')<br>' +
          'TOPSIS: ' + r.topsis.toFixed(2)
        );
      }),
      hovertemplate: '%{text}<br>Δv: %{x:.3f} km/s | Duração: %{y} d<extra></extra>',
      marker: {
        size: topPlot.map(function (r) {
          return Math.min(isMobile ? 22 : 30, Math.max(7, Math.log1p(r.diam) * (isMobile ? 3.2 : 4.0)));
        }),
        color: scores,
        cmin: equalScores ? 0 : minScore,
        cmax: equalScores ? 100 : maxScore,
        colorscale: [[0, '#23145B'], [0.35, '#09456C'], [0.7, '#026F6E'], [1, '#1CA39E']],
        colorbar: {
          title: { text: isMobile ? 'TOPSIS' : 'Score<br>TOPSIS', font: { size: isMobile ? 10 : 11.5 } },
          thickness: isMobile ? 10 : 16,
          len: isMobile ? 0.80 : 0.90,
          tickfont: { size: isMobile ? 9.5 : 11 }
        },
        line: { color: 'rgba(255,255,255,0.6)', width: 1 }
      }
    };

    const layout = {
      title: { text: '' },
      height: isMobile ? 355 : (isTablet ? 420 : 460),
      paper_bgcolor: '#1D1226',
      plot_bgcolor: '#160D18',
      font: {
        color: '#F1F5F9',
        family: 'Segoe UI, sans-serif',
        size: isMobile ? 10.5 : (isTablet ? 11.2 : 12)
      },
      margin: { l: isMobile ? 44 : 55, r: isMobile ? 12 : 24, t: 18, b: 46 },
      xaxis: {
        title: { text: 'Δv Mínimo NASA NHATS (km/s)' },
        gridcolor: '#2A2438',
        automargin: true,
        autorange: true
      },
      yaxis: {
        title: { text: 'Duração da Missão (Dias)' },
        gridcolor: '#2A2438',
        automargin: true,
        autorange: true
      }
    };

    const hasFallback = String(el.innerHTML || '').indexOf('plot-empty-state') !== -1;
    const missingPlotDom = typeof el.querySelector === 'function'
      ? !el.querySelector('.plot-container')
      : !el._fullLayout;

    if (hasFallback || !el._fullLayout || missingPlotDom) {
      global.Plotly.purge(el);
      el.innerHTML = '';
      global.Plotly.newPlot(el, [trace], layout, PLOT_CONFIG);
    } else {
      global.Plotly.react(el, [trace], layout, PLOT_CONFIG);
    }
  }

  function adaptPlotsForViewport(figures, onlyContainerId) {
    if (!global.NewSpaceDataLoader.isPlotlyReady() || !figures) return;

    const vw = global.innerWidth || document.documentElement.clientWidth || 1280;
    const isMobile = vw <= 640;
    const isTablet = vw > 640 && vw <= 1024;
    const baseFont = isMobile ? 10.5 : (isTablet ? 11.2 : 12);
    const stdHeight = isMobile ? 345 : (isTablet ? 385 : 415);

    const extractMax = global.NewSpaceDataLoader.extractNumericMax;
    const maxEngLat = extractMax(figures.engLat && figures.engLat.data[0] && figures.engLat.data[0].x, 45);
    const maxEngSto = extractMax(figures.engSto && figures.engSto.data[0] && figures.engSto.data[0].x, 25);
    const maxMlImp = extractMax(figures.mlImp && figures.mlImp.data[0] && figures.mlImp.data[0].x, 20);

    const legendTop = {
      orientation: 'h',
      x: 0,
      xanchor: 'left',
      y: 1.03,
      yanchor: 'bottom',
      font: { size: isMobile ? 10 : 11.2 }
    };

    const updates = {
      'plot-eng-lat': {
        height: stdHeight,
        'font.size': baseFont,
        margin: { l: 14, r: isMobile ? 16 : 24, t: 16, b: 42 },
        'xaxis.range': [0, maxEngLat * (isMobile ? 1.62 : (isTablet ? 1.42 : 1.32))]
      },
      'plot-eng-sto': {
        height: stdHeight,
        'font.size': baseFont,
        margin: { l: 14, r: isMobile ? 16 : 24, t: 16, b: 42 },
        'xaxis.range': [0, maxEngSto * (isMobile ? 1.65 : (isTablet ? 1.44 : 1.34))]
      },
      'plot-ml-clf': {
        height: isMobile ? 365 : stdHeight,
        'font.size': baseFont,
        margin: { l: 18, r: 16, t: isMobile ? 54 : 42, b: isMobile ? 64 : 44 },
        'xaxis.tickangle': isMobile ? -32 : (isTablet ? -18 : 0),
        legend: legendTop
      },
      'plot-ml-imp': {
        height: isMobile ? 375 : stdHeight,
        'font.size': baseFont,
        margin: { l: 14, r: isMobile ? 16 : 24, t: 16, b: 42 },
        'xaxis.range': [0, maxMlImp * (isMobile ? 1.48 : (isTablet ? 1.32 : 1.25))]
      },
      'plot-obs-law': {
        height: stdHeight,
        'font.size': baseFont,
        margin: { l: 18, r: 16, t: 18, b: isMobile ? 52 : 42 },
        'xaxis.tickangle': isMobile ? -35 : 0
      },
      'plot-corridors-tr': {
        height: isMobile ? 370 : stdHeight,
        'font.size': baseFont,
        margin: { l: 18, r: 16, t: isMobile ? 56 : 42, b: isMobile ? 68 : 48 },
        'xaxis.tickangle': isMobile ? -35 : -18,
        legend: legendTop
      },
      'plot-corridors-act': {
        height: isMobile ? 370 : stdHeight,
        'font.size': baseFont,
        margin: { l: 18, r: 16, t: isMobile ? 56 : 42, b: isMobile ? 68 : 48 },
        'xaxis.tickangle': isMobile ? -35 : -18,
        legend: legendTop
      },
      'plot-actuarial-mat': {
        height: isMobile ? 370 : stdHeight,
        'font.size': baseFont,
        margin: { l: 18, r: 16, t: isMobile ? 58 : 44, b: 46 },
        legend: legendTop
      },
      'plot-sim-scatter': {
        height: isMobile ? 355 : (isTablet ? 420 : 460),
        'font.size': baseFont,
        margin: { l: isMobile ? 44 : 55, r: isMobile ? 12 : 24, t: 18, b: 46 }
      }
    };

    const origAnn = (figures.trends && figures.trends.layout && figures.trends.layout.annotations) || [];
    const trAnn = origAnn.map(function (a, idx) {
      const copy = deepClone(a);
      if (idx === 0) {
        copy.text = isMobile
          ? '1. Produção Anual de Telemetria & Objetos em Órbita<br>(NORAD SATCAT, 1957–2025)'
          : a.text;
        copy.font = Object.assign({}, copy.font || {}, { size: isMobile ? 10.8 : 13 });
      } else if (idx === 1) {
        copy.text = isMobile
          ? '2. Descobertas Anuais de Asteroides Próximos à Terra<br>(NASA JPL SBDB)'
          : a.text;
        copy.font = Object.assign({}, copy.font || {}, { size: isMobile ? 10.8 : 13 });
      } else if (idx === 2) {
        copy.text = isMobile
          ? '3. Índice Composto de Produção de Informações (Base 2010=100)<br>+ Projeção 2030 (IC 95%)'
          : a.text;
        copy.font = Object.assign({}, copy.font || {}, { size: isMobile ? 10.8 : 13 });
      } else {
        copy.font = Object.assign({}, copy.font || {}, { size: isMobile ? 9.5 : 11 });
        copy.ax = isMobile ? -55 : -90;
      }
      return copy;
    });

    updates['plot-trends'] = {
      height: isMobile ? 740 : (isTablet ? 750 : 770),
      'font.size': baseFont,
      annotations: trAnn,
      margin: isMobile
        ? { l: 34, r: 14, t: 42, b: 122 }
        : { l: 42, r: 24, t: 68, b: 46 },
      legend: isMobile
        ? { orientation: 'h', x: 0, xanchor: 'left', y: -0.07, yanchor: 'top', font: { size: 10 } }
        : { orientation: 'h', x: 0, xanchor: 'left', y: 1.04, yanchor: 'bottom', font: { size: 11.2 } }
    };

    const root = onlyContainerId ? document.getElementById(onlyContainerId) : document;
    if (!root) return;

    Object.keys(updates).forEach(function (plotId) {
      const el = document.getElementById(plotId);
      if (!el || !root.contains(el) || !el.data) return;
      if (el.offsetParent === null) return;
      global.Plotly.relayout(el, updates[plotId]);
      global.Plotly.Plots.resize(el);
    });
  }

  global.NewSpaceChartRenderer = {
    renderStaticPlots: renderStaticPlots,
    renderSimScatter: renderSimScatter,
    adaptPlotsForViewport: adaptPlotsForViewport
  };
})(typeof window !== 'undefined' ? window : globalThis);
