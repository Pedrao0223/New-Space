/* Module 1: Data & Library Validation (Single Responsibility: Safe Payload Access) */
(function (global) {
  'use strict';

  function loadDashboardPayload() {
    const payload = global.NEWSPACE_DATA;
    if (!payload || typeof payload !== 'object') {
      throw new Error(
        'Arquivo de dados reais (assets/data/dashboard-data.js) não foi carregado ou está inacessível.'
      );
    }
    if (!Array.isArray(payload.nhatsRecords) || payload.nhatsRecords.length === 0) {
      throw new Error(
        'A base auditada NASA JPL NHATS está vazia ou malformada no payload do dashboard.'
      );
    }
    if (!payload.figures || typeof payload.figures !== 'object') {
      throw new Error(
        'As especificações dos gráficos analíticos (payload.figures) não foram encontradas.'
      );
    }
    return payload;
  }

  function isPlotlyReady() {
    return typeof global.Plotly !== 'undefined' && typeof global.Plotly.newPlot === 'function';
  }

  function extractNumericMax(values, fallback) {
    const safeFallback = typeof fallback === 'number' ? fallback : 1.0;
    if (!values) return safeFallback;
    if (Array.isArray(values) || ArrayBuffer.isView(values)) {
      let maxVal = -Infinity;
      for (let i = 0; i < values.length; i += 1) {
        const num = Number(values[i]);
        if (Number.isFinite(num) && num > maxVal) {
          maxVal = num;
        }
      }
      return maxVal > -Infinity ? maxVal : safeFallback;
    }
    return safeFallback;
  }

  global.NewSpaceDataLoader = {
    loadDashboardPayload: loadDashboardPayload,
    isPlotlyReady: isPlotlyReady,
    extractNumericMax: extractNumericMax
  };
})(typeof window !== 'undefined' ? window : globalThis);
