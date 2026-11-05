/* Module 5: Interactive UI Controller & Fault-Tolerant Bootstrap */
(function (global) {
  'use strict';

  let payload = null;
  let currentRanked = [];
  let resizeTimer = null;
  let initialized = false;

  function readControlsState() {
    const def = global.NewSpaceTopsisEngine.DEFAULT_CRITERIA;
    function numVal(id, fallback) {
      const el = document.getElementById(id);
      if (!el) return fallback;
      const parsed = parseFloat(el.value);
      return Number.isFinite(parsed) ? parsed : fallback;
    }
    const grpEl = document.getElementById('ctrl-grp');
    return {
      maxDv: numVal('ctrl-dv', def.maxDv),
      maxDur: Math.round(numVal('ctrl-dur', def.maxDur)),
      maxOcc: Math.round(numVal('ctrl-occ', def.maxOcc)),
      minObs: Math.round(numVal('ctrl-obs', def.minObs)),
      minDiam: numVal('ctrl-diam', def.minDiam),
      grpFilter: grpEl ? grpEl.value : def.grpFilter,
      w1: numVal('ctrl-w1', def.w1),
      w2: numVal('ctrl-w2', def.w2),
      w3: numVal('ctrl-w3', def.w3),
      w4: numVal('ctrl-w4', def.w4)
    };
  }

  function syncControlLabels(state) {
    const map = {
      'lbl-dv': state.maxDv.toFixed(2),
      'lbl-dur': String(state.maxDur),
      'lbl-occ': String(state.maxOcc),
      'lbl-obs': String(state.minObs),
      'lbl-diam': String(state.minDiam),
      'lbl-w1': state.w1.toFixed(2),
      'lbl-w2': state.w2.toFixed(2),
      'lbl-w3': state.w3.toFixed(2),
      'lbl-w4': state.w4.toFixed(2)
    };
    Object.keys(map).forEach(function (id) {
      const el = document.getElementById(id);
      if (el) el.textContent = map[id];
    });
  }

  function recomputeSimulation() {
    if (!payload) return;
    const state = readControlsState();
    syncControlLabels(state);
    currentRanked = global.NewSpaceTopsisEngine.rankCandidatesTOPSIS(payload.nhatsRecords, state);
    global.NewSpaceTableRenderer.renderSimulationSummary(currentRanked, payload.nhatsRecords.length);
    global.NewSpaceChartRenderer.renderSimScatter(currentRanked);
  }

  function resetDefaults() {
    const def = global.NewSpaceTopsisEngine.DEFAULT_CRITERIA;
    const assignments = {
      'ctrl-dv': def.maxDv,
      'ctrl-dur': def.maxDur,
      'ctrl-occ': def.maxOcc,
      'ctrl-obs': def.minObs,
      'ctrl-diam': def.minDiam,
      'ctrl-grp': def.grpFilter,
      'ctrl-w1': def.w1,
      'ctrl-w2': def.w2,
      'ctrl-w3': def.w3,
      'ctrl-w4': def.w4
    };
    Object.keys(assignments).forEach(function (id) {
      const el = document.getElementById(id);
      if (el) el.value = assignments[id];
    });
    recomputeSimulation();
  }

  function exportFilteredCSV() {
    if (!currentRanked.length) return;
    const csvContent = global.NewSpaceTopsisEngine.buildQualifiedCSV(currentRanked);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'newspace_selecao_topsis_qualificada.csv';
    a.click();
  }

  function switchTab(tabId, btn) {
    document.querySelectorAll('.tab-panel').forEach(function (el) {
      el.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(function (el) {
      el.classList.remove('active');
    });
    const panel = document.getElementById(tabId);
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');
    if (payload) {
      const runAdapt = function () {
        global.NewSpaceChartRenderer.adaptPlotsForViewport(payload.figures, tabId);
      };
      if (typeof global.requestAnimationFrame === 'function') {
        global.requestAnimationFrame(runAdapt);
      } else {
        runAdapt();
      }
    }
  }

  function toggleMobileFilters() {
    const body = document.getElementById('ctrl-body');
    const icon = document.getElementById('filter-toggle-icon');
    const btn = document.getElementById('btn-filter-toggle');
    if (!body) return;
    const collapsed = body.classList.toggle('collapsed');
    if (btn) btn.setAttribute('aria-expanded', String(!collapsed));
    if (icon) icon.innerHTML = collapsed ? '&#9662; Mostrar' : '&#9652; Ocultar';
  }

  function wireEvents() {
    document.querySelectorAll('.tab-btn[data-tab]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        switchTab(btn.getAttribute('data-tab'), btn);
      });
    });

    const rangeIds = [
      'ctrl-dv', 'ctrl-dur', 'ctrl-occ', 'ctrl-obs', 'ctrl-diam',
      'ctrl-w1', 'ctrl-w2', 'ctrl-w3', 'ctrl-w4'
    ];
    rangeIds.forEach(function (id) {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', recomputeSimulation);
        el.addEventListener('change', recomputeSimulation);
      }
    });

    const grpEl = document.getElementById('ctrl-grp');
    if (grpEl) {
      grpEl.addEventListener('change', recomputeSimulation);
      grpEl.addEventListener('input', recomputeSimulation);
    }

    const btnExport = document.getElementById('btn-export-csv');
    if (btnExport) btnExport.addEventListener('click', exportFilteredCSV);

    const btnReset = document.getElementById('btn-reset-defaults');
    if (btnReset) btnReset.addEventListener('click', resetDefaults);

    const btnToggle = document.getElementById('btn-filter-toggle');
    if (btnToggle) btnToggle.addEventListener('click', toggleMobileFilters);

    global.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        if (!payload) return;
        global.NewSpaceChartRenderer.adaptPlotsForViewport(payload.figures);
        const simTab = document.getElementById('tab-sim');
        if (simTab && simTab.classList.contains('active')) {
          global.NewSpaceChartRenderer.renderSimScatter(currentRanked);
        }
      }, 140);
    });
  }

  function initDashboard() {
    if (initialized) return;
    initialized = true;
    try {
      global.NewSpaceTableRenderer.clearErrorBanner();
      payload = global.NewSpaceDataLoader.loadDashboardPayload();
      wireEvents();
      global.NewSpaceChartRenderer.renderStaticPlots(payload.figures);
      recomputeSimulation();
      global.NewSpaceChartRenderer.adaptPlotsForViewport(payload.figures, 'tab-sim');
    } catch (err) {
      global.NewSpaceTableRenderer.showErrorBanner(err && err.message ? err.message : String(err));
    }
  }

  global.NewSpaceUIController = {
    initDashboard: initDashboard,
    recomputeSimulation: recomputeSimulation,
    resetDefaults: resetDefaults,
    exportFilteredCSV: exportFilteredCSV,
    switchTab: switchTab,
    toggleMobileFilters: toggleMobileFilters,
    getCurrentRanked: function () { return currentRanked; }
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initDashboard);
    } else {
      initDashboard();
    }
  }
})(typeof window !== 'undefined' ? window : globalThis);
