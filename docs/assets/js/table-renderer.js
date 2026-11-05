/* Module 3: KPI & Table DOM Renderer (Single Responsibility: HTML State & Error Feedback) */
(function (global) {
  'use strict';

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function showErrorBanner(message) {
    const banner = document.getElementById('app-error-banner');
    if (!banner) return;
    banner.innerHTML = '<strong>Falha de Inicialização:</strong> ' + escapeHtml(message);
    banner.classList.add('visible');
  }

  function clearErrorBanner() {
    const banner = document.getElementById('app-error-banner');
    if (!banner) return;
    banner.textContent = '';
    banner.classList.remove('visible');
  }

  function renderSimulationSummary(ranked, totalNhatsCount) {
    const countEl = document.getElementById('kpi-cand-count');
    const retEl = document.getElementById('kpi-retention');
    const top1NameEl = document.getElementById('kpi-top1-name');
    const top1MetaEl = document.getElementById('kpi-top1-meta');
    const titleEl = document.getElementById('sim-scatter-title');
    const tbody = document.getElementById('tbl-sim-body');

    const safeTotal = totalNhatsCount || 1;
    const pct = ((100.0 * ranked.length) / safeTotal).toFixed(2);

    if (countEl) countEl.textContent = ranked.length.toLocaleString('pt-BR');
    if (retEl) {
      retEl.textContent = pct + '% dos ' + totalNhatsCount.toLocaleString('pt-BR') + ' alvos NHATS';
    }

    if (ranked.length === 0) {
      if (top1NameEl) top1NameEl.textContent = 'Nenhum candidato';
      if (top1MetaEl) top1MetaEl.textContent = 'Relaxe os filtros selecionados ao lado';
      if (titleEl) {
        titleEl.textContent = 'Espaço de Decisão Astrodinâmico — 0 Alvos Qualificados';
      }
      if (tbody) {
        tbody.innerHTML =
          '<tr><td colspan="10" class="txt-col" style="padding:16px;color:#cbd5e1;">' +
          'Nenhum asteroide real da base NASA JPL NHATS atende simultaneamente às restrições atuais. ' +
          'Amplie o Δv máximo, a incerteza OCC ou reduza o diâmetro mínimo.' +
          '</td></tr>';
      }
      return;
    }

    const top1 = ranked[0];
    if (top1NameEl) top1NameEl.textContent = top1.name;
    if (top1MetaEl) {
      top1MetaEl.textContent =
        'Δv ' + top1.dv.toFixed(3) + ' km/s • TOPSIS ' + top1.topsis.toFixed(1) + ' • Espectro: ' + top1.spec;
    }

    const plotCount = Math.min(ranked.length, 300);
    if (titleEl) {
      titleEl.textContent =
        'Espaço de Decisão Astrodinâmico — Top ' + plotCount + ' Alvos Qualificados (NASA JPL NHATS)';
    }

    if (tbody) {
      tbody.innerHTML = ranked.slice(0, 20).map(function (r, idx) {
        return (
          '<tr>' +
          '<td class="txt-col"><strong>#' + (idx + 1) + '</strong></td>' +
          '<td class="txt-col">' + escapeHtml(r.pdes) + '</td>' +
          '<td class="txt-col">' + escapeHtml(r.name) + '</td>' +
          '<td class="txt-col">' + escapeHtml(r.spec) + '</td>' +
          '<td class="num-col" style="color:#34D399;font-weight:700;">' + r.dv.toFixed(3) + '</td>' +
          '<td class="num-col">' + r.dur + '</td>' +
          '<td class="num-col">' + r.diam.toFixed(1) + '</td>' +
          '<td class="num-col">' + r.obs.toLocaleString('pt-BR') + '</td>' +
          '<td class="num-col">' + r.occ + '</td>' +
          '<td class="num-col" style="color:#1CA39E;font-weight:800;">' + r.topsis.toFixed(2) + '</td>' +
          '</tr>'
        );
      }).join('');
    }
  }

  global.NewSpaceTableRenderer = {
    showErrorBanner: showErrorBanner,
    clearErrorBanner: clearErrorBanner,
    renderSimulationSummary: renderSimulationSummary
  };
})(typeof window !== 'undefined' ? window : globalThis);
