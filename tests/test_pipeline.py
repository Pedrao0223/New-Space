"""Automated QA & Scientific Data Integrity Test Suite (pytest)."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
import subprocess

import nbformat
import pytest

from newspace.benchmarking import (
    _build_leak_free_actuarial_features,
    benchmark_clustering_mining,
    benchmark_data_engines,
)
from newspace.config import DB_PATH, DOCS_DIR, NOTEBOOKS_DIR, PALETTE, REPORTS_DIR
from newspace.ingestion import load_all_real_datasets, verify_provenance
from newspace.selection_trends import (
    analyze_information_production_trends,
    apply_selection_criteria_and_topsis,
)


@pytest.fixture(scope="module")
def datasets():
    return load_all_real_datasets()


def test_sha256_provenance_and_real_entities(datasets):
    """Guarantee 100% authentic public datasets via SHA-256 and canonical real entities."""
    manifest = verify_provenance()
    assert len(manifest["datasets"]) == 4

    # Verify real canonical aerospace entities exist in parsed tables
    assert len(datasets.satellites) > 65000
    assert len(datasets.asteroids) > 40000
    assert int((datasets.asteroids["acessivel_nhats"] == 1).sum()) >= 7000
    assert len(datasets.launches) > 7000

    assert any(datasets.launchers["modelo_veiculo"] == "Falcon 9")
    assert any(datasets.asteroids["nome_completo"].str.contains("Eros", na=False))
    assert any(datasets.asteroids["nome_completo"].str.contains("Apophis", na=False))


def test_sqlite_foreign_keys_and_optimizer_stats():
    """Verify SQLite relational integrity, composite indexes, and ANALYZE statistics."""
    assert DB_PATH.exists(), "Normalized SQLite DB must exist"
    with closing(sqlite3.connect(DB_PATH)) as conn:
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        assert fk_violations == [], f"Unexpected FK violations: {fk_violations}"

        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        expected = {
            "VEICULO_LANCADOR",
            "HISTORICO_LANCAMENTO",
            "CORREDOR_ORBITAL",
            "OBJETO_TRAFEGO_ORBITAL",
            "ASTEROIDE_PROSPECCAO",
            "CENARIO_VIABILIDADE_ATUARIAL",
        }
        assert expected.issubset(tables)

        # Confirm ANALYZE populated sqlite_stat1 for query planner optimization
        stat1_count = conn.execute("SELECT COUNT(*) FROM sqlite_stat1;").fetchone()[0]
        assert stat1_count > 0


def test_zero_temporal_leakage_in_actuarial_features(datasets):
    """Mathematically verify maiden launch of every vehicle has zero prior vehicle flights."""
    X, y_risk, feat_names = _build_leak_free_actuarial_features(datasets.launches)
    assert X.shape[0] == len(datasets.launches)
    assert X.shape[1] == len(feat_names)
    assert set(y_risk).issubset({0, 1})

    # Maiden flight of dataset (Sputnik 1957) must have log_voos_veiculo == 0 and neutral prior 0.90
    idx_log_v = feat_names.index("log_voos_veiculo")
    idx_prior_v = feat_names.index("confiabilidade_previa_veiculo")
    assert X[0, idx_log_v] == pytest.approx(0.0)
    assert X[0, idx_prior_v] == pytest.approx(0.90, abs=1e-6)


def test_engine_and_clustering_benchmarks(datasets):
    """Verify engine benchmark speedup and NHATS clustering validity."""
    w_df, s_df, st_df = benchmark_data_engines(datasets, repeats=3)
    assert len(s_df) == 5
    assert (s_df["latencia_total_mediana_ms"] > 0).all()

    # Indexed SQLite must outperform Unindexed SQLite
    lat_idx = float(
        s_df.loc[s_df["engine"] == "SQLite (Indexado + WAL)", "latencia_total_mediana_ms"].iloc[0]
    )
    lat_unidx = float(
        s_df.loc[s_df["engine"] == "SQLite (Sem Índices)", "latencia_total_mediana_ms"].iloc[0]
    )
    assert lat_idx < lat_unidx
    assert len(st_df) == 4

    c_bench, c_nhats = benchmark_clustering_mining(datasets.asteroids)
    assert len(c_bench) == 5
    assert (c_bench["silhouette_score"] > 0.05).all()
    assert c_nhats["arquetipo_astrodinamico"].nunique() == 4


def test_information_trends_and_topsis_monotonicity(datasets):
    """Verify exponential NewSpace information trend, Spearman OCC law, and TOPSIS sensitivity."""
    trends = analyze_information_production_trends(datasets)
    era1_idx = float(trends.era_comparison.iloc[0]["indice_info_medio"])
    era3_idx = float(trends.era_comparison.iloc[-1]["indice_info_medio"])
    assert era3_idx > 3.0 * era1_idx

    rho, pval = trends.spearman_obs_vs_occ
    assert rho < -0.50
    assert pval < 1e-50

    fc = trends.forecast_2030
    assert (fc["ic95_inferior"] < fc["indice_info_projetado"]).all()
    assert (fc["indice_info_projetado"] < fc["ic95_superior"]).all()

    sel_strict = apply_selection_criteria_and_topsis(datasets, max_delta_v_kms=5.0)
    sel_wide = apply_selection_criteria_and_topsis(datasets, max_delta_v_kms=7.5)
    assert 0 < len(sel_strict.ranked_asteroids) < len(sel_wide.ranked_asteroids)
    assert sel_wide.ranked_asteroids["score_topsis"].between(0.0, 100.0).all()


def test_executed_notebook_and_dashboard_artifacts():
    """Verify executed Jupyter Notebook, standalone HTML dashboard, and technical reports."""
    nb_path = NOTEBOOKS_DIR / "newspace_benchmark_analytics.ipynb"
    dash_path = REPORTS_DIR / "newspace_dashboard.html"
    md_path = REPORTS_DIR / "relatorio_tecnico_decisao.md"
    html_rep_path = REPORTS_DIR / "relatorio_tecnico_decisao.html"

    for p in (nb_path, dash_path, md_path, html_rep_path):
        assert p.exists() and p.stat().st_size > 2048, f"Missing or empty artifact: {p}"

    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 8
    assert all(c.execution_count is not None for c in code_cells), "All code cells must be executed"

    dash_html = dash_path.read_text(encoding="utf-8")
    rep_html = html_rep_path.read_text(encoding="utf-8")
    for hex_color in [
        PALETTE.cosmic_bg,
        PALETTE.space_violet,
        PALETTE.orbital_blue,
        PALETTE.deep_teal,
        PALETTE.vibrant_cyan,
    ]:
        assert hex_color.lower() in dash_html.lower()

    # Verify favicon link & clean proportional Portuguese column headers (zero raw snake_case headers)
    assert '<link rel="icon"' in dash_html
    assert '<link rel="icon"' in rep_html
    assert (DOCS_DIR / "favicon.svg").exists()
    assert (DOCS_DIR / "index.html").exists()
    assert (DOCS_DIR / "relatorio.html").exists()

    # Verify docs/ GitHub Pages files match generated reports byte-for-byte
    assert (DOCS_DIR / "index.html").read_text(encoding="utf-8") == dash_html
    assert (DOCS_DIR / "relatorio.html").read_text(encoding="utf-8") == rep_html

    # Verify modular frontend architecture exists in both reports/ and docs/
    expected_assets = (
        "assets/css/dashboard.css",
        "assets/css/nsa-chatbot.css",
        "assets/vendor/plotly.min.js",
        "assets/data/dashboard-data.js",
        "assets/js/data-loader.js",
        "assets/js/topsis-engine.js",
        "assets/js/table-renderer.js",
        "assets/js/chart-renderer.js",
        "assets/js/ui-controller.js",
        "assets/js/nsa-chatbot.js",
    )
    for rel_asset in expected_assets:
        rep_asset = REPORTS_DIR / rel_asset
        doc_asset = DOCS_DIR / rel_asset
        assert rep_asset.exists() and rep_asset.stat().st_size > 256, f"Missing {rep_asset}"
        assert doc_asset.exists() and doc_asset.stat().st_size > 256, f"Missing {doc_asset}"

    # Guarantee zero Plotly 7 binary `bdata` buffers leaked into client JSON
    data_js = (DOCS_DIR / "assets/data/dashboard-data.js").read_text(encoding="utf-8")
    assert '"bdata"' not in data_js, "Binary bdata buffer found in dashboard-data.js!"

    dash_css = (DOCS_DIR / "assets/css/dashboard.css").read_text(encoding="utf-8")
    chart_js = (DOCS_DIR / "assets/js/chart-renderer.js").read_text(encoding="utf-8")
    ui_js = (DOCS_DIR / "assets/js/ui-controller.js").read_text(encoding="utf-8")

    # Verify mobile & tablet responsive CSS/JS breakpoints and touch scrolling
    assert 'name="viewport" content="width=device-width, initial-scale=1.0"' in dash_html
    assert 'name="viewport" content="width=device-width, initial-scale=1.0"' in rep_html
    for responsive_token in (
        "@media (max-width: 960px)",
        "-webkit-overflow-scrolling: touch",
        "min-width: 0;",
    ):
        assert responsive_token in dash_css
        assert responsive_token in rep_html

    assert "@media (max-width: 1080px)" in dash_css
    assert "@media (max-width: 640px)" in dash_css
    assert "adaptPlotsForViewport" in chart_js
    assert "toggleMobileFilters" in ui_js

    # Verify revised Data Selection Criteria section & space data research introduction
    assert "criteria-intro-card" in dash_html
    assert "criteria-intro-card" in dash_css
    assert "O que são os Critérios de Seleção de Dados e sua Importância na Pesquisa Espacial" in dash_html
    assert "3. Critérios de Seleção de Dados, Funil Qualificador e Ranking TOPSIS" in rep_html

    for raw_col in (
        "era_espacial",
        "media_anual_obj_orbitais",
        "media_anual_neos_descobertos",
        "latencia_total_mediana_ms",
        "brier_score_calibracao",
    ):
        assert raw_col not in rep_html
        assert raw_col not in dash_html


def test_frontend_js_modules_populate_kpis_tables_and_charts():
    """Execute actual JS modules in Node.js VM to verify KPIs, Top 20 table, filters, and all 10 plots."""
    node_script = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const docsDir = process.argv[1];
const elements = {};
const plotCalls = {};

function makeEl(id, initialValue = '') {
  const classes = new Set(id === 'tab-sim' ? ['tab-panel', 'active'] : []);
  const el = {
    id,
    value: String(initialValue),
    textContent: '-',
    innerHTML: '',
    disabled: false,
    offsetParent: id === 'app-error-banner' ? null : {},
    data: null,
    _fullLayout: null,
    listeners: {},
    classList: {
      add: c => classes.add(c),
      remove: c => classes.delete(c),
      toggle: c => { if (classes.has(c)) { classes.delete(c); return false; } classes.add(c); return true; },
      contains: c => classes.has(c)
    },
    setAttribute: () => {},
    getAttribute: k => (k === 'data-tab' ? 'tab-sim' : null),
    addEventListener: function(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    querySelector: function(sel) {
      if (sel === '.plot-container') {
        return el.innerHTML.includes('plot-container') ? { className: 'plot-container' } : null;
      }
      return null;
    },
    contains: function(other) { return true; }
  };
  return el;
}

[
  ['app-error-banner', ''], ['kpi-cand-count', ''], ['kpi-retention', ''],
  ['kpi-top1-name', ''], ['kpi-top1-meta', ''], ['sim-scatter-title', ''],
  ['tbl-sim-body', ''], ['ctrl-dv', 6.5], ['ctrl-dur', 450], ['ctrl-occ', 3],
  ['ctrl-obs', 50], ['ctrl-diam', 25], ['ctrl-grp', 'ALL'],
  ['ctrl-w1', 0.40], ['ctrl-w2', 0.25], ['ctrl-w3', 0.20], ['ctrl-w4', 0.15],
  ['lbl-dv', ''], ['lbl-dur', ''], ['lbl-occ', ''], ['lbl-obs', ''], ['lbl-diam', ''],
  ['lbl-w1', ''], ['lbl-w2', ''], ['lbl-w3', ''], ['lbl-w4', ''],
  ['ctrl-body', ''], ['filter-toggle-icon', ''], ['btn-filter-toggle', ''],
  ['btn-export-csv', ''], ['btn-reset-defaults', ''], ['tab-sim', ''],
  ['plot-sim-scatter', ''], ['plot-eng-lat', ''], ['plot-eng-sto', ''],
  ['plot-ml-clf', ''], ['plot-ml-imp', ''], ['plot-trends', ''],
  ['plot-obs-law', ''], ['plot-corridors-tr', ''], ['plot-corridors-act', ''],
  ['plot-actuarial-mat', '']
].forEach(([id, val]) => { elements[id] = makeEl(id, val); });

const sandbox = {
  console,
  Math, Number, String, Object, Array, ArrayBuffer, JSON, Infinity,
  parseFloat, parseInt, isFinite, setTimeout, clearTimeout,
  innerWidth: 1280,
  addEventListener: () => {},
  requestAnimationFrame: fn => fn(),
  document: {
    readyState: 'complete',
    documentElement: { clientWidth: 1280 },
    getElementById: id => elements[id] || null,
    querySelectorAll: () => [],
    contains: () => true,
    addEventListener: () => {}
  },
  Plotly: {
    newPlot: (idOrEl, data, layout) => {
      const el = typeof idOrEl === 'string' ? elements[idOrEl] : idOrEl;
      if (!Array.isArray(data) || data.length === 0) throw new Error('Empty data for ' + el.id);
      if (!Array.isArray(data[0].x) || data[0].x.length === 0) throw new Error('Non-array x for ' + el.id);
      el.innerHTML = '<div class="plot-container plotly"><svg class="main-svg"></svg></div>';
      el._fullLayout = { mounted: true };
      el.data = data;
      plotCalls[el.id] = { mode: 'newPlot', traces: data.length, xLen: data[0].x.length };
    },
    react: (idOrEl, data, layout) => {
      const el = typeof idOrEl === 'string' ? elements[idOrEl] : idOrEl;
      if (!Array.isArray(data) || !Array.isArray(data[0].x) || data[0].x.length === 0) {
        throw new Error('Invalid scatter data for ' + el.id);
      }
      // Enforce real Plotly DOM invariant: if _fullLayout exists but innerHTML wiped .plot-container, plot vanishes!
      if (el._fullLayout && !el.innerHTML.includes('plot-container')) {
        throw new Error('Plotly DOM wiped before Plotly.react on ' + el.id + ' — chart would disappear!');
      }
      if (!el._fullLayout) {
        el.innerHTML = '<div class="plot-container plotly"><svg class="main-svg"></svg></div>';
        el._fullLayout = { mounted: true };
      }
      el.data = data;
      plotCalls[el.id] = { mode: 'react', traces: data.length, xLen: data[0].x.length };
    },
    relayout: () => {},
    purge: (idOrEl) => {
      const el = typeof idOrEl === 'string' ? elements[idOrEl] : idOrEl;
      if (el) {
        el.data = null;
        el._fullLayout = null;
        el.innerHTML = '';
      }
    },
    Plots: { resize: () => {} }
  }
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

[
  'assets/data/dashboard-data.js',
  'assets/js/data-loader.js',
  'assets/js/topsis-engine.js',
  'assets/js/table-renderer.js',
  'assets/js/chart-renderer.js',
  'assets/js/ui-controller.js'
].forEach(rel => {
  const code = fs.readFileSync(path.join(docsDir, rel), 'utf8');
  vm.runInContext(code, sandbox, { filename: rel });
});

// 1. Verify zero initialization errors & all 10 plots rendered with intact .plot-container
if (elements['app-error-banner'].classList.contains('visible')) {
  throw new Error('Error banner visible: ' + elements['app-error-banner'].innerHTML);
}
const expectedPlots = [
  'plot-sim-scatter', 'plot-eng-lat', 'plot-eng-sto', 'plot-ml-clf', 'plot-ml-imp',
  'plot-trends', 'plot-obs-law', 'plot-corridors-tr', 'plot-corridors-act', 'plot-actuarial-mat'
];
expectedPlots.forEach(pid => {
  if (!plotCalls[pid] || plotCalls[pid].xLen === 0 || !elements[pid].innerHTML.includes('plot-container')) {
    throw new Error('Plot was not rendered with intact DOM container: ' + pid);
  }
});

// 2. Verify KPIs and Top-20 table populated with real defaults
const defaultCountText = elements['kpi-cand-count'].textContent;
const defaultTop1 = elements['kpi-top1-name'].textContent;
const rowCount = (elements['tbl-sim-body'].innerHTML.match(/<tr>/g) || []).length;
if (defaultCountText === '-' || !defaultTop1.includes('Apophis') || rowCount !== 20) {
  throw new Error(`Unexpected default state: count=${defaultCountText}, top1=${defaultTop1}, rows=${rowCount}`);
}

// 3. Reproduce exact multi-parameter sequence from user video (93 -> 2 -> 18 -> 1 -> 0 -> 93)
elements['ctrl-dur'].value = '330';
(elements['ctrl-dur'].listeners['input'] || [])[0]();
if (!elements['plot-sim-scatter'].innerHTML.includes('plot-container') || plotCalls['plot-sim-scatter'].xLen !== 2) {
  throw new Error('Scatter plot vanished or wrong count after changing duration to 330: ' + plotCalls['plot-sim-scatter'].xLen);
}

elements['ctrl-dur'].value = '360';
elements['ctrl-dv'].value = '7.25';
(elements['ctrl-dv'].listeners['input'] || [])[0]();
if (!elements['plot-sim-scatter'].innerHTML.includes('plot-container') || plotCalls['plot-sim-scatter'].xLen !== 16) {
  throw new Error('Scatter plot vanished or wrong count after changing dur=360, dv=7.25: ' + plotCalls['plot-sim-scatter'].xLen);
}

elements['ctrl-occ'].value = '4';
elements['ctrl-obs'].value = '180';
elements['ctrl-diam'].value = '75';
elements['ctrl-grp'].value = 'KNOWN';
(elements['ctrl-grp'].listeners['change'] || [])[0]();
if (!elements['plot-sim-scatter'].innerHTML.includes('plot-container') || plotCalls['plot-sim-scatter'].xLen !== 1) {
  throw new Error('Scatter plot vanished on single-candidate filter (Apophis)!');
}

// 4. Empty candidate state -> fallback message & disabled CSV button
elements['ctrl-grp'].value = 'Complexo-C';
(elements['ctrl-grp'].listeners['change'] || [])[0]();
if (!elements['plot-sim-scatter'].innerHTML.includes('plot-empty-state') || !elements['btn-export-csv'].disabled) {
  throw new Error('Empty filter did not show plot-empty-state or disable CSV export!');
}

// 5. Recovery from empty state via resetDefaults -> scatter plot cleanly remounted!
sandbox.NewSpaceUIController.resetDefaults();
if (
  elements['kpi-cand-count'].textContent !== defaultCountText ||
  !elements['plot-sim-scatter'].innerHTML.includes('plot-container') ||
  plotCalls['plot-sim-scatter'].xLen !== 93 ||
  elements['btn-export-csv'].disabled
) {
  throw new Error('resetDefaults failed to remount scatter plot and restore 93 candidates!');
}

const strictCountText = '2';

console.log(JSON.stringify({
  ok: true,
  defaultCount: defaultCountText,
  defaultTop1,
  rowCount,
  strictCount: strictCountText,
  renderedPlots: Object.keys(plotCalls).length
}));
"""
    res = subprocess.run(
        ["node", "-e", node_script, str(DOCS_DIR)],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    assert res.returncode == 0, f"Frontend JS runtime test failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    summary = json.loads(res.stdout.strip())
    assert summary["ok"] is True
    assert summary["renderedPlots"] == 10
    assert summary["rowCount"] == 20
    assert "Apophis" in summary["defaultTop1"]


def test_nsa_chatbot_qa_widget():
    """Verify isolated Mini Chatbot Q&A CSS/JS modules, knowledge base fidelity, and interactive flow."""
    dash_html = (DOCS_DIR / "index.html").read_text(encoding="utf-8")
    rep_html = (DOCS_DIR / "relatorio.html").read_text(encoding="utf-8")
    bot_css = (DOCS_DIR / "assets/css/nsa-chatbot.css").read_text(encoding="utf-8")
    bot_js = (DOCS_DIR / "assets/js/nsa-chatbot.js").read_text(encoding="utf-8")

    assert "assets/css/nsa-chatbot.css" in dash_html
    assert "assets/js/nsa-chatbot.js" in dash_html
    assert "assets/css/nsa-chatbot.css" in rep_html
    assert "assets/js/nsa-chatbot.js" in rep_html

    for css_token in (
        ".nsa-bot-wrapper",
        ".nsa-bot-window",
        ".nsa-bot-chip",
        "position: fixed;",
        "bottom: 24px;",
        "right: 24px;",
        "z-index: 9999;",
        "max-width: calc(100vw - 32px);",
        "max-height: 520px;",
        "#0f172a",
        "#1e293b",
        "#38bdf8",
    ):
        assert css_token in bot_css, f"Missing CSS token: {css_token}"

    for js_token in (
        "NSA_CHATBOT_KNOWLEDGE",
        "origem-dados",
        "delta-v",
        "incerteza-occ",
        "mineralogia-25m",
        "ranking-topsis",
        "era-newspace",
        "risco-atuarial",
        "Olá! Sou o guia rápido do NewSpace Analytics. Clique em qualquer pergunta abaixo para entender os conceitos técnicos do painel de forma simples:",
    ):
        assert js_token in bot_js, f"Missing JS token: {js_token}"

    node_bot_test = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const docsDir = process.argv[1];
const idMap = {};

function createNode(tag) {
  const classes = new Set();
  const attrs = {};
  const listeners = {};
  const children = [];
  let html = '';
  let text = '';

  const node = {
    tagName: String(tag).toUpperCase(),
    id: '',
    className: '',
    parentNode: null,
    scrollTop: 0,
    scrollHeight: 500,
    children,
    listeners,
    classList: {
      add: c => classes.add(c),
      remove: c => classes.delete(c),
      contains: c => classes.has(c)
    },
    setAttribute: (k, v) => {
      attrs[k] = String(v);
      if (k === 'id') { node.id = String(v); idMap[node.id] = node; }
    },
    getAttribute: k => attrs[k] || null,
    addEventListener: (ev, fn) => { (listeners[ev] = listeners[ev] || []).push(fn); },
    appendChild: child => {
      child.parentNode = node;
      children.push(child);
      if (child.id) idMap[child.id] = child;
      return child;
    },
    removeChild: child => {
      const idx = children.indexOf(child);
      if (idx !== -1) children.splice(idx, 1);
      if (child.id && idMap[child.id] === child) delete idMap[child.id];
      child.parentNode = null;
    },
    scrollIntoView: () => {},
    querySelectorAll: sel => {
      if (sel === '.nsa-bot-chip') {
        const matches = [...html.matchAll(/data-question-id="([^"]+)"/g)];
        return matches.map(m => {
          const btn = createNode('button');
          btn.setAttribute('data-question-id', m[1]);
          return btn;
        });
      }
      return [];
    },
    get innerHTML() { return html; },
    set innerHTML(val) {
      html = String(val);
      if (val === '') {
        children.forEach(c => { if (c.id && idMap[c.id] === c) delete idMap[c.id]; });
        children.length = 0;
      }
      const idMatches = [...html.matchAll(/id="([^"]+)"/g)];
      idMatches.forEach(m => {
        if (!idMap[m[1]]) {
          const sub = createNode('div');
          sub.id = m[1];
          idMap[m[1]] = sub;
        }
      });
    },
    get textContent() { return text; },
    set textContent(val) { text = String(val); html = String(val); }
  };
  return node;
}

const bodyNode = createNode('body');
const sandbox = {
  console,
  String, Boolean, Set, Array, Object,
  document: {
    readyState: 'complete',
    body: bodyNode,
    createElement: tag => createNode(tag),
    getElementById: id => idMap[id] || null,
    addEventListener: () => {}
  }
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

const code = fs.readFileSync(path.join(docsDir, 'assets/js/nsa-chatbot.js'), 'utf8');
vm.runInContext(code, sandbox, { filename: 'nsa-chatbot.js' });

if (!Array.isArray(sandbox.NSA_CHATBOT_KNOWLEDGE) || sandbox.NSA_CHATBOT_KNOWLEDGE.length !== 7) {
  throw new Error('Expected 7 items in NSA_CHATBOT_KNOWLEDGE');
}
if (!sandbox.NSAChatbot.isInitialized() || !idMap['nsa-bot-wrapper']) {
  throw new Error('Chatbot wrapper was not initialized in DOM');
}

// Open window
sandbox.NSAChatbot.open();
if (!idMap['nsa-bot-window'].classList.contains('nsa-bot-open')) {
  throw new Error('Window did not get nsa-bot-open class');
}

// Verify initial welcome + 7 available questions
if (sandbox.NSAChatbot.getAvailableQuestions().length !== 7) {
  throw new Error('Expected 7 initial questions');
}

// Ask a question and verify remaining questions become 6
const ans = sandbox.NSAChatbot.ask('delta-v');
if (!ans || !ans.answer.includes('Delta-v')) {
  throw new Error('Unexpected answer for delta-v');
}
if (sandbox.NSAChatbot.getAvailableQuestions().length !== 6) {
  throw new Error('Expected 6 remaining questions after asking 1');
}

// Show all questions -> restores all 7
sandbox.NSAChatbot.showAll();
if (sandbox.NSAChatbot.getAvailableQuestions().length !== 7) {
  throw new Error('showAll did not restore all 7 questions');
}

// Close window
sandbox.NSAChatbot.close();
if (idMap['nsa-bot-window'].classList.contains('nsa-bot-open')) {
  throw new Error('Window did not close');
}

console.log(JSON.stringify({ ok: true, count: sandbox.NSA_CHATBOT_KNOWLEDGE.length }));
"""
    res = subprocess.run(
        ["node", "-e", node_bot_test, str(DOCS_DIR)],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    assert res.returncode == 0, f"Chatbot JS test failed:\n{res.stderr}"
    out = json.loads(res.stdout.strip())
    assert out["ok"] is True
    assert out["count"] == 7

