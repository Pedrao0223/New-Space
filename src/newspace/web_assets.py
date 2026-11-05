"""Modular Frontend Assets & Pure JSON Serialization for NewSpace Analytics Cockpit.

Separates client-side responsibilities into clean, single-responsibility modules:
- assets/css/dashboard.css: Responsive cosmic design system (desktop, tablet, mobile)
- assets/js/data-loader.js: Payload & Plotly validation + safe numeric extraction
- assets/js/topsis-engine.js: Candidate filtering, MCDA-TOPSIS math & CSV generation
- assets/js/table-renderer.js: KPI cards, Top-20 table DOM rendering, empty & error states
- assets/js/chart-renderer.js: Static & dynamic Plotly charts + viewport responsiveness
- assets/js/ui-controller.js: Tabs, controls, mobile drawer & fault-tolerant bootstrap
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import plotly.graph_objects as go

from newspace.config import BrandPalette


def decode_plotly_pure_json(obj: Any) -> Any:
    """Recursively decode Plotly 6+/7+ binary `bdata` buffers into pure JSON lists.

    Guarantees standard JSON arrays (`[x1, x2, ...]`) compatible with all JavaScript
    iterators and Plotly.js versions, while sanitizing NaN/Inf to None.
    """
    if isinstance(obj, dict):
        if "dtype" in obj and "bdata" in obj:
            raw_bytes = base64.b64decode(obj["bdata"])
            arr = np.frombuffer(raw_bytes, dtype=np.dtype(obj["dtype"]))
            if "shape" in obj and obj["shape"]:
                dims = [int(part.strip()) for part in str(obj["shape"]).split(",") if part.strip()]
                if dims:
                    arr = arr.reshape(dims)
            return decode_plotly_pure_json(arr.tolist())
        return {str(k): decode_plotly_pure_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [decode_plotly_pure_json(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return decode_plotly_pure_json(obj.tolist())

    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)

    return obj


def figure_to_pure_dict(fig: go.Figure) -> dict[str, Any]:
    """Convert a Plotly Figure into a pure JSON-serializable dict with zero `bdata` objects."""
    raw_dict = json.loads(fig.to_json())
    pure_dict = decode_plotly_pure_json(raw_dict)
    serialized_check = json.dumps(pure_dict, ensure_ascii=False, allow_nan=False)
    if '"bdata"' in serialized_check:
        raise ValueError("Unexpected binary bdata remained after figure serialization.")
    return pure_dict


def build_dashboard_css(palette: BrandPalette) -> str:
    """Build responsive CSS stylesheet covering desktop, tablet, and smartphone viewports."""
    return f"""/* NewSpace Analytics Cockpit — Responsive Stylesheet */
:root {{
  --bg: {palette.cosmic_bg};
  --violet: {palette.space_violet};
  --blue: {palette.orbital_blue};
  --teal: {palette.deep_teal};
  --cyan: {palette.vibrant_cyan};
  --text: {palette.text_primary};
  --muted: {palette.text_muted};
}}

* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; scroll-behavior: smooth; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  overflow-x: hidden;
}}

header {{
  background: linear-gradient(90deg, var(--bg) 0%, var(--violet) 48%, var(--blue) 100%);
  border-bottom: 2px solid var(--cyan);
  padding: clamp(12px, 2.2vw, 16px) clamp(14px, 3vw, 28px);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}}

.brand-wrap {{ display: flex; align-items: center; gap: 12px; min-width: 0; flex: 1 1 280px; }}
.brand-logo {{ width: clamp(34px, 6vw, 42px); height: clamp(34px, 6vw, 42px); flex-shrink: 0; }}
.brand-logo svg {{ width: 100%; height: 100%; display: block; }}
.brand-text {{ min-width: 0; }}
.brand-title {{ font-size: clamp(14.5px, 2.4vw, 20px); font-weight: 800; letter-spacing: 0.02em; line-height: 1.22; }}
.brand-sub {{ font-size: clamp(11px, 1.8vw, 12px); color: #cbd5e1; margin-top: 3px; line-height: 1.35; }}
.header-right {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.palette-pills {{ display: flex; gap: 6px; align-items: center; font-size: 11px; color: #cbd5e1; }}
.swatch {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.4); }}

.report-btn {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 40px;
  background: rgba(28, 163, 158, 0.20);
  color: #fff;
  border: 1px solid var(--cyan);
  padding: 8px 15px;
  border-radius: 8px;
  text-decoration: none;
  font-size: 12.5px;
  font-weight: 700;
  transition: all 0.15s;
  white-space: nowrap;
}}
.report-btn:hover {{ background: var(--cyan); color: #06181c; }}

nav.tabs {{
  display: flex;
  gap: 8px;
  padding: clamp(10px, 2vw, 12px) clamp(12px, 3vw, 28px);
  background: #1c1124;
  border-bottom: 1px solid #2d223b;
  flex-wrap: wrap;
}}

.tab-btn {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  min-height: 42px;
  line-height: 1.25;
  background: rgba(35, 20, 91, 0.55);
  color: var(--text);
  border: 1px solid rgba(28, 163, 158, 0.35);
  padding: 9px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  font-size: 13px;
  transition: all 0.15s;
}}
.tab-btn:hover {{ background: var(--blue); }}
.tab-btn.active {{ background: var(--cyan); color: #091318; border-color: #fff; font-weight: 800; }}

main {{ padding: clamp(14px, 2.5vw, 22px) clamp(10px, 3vw, 28px); max-width: 1560px; margin: 0 auto; }}

.error-banner {{
  display: none;
  background: rgba(239, 68, 68, 0.16);
  border: 1px solid #f87171;
  color: #fecaca;
  padding: 12px 16px;
  border-radius: 10px;
  margin-bottom: 16px;
  font-size: 13.5px;
  line-height: 1.45;
}}
.error-banner.visible {{ display: block; }}

.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}

.kpi-row {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: clamp(10px, 1.8vw, 14px);
  margin-bottom: clamp(14px, 2.2vw, 20px);
}}
.kpi-row.kpi-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}

.kpi {{
  min-width: 0;
  background: linear-gradient(135deg, rgba(35,20,91,0.75), rgba(9,69,108,0.65));
  border: 1px solid rgba(28,163,158,0.42);
  border-radius: 10px;
  padding: clamp(12px, 2vw, 14px) clamp(12px, 2.2vw, 18px);
}}
.kpi .lbl {{ font-size: clamp(10px, 1.7vw, 11px); color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; line-height: 1.3; }}
.kpi .val {{ font-size: clamp(18px, 3vw, 23px); font-weight: 800; color: var(--cyan); margin-top: 4px; line-height: 1.2; word-break: break-word; }}
.kpi .val.val-compact {{ font-size: clamp(14.5px, 2.2vw, 18.5px); }}
.kpi .hint {{ font-size: clamp(10.8px, 1.8vw, 11.5px); color: #cbd5e1; margin-top: 4px; line-height: 1.3; }}

.grid-2 {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: clamp(14px, 2vw, 20px);
  margin-bottom: clamp(14px, 2vw, 20px);
}}
.grid-2 > *, .sim-layout > * {{ min-width: 0; }}

.chart-card {{
  min-width: 0;
  background: #1D1226;
  border: 1px solid rgba(28,163,158,0.28);
  border-radius: 12px;
  padding: clamp(12px, 2vw, 16px);
  margin-bottom: clamp(14px, 2vw, 20px);
  box-shadow: 0 10px 26px rgba(0,0,0,0.35);
  overflow: hidden;
}}
.grid-2 .chart-card {{ margin-bottom: 0; }}

.section-subtitle {{
  margin: 2px 2px 10px;
  color: var(--cyan);
  font-size: clamp(13.5px, 2.2vw, 15.5px);
  font-weight: 700;
  line-height: 1.35;
}}

.plot-box {{ width: 100%; min-height: 320px; }}
.plot-empty-state {{
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  padding: 20px;
  text-align: center;
  color: #cbd5e1;
  background: rgba(22, 13, 24, 0.72);
  border: 1px dashed rgba(28, 163, 158, 0.45);
  border-radius: 8px;
  font-size: 13.5px;
}}

.sim-layout {{ display: grid; grid-template-columns: 380px minmax(0, 1fr); gap: 20px; align-items: start; }}

.criteria-intro-card {{
  background: linear-gradient(135deg, rgba(35, 20, 91, 0.58) 0%, rgba(9, 69, 108, 0.48) 100%);
  border: 1px solid rgba(28, 163, 158, 0.38);
  border-left: 4px solid var(--cyan);
  border-radius: 12px;
  padding: clamp(14px, 2.2vw, 18px) clamp(14px, 2.5vw, 22px);
  margin-bottom: clamp(14px, 2.2vw, 20px);
}}
.criteria-intro-title {{
  margin: 0 0 6px;
  font-size: clamp(14.5px, 2.2vw, 16.5px);
  font-weight: 800;
  color: var(--cyan);
  letter-spacing: 0.01em;
}}
.criteria-intro-lead {{
  margin: 0 0 12px;
  font-size: clamp(12.5px, 1.9vw, 13.8px);
  color: #e2e8f0;
  line-height: 1.55;
}}
.criteria-pillars {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}}
.criteria-pillar {{
  min-width: 0;
  background: rgba(22, 13, 24, 0.62);
  border: 1px solid rgba(28, 163, 158, 0.24);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: clamp(11.8px, 1.8vw, 12.6px);
  color: #cbd5e1;
  line-height: 1.45;
}}
.criteria-pillar strong {{
  display: block;
  color: #fff;
  font-size: 12.6px;
  margin-bottom: 3px;
}}

.control-box {{
  min-width: 0;
  background: linear-gradient(180deg, rgba(35,20,91,0.78), rgba(22,13,24,0.95));
  border: 1px solid var(--cyan);
  border-radius: 12px;
  padding: clamp(14px, 2.4vw, 18px);
}}

.mobile-filter-toggle {{
  display: none;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  background: rgba(28, 163, 158, 0.18);
  color: #fff;
  border: 1px solid var(--cyan);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  min-height: 44px;
}}
.ctrl-body.collapsed {{ display: none; }}
.ctrl-columns {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
.ctrl-col {{ min-width: 0; }}
.control-box h3 {{ margin: 0 0 6px; font-size: 14.5px; color: var(--cyan); border-bottom: 1px solid rgba(28,163,158,0.3); padding-bottom: 6px; }}
.ctrl-section-note {{ margin: 0 0 12px; font-size: 11.4px; color: #cbd5e1; line-height: 1.4; }}
.ctrl-group {{ margin-bottom: 12px; }}
.ctrl-group label {{ display: flex; justify-content: space-between; align-items: baseline; gap: 8px; font-size: 12.2px; font-weight: 600; margin-bottom: 3px; color: #f1f5f9; }}
.ctrl-hint {{ display: block; font-size: 11px; color: var(--muted); margin-top: 2px; line-height: 1.32; }}
.ctrl-group input[type=range] {{ width: 100%; accent-color: var(--cyan); height: 26px; cursor: pointer; }}
.ctrl-group select {{
  width: 100%;
  background: var(--bg);
  color: #fff;
  border: 1px solid var(--teal);
  padding: 8px 10px;
  border-radius: 6px;
  min-height: 40px;
  font-size: 13px;
}}

.ctrl-actions {{ display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 10px; }}
.action-btn {{
  width: 100%;
  min-height: 42px;
  padding: 10px 12px;
  background: var(--cyan);
  color: #07191e;
  border: none;
  border-radius: 8px;
  font-weight: 800;
  cursor: pointer;
  font-size: 13px;
}}
.action-btn.secondary {{ background: var(--blue); color: #fff; border: 1px solid var(--cyan); }}

.mobile-scroll-hint {{
  display: none;
  font-size: 11px;
  color: var(--cyan);
  margin: 2px 2px 8px;
  font-weight: 600;
}}
.table-wrap {{
  width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  overscroll-behavior-x: contain;
  border-radius: 8px;
  border: 1px solid rgba(28, 163, 158, 0.22);
}}
.table-scroll-y {{ max-height: 410px; overflow-y: auto; }}

table.ranking-tbl, table.data-table {{ width: 100%; border-collapse: collapse; font-size: clamp(11.8px, 2vw, 12.8px); }}
table.ranking-tbl th, table.data-table th {{
  background: var(--blue);
  color: #fff;
  padding: 10px 12px;
  font-weight: 700;
  position: sticky;
  top: 0;
  z-index: 2;
  border-bottom: 2px solid var(--cyan);
  white-space: nowrap;
}}
table.ranking-tbl td, table.data-table td {{
  padding: 8.5px 12px;
  border-bottom: 1px solid rgba(148,163,184,0.16);
  vertical-align: middle;
}}
.txt-col {{ text-align: left; }}
.num-col {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
table.ranking-tbl tr:nth-child(even), table.data-table tr:nth-child(even) {{ background: rgba(35,20,91,0.22); }}
table.ranking-tbl tr:hover, table.data-table tr:hover {{ background: rgba(28,163,158,0.14); }}

/* Tablet Landscape & Portrait (<= 1080px) */
@media (max-width: 1080px) {{
  .sim-layout {{ grid-template-columns: 1fr; }}
  .ctrl-columns {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }}
  nav.tabs {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .tab-btn {{ width: 100%; }}
}}

/* Tablet Portrait & Phablets (<= 960px) */
@media (max-width: 960px) {{
  .grid-2 {{ grid-template-columns: 1fr; }}
  .criteria-pillars {{ grid-template-columns: 1fr; gap: 8px; }}
  .kpi-row, .kpi-row.kpi-3 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .kpi-row.kpi-3 .kpi:last-child {{ grid-column: 1 / -1; }}
  .mobile-scroll-hint {{ display: block; }}
  table.ranking-tbl th:first-child, table.ranking-tbl td:first-child,
  table.data-table th:first-child, table.data-table td:first-child {{
    position: sticky;
    left: 0;
    z-index: 3;
    box-shadow: 2px 0 6px rgba(0, 0, 0, 0.45);
  }}
  table.ranking-tbl th:first-child, table.data-table th:first-child {{ background: var(--blue); z-index: 4; }}
  table.ranking-tbl td:first-child, table.data-table td:first-child {{ background: #1d1226; }}
  table.ranking-tbl tr:nth-child(even) td:first-child,
  table.data-table tr:nth-child(even) td:first-child {{ background: #211536; }}
}}

/* Smartphones (<= 640px) */
@media (max-width: 640px) {{
  .header-right {{ width: 100%; justify-content: space-between; }}
  .palette-pills {{ display: none; }}
  .report-btn {{ width: 100%; min-height: 44px; font-size: 13px; }}
  .tab-btn {{ font-size: 12px; padding: 8px 10px; min-height: 46px; }}
  .mobile-filter-toggle {{ display: flex; }}
  .ctrl-body {{ margin-top: 14px; }}
  .ctrl-columns {{ grid-template-columns: 1fr; gap: 14px; }}
  .ctrl-group select {{ font-size: 16px; }}
  .ctrl-group input[type=range] {{ height: 32px; }}
  table.ranking-tbl th, table.ranking-tbl td,
  table.data-table th, table.data-table td {{ padding: 8px 9px; }}
}}

/* Compact Smartphones (<= 380px) */
@media (max-width: 380px) {{
  nav.tabs {{ grid-template-columns: 1fr; }}
  .kpi-row, .kpi-row.kpi-3 {{ grid-template-columns: 1fr; }}
  .kpi-row.kpi-3 .kpi:last-child {{ grid-column: auto; }}
}}
"""


DATA_LOADER_JS = """/* Module 1: Data & Library Validation (Single Responsibility: Safe Payload Access) */
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
"""


TOPSIS_ENGINE_JS = """/* Module 2: Candidate Filtering & MCDA-TOPSIS Engine (Single Responsibility: Domain Math & CSV) */
(function (global) {
  'use strict';

  const DEFAULT_CRITERIA = Object.freeze({
    maxDv: 6.5,
    maxDur: 450,
    maxOcc: 3,
    minObs: 50,
    minDiam: 25,
    grpFilter: 'ALL',
    w1: 0.40,
    w2: 0.25,
    w3: 0.20,
    w4: 0.15
  });

  function filterCandidates(records, criteria) {
    return records.filter(function (d) {
      if (d.dv > criteria.maxDv) return false;
      if (d.dur > criteria.maxDur) return false;
      if (d.occ > criteria.maxOcc) return false;
      if (d.obs < criteria.minObs) return false;
      if (d.diam < criteria.minDiam) return false;
      if (criteria.grpFilter === 'ALL') return true;
      if (criteria.grpFilter === 'KNOWN') return d.has_spec === 1;
      return String(d.grp || '').indexOf(criteria.grpFilter) === 0;
    });
  }

  function rankCandidatesTOPSIS(records, criteria) {
    const cand = filterCandidates(records, criteria);
    if (cand.length === 0) {
      return [];
    }

    let maxLogObs = 1.0;
    let maxLogArc = 1.0;
    for (let i = 0; i < cand.length; i += 1) {
      const lo = Math.log1p(cand[i].obs);
      const la = Math.log1p(cand[i].arc);
      if (lo > maxLogObs) maxLogObs = lo;
      if (la > maxLogArc) maxLogArc = la;
    }

    const feats = cand.map(function (d) {
      const boundedOcc = Math.min(9, Math.max(0, d.occ));
      const infoIdx = 100.0 * (
        0.35 * (Math.log1p(d.obs) / maxLogObs) +
        0.25 * (Math.log1p(d.arc) / maxLogArc) +
        0.25 * ((9.0 - boundedOcc) / 9.0) +
        0.10 * d.has_spec +
        0.05 * d.diam_m_real
      );
      const c1 = 1.0 / (d.dv * Math.sqrt(Math.max(d.dur, 1.0)));
      const c2 = infoIdx;
      const c3 = Math.log1p(d.diam);
      const c4 = Math.log1p(d.traj);
      return Object.assign({}, d, { infoIdx: infoIdx, c1: c1, c2: c2, c3: c3, c4: c4 });
    });

    function vectorNorm(key) {
      let sumSq = 0.0;
      for (let i = 0; i < feats.length; i += 1) {
        sumSq += feats[i][key] * feats[i][key];
      }
      return Math.sqrt(sumSq) || 1.0;
    }

    const n1 = vectorNorm('c1');
    const n2 = vectorNorm('c2');
    const n3 = vectorNorm('c3');
    const n4 = vectorNorm('c4');

    const wSum = (criteria.w1 + criteria.w2 + criteria.w3 + criteria.w4) || 1.0;
    const nw1 = criteria.w1 / wSum;
    const nw2 = criteria.w2 / wSum;
    const nw3 = criteria.w3 / wSum;
    const nw4 = criteria.w4 / wSum;

    let ib1 = -Infinity, iw1 = Infinity;
    let ib2 = -Infinity, iw2 = Infinity;
    let ib3 = -Infinity, iw3 = Infinity;
    let ib4 = -Infinity, iw4 = Infinity;

    for (let i = 0; i < feats.length; i += 1) {
      const r = feats[i];
      r.v1 = (r.c1 / n1) * nw1;
      r.v2 = (r.c2 / n2) * nw2;
      r.v3 = (r.c3 / n3) * nw3;
      r.v4 = (r.c4 / n4) * nw4;

      if (r.v1 > ib1) ib1 = r.v1;
      if (r.v1 < iw1) iw1 = r.v1;
      if (r.v2 > ib2) ib2 = r.v2;
      if (r.v2 < iw2) iw2 = r.v2;
      if (r.v3 > ib3) ib3 = r.v3;
      if (r.v3 < iw3) iw3 = r.v3;
      if (r.v4 > ib4) ib4 = r.v4;
      if (r.v4 < iw4) iw4 = r.v4;
    }

    for (let i = 0; i < feats.length; i += 1) {
      const r = feats[i];
      const db = Math.hypot(r.v1 - ib1, r.v2 - ib2, r.v3 - ib3, r.v4 - ib4);
      const dw = Math.hypot(r.v1 - iw1, r.v2 - iw2, r.v3 - iw3, r.v4 - iw4);
      r.topsis = (db + dw) > 0 ? (100.0 * dw / (db + dw)) : 50.0;
    }

    feats.sort(function (a, b) {
      return (b.topsis - a.topsis) || (a.dv - b.dv);
    });

    return feats;
  }

  function buildQualifiedCSV(ranked) {
    const header = [
      'rank', 'spkid', 'designacao', 'asteroide_oficial', 'espectro',
      'delta_v_kms', 'duracao_dias', 'diametro_ref_m', 'observacoes', 'occ', 'score_topsis'
    ];
    const rows = ranked.map(function (r, idx) {
      const safePdes = '"' + String(r.pdes).replace(/"/g, '""') + '"';
      const safeName = '"' + String(r.name).replace(/"/g, '""') + '"';
      const safeSpec = '"' + String(r.spec).replace(/"/g, '""') + '"';
      return [
        idx + 1, r.spkid, safePdes, safeName, safeSpec,
        r.dv, r.dur, r.diam, r.obs, r.occ, r.topsis.toFixed(2)
      ].join(',');
    });
    return '\\uFEFF' + [header.join(',')].concat(rows).join('\\n');
  }

  global.NewSpaceTopsisEngine = {
    DEFAULT_CRITERIA: DEFAULT_CRITERIA,
    filterCandidates: filterCandidates,
    rankCandidatesTOPSIS: rankCandidatesTOPSIS,
    buildQualifiedCSV: buildQualifiedCSV
  };
})(typeof window !== 'undefined' ? window : globalThis);
"""


TABLE_RENDERER_JS = """/* Module 3: KPI & Table DOM Renderer (Single Responsibility: HTML State & Error Feedback) */
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
"""


CHART_RENDERER_JS = """/* Module 4: Plotly Chart Renderer & Responsive Viewport Engine */
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

    el.innerHTML = '';
    const vw = global.innerWidth || document.documentElement.clientWidth || 1280;
    const isMobile = vw <= 640;
    const isTablet = vw > 640 && vw <= 1024;
    const topPlot = ranked.slice(0, 300);

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
        color: topPlot.map(function (r) { return r.topsis; }),
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
      xaxis: { title: { text: 'Δv Mínimo NASA NHATS (km/s)' }, gridcolor: '#2A2438', automargin: true },
      yaxis: { title: { text: 'Duração da Missão (Dias)' }, gridcolor: '#2A2438', automargin: true }
    };

    global.Plotly.react(el, [trace], layout, PLOT_CONFIG);
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
"""


UI_CONTROLLER_JS = """/* Module 5: Interactive UI Controller & Fault-Tolerant Bootstrap */
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
      if (el) el.addEventListener('input', recomputeSimulation);
    });

    const grpEl = document.getElementById('ctrl-grp');
    if (grpEl) grpEl.addEventListener('change', recomputeSimulation);

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
"""
