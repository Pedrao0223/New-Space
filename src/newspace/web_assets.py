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
    """Build the minimalist deep-space responsive stylesheet (desktop, tablet and smartphone)."""
    tokens = {
        "__BG__": palette.cosmic_bg,
        "__NEBULA__": palette.space_violet,
        "__INDIGO__": palette.orbital_blue,
        "__VIOLET__": palette.deep_teal,
        "__ACCENT__": palette.vibrant_cyan,
        "__TEXT__": palette.text_primary,
        "__MUTED__": palette.text_muted,
        "__MINT__": palette.accent_mint,
        "__CORAL__": palette.accent_coral,
    }
    css = DASHBOARD_CSS_TEMPLATE
    for key, value in tokens.items():
        css = css.replace(key, value)
    return css


DASHBOARD_CSS_TEMPLATE = """/* NewSpace Analytics Cockpit — Minimalist Deep-Space Stylesheet */
:root {
  --bg: __BG__;
  --nebula: __NEBULA__;
  --indigo: __INDIGO__;
  --violet: __VIOLET__;
  --accent: __ACCENT__;
  --text: __TEXT__;
  --muted: __MUTED__;
  --mint: __MINT__;
  --coral: __CORAL__;
  --surface: rgba(255, 255, 255, 0.025);
  --surface-2: rgba(255, 255, 255, 0.05);
  --solid: #0A0D17;
  --border: rgba(148, 163, 184, 0.10);
  --border-strong: rgba(148, 163, 184, 0.20);
  --accent-soft: rgba(56, 189, 248, 0.12);
  --radius: 16px;
  --radius-sm: 10px;
  --font: Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; scroll-behavior: smooth; }
body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}
a { color: inherit; }
::selection { background: rgba(56, 189, 248, 0.30); }

/* ---------- Space backdrop: soft nebula glows + faint star field ---------- */
.space-backdrop {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  background:
    radial-gradient(1100px 620px at 88% -12%, rgba(99, 102, 241, 0.18), transparent 62%),
    radial-gradient(900px 520px at -8% 6%, rgba(56, 189, 248, 0.10), transparent 60%),
    radial-gradient(900px 700px at 50% 118%, rgba(139, 92, 246, 0.12), transparent 62%),
    var(--bg);
}
.space-backdrop::after {
  content: '';
  position: absolute;
  inset: 0;
  opacity: 0.55;
  background-image:
    radial-gradient(1px 1px at 24px 36px, rgba(255, 255, 255, 0.55), transparent 60%),
    radial-gradient(1px 1px at 132px 92px, rgba(255, 255, 255, 0.35), transparent 60%),
    radial-gradient(1.5px 1.5px at 210px 170px, rgba(186, 230, 253, 0.45), transparent 60%),
    radial-gradient(1px 1px at 68px 214px, rgba(255, 255, 255, 0.30), transparent 60%),
    radial-gradient(1px 1px at 178px 26px, rgba(221, 214, 254, 0.40), transparent 60%);
  background-size: 260px 260px;
}

/* ---------- Top bar ---------- */
.topbar {
  position: sticky;
  top: 0;
  z-index: 50;
  background: rgba(5, 7, 13, 0.72);
  backdrop-filter: saturate(140%) blur(14px);
  -webkit-backdrop-filter: saturate(140%) blur(14px);
  border-bottom: 1px solid var(--border);
}
.topbar-inner {
  max-width: 1400px;
  margin: 0 auto;
  padding: 12px clamp(16px, 4vw, 40px);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.brand { display: flex; align-items: center; gap: 12px; min-width: 0; text-decoration: none; }
.brand-logo { width: 32px; height: 32px; flex-shrink: 0; }
.brand-logo svg { width: 100%; height: 100%; display: block; }
.brand-text { display: flex; flex-direction: column; min-width: 0; }
.brand-title { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
.brand-sub { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.report-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 38px;
  padding: 8px 16px;
  border-radius: 999px;
  border: 1px solid var(--border-strong);
  background: var(--surface);
  color: var(--text);
  text-decoration: none;
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
  transition: border-color 0.2s, background 0.2s, color 0.2s;
}
.report-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }

/* ---------- Hero ---------- */
.hero {
  max-width: 1400px;
  margin: 0 auto;
  padding: clamp(28px, 5vw, 56px) clamp(16px, 4vw, 40px) clamp(14px, 2.4vw, 22px);
}
.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-weight: 500;
  color: var(--accent);
  letter-spacing: 0.02em;
}
.eyebrow::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 12px var(--accent);
}
.hero-title {
  margin: 10px 0 8px;
  font-size: clamp(24px, 4.2vw, 40px);
  font-weight: 600;
  letter-spacing: -0.025em;
  line-height: 1.12;
  background: linear-gradient(92deg, #FFFFFF 0%, #C7D2FE 55%, var(--accent) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero-sub { margin: 0; color: var(--muted); font-size: clamp(12.5px, 1.6vw, 14px); }

/* ---------- Tabs (segmented control, horizontally scrollable on mobile) ---------- */
nav.tabs {
  max-width: 1400px;
  margin: 0 auto;
  padding: 8px clamp(16px, 4vw, 40px) 0;
}
.tabs-inner {
  display: inline-flex;
  gap: 4px;
  padding: 4px;
  border-radius: 999px;
  background: var(--surface);
  border: 1px solid var(--border);
  max-width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
}
.tabs-inner::-webkit-scrollbar { display: none; }
.tab-btn {
  flex: 0 0 auto;
  min-height: 38px;
  padding: 8px 18px;
  border-radius: 999px;
  border: 0;
  background: transparent;
  color: var(--muted);
  font-family: inherit;
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
  cursor: pointer;
  transition: color 0.2s, background 0.2s;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { background: var(--accent-soft); color: var(--accent); box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.28); }
.tab-btn:focus-visible, .action-btn:focus-visible, .report-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

main {
  max-width: 1400px;
  margin: 0 auto;
  padding: clamp(18px, 3vw, 28px) clamp(16px, 4vw, 40px) 40px;
}

.error-banner {
  display: none;
  background: rgba(251, 113, 133, 0.10);
  border: 1px solid rgba(251, 113, 133, 0.45);
  color: #FECDD3;
  padding: 12px 16px;
  border-radius: var(--radius-sm);
  margin-bottom: 18px;
  font-size: 13.5px;
}
.error-banner.visible { display: block; }

.tab-panel { display: none; animation: fade-in 0.25s ease; }
.tab-panel.active { display: block; }
@keyframes fade-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

/* ---------- KPIs ---------- */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.kpi-row.kpi-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.kpi {
  min-width: 0;
  padding: 18px 20px;
  border-radius: var(--radius);
  background: var(--surface);
  border: 1px solid var(--border);
}
.kpi .lbl { font-size: 11.5px; color: var(--muted); letter-spacing: 0.01em; }
.kpi .val {
  margin-top: 6px;
  font-size: clamp(22px, 2.6vw, 28px);
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.15;
  font-variant-numeric: tabular-nums;
  word-break: break-word;
}
.kpi .val small { font-size: 0.55em; font-weight: 500; color: var(--muted); }
.kpi .val.val-compact { font-size: clamp(15px, 1.7vw, 18px); line-height: 1.3; }
.kpi .hint { margin-top: 6px; font-size: 12px; color: var(--muted); line-height: 1.35; }
.kpi-accent { background: linear-gradient(160deg, rgba(56, 189, 248, 0.10), rgba(99, 102, 241, 0.05) 70%); border-color: rgba(56, 189, 248, 0.22); }
.kpi-accent .val { color: var(--accent); }

/* ---------- Criteria intro (collapsible) ---------- */
.criteria-intro-card {
  margin-bottom: 16px;
  border-radius: var(--radius);
  background: var(--surface);
  border: 1px solid var(--border);
}
.criteria-intro-card summary {
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 20px;
  cursor: pointer;
}
.criteria-intro-card summary::-webkit-details-marker { display: none; }
.criteria-intro-title { margin: 0; font-size: 14px; font-weight: 500; color: var(--text); }
.summary-toggle {
  flex-shrink: 0;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  border: 1px solid var(--border-strong);
  position: relative;
}
.summary-toggle::before, .summary-toggle::after {
  content: '';
  position: absolute;
  left: 50%;
  top: 50%;
  width: 10px;
  height: 1.5px;
  background: var(--muted);
  transform: translate(-50%, -50%);
  transition: transform 0.2s;
}
.summary-toggle::after { transform: translate(-50%, -50%) rotate(90deg); }
.criteria-intro-card[open] .summary-toggle::after { transform: translate(-50%, -50%) rotate(0deg); }
.criteria-intro-lead { margin: 0; padding: 0 20px 14px; color: #CBD5E1; font-size: 13.5px; line-height: 1.65; }
.criteria-pillars {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  padding: 0 20px 20px;
}
.criteria-pillar {
  min-width: 0;
  padding: 14px 16px;
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 12.8px;
  line-height: 1.55;
}
.criteria-pillar strong { display: block; margin-bottom: 4px; color: var(--text); font-size: 13px; font-weight: 600; }

/* ---------- Simulator layout ---------- */
.sim-layout { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 16px; align-items: start; }
.grid-2 > *, .sim-layout > * { min-width: 0; }

.control-box {
  position: sticky;
  top: 76px;
  min-width: 0;
  padding: 20px;
  border-radius: var(--radius);
  background: var(--surface);
  border: 1px solid var(--border);
  max-height: calc(100vh - 92px);
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--border-strong) transparent;
}
.mobile-filter-toggle {
  display: none;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  min-height: 44px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-strong);
  background: transparent;
  color: var(--text);
  font-family: inherit;
  font-size: 13.5px;
  font-weight: 500;
  cursor: pointer;
}
#filter-toggle-icon { color: var(--muted); font-size: 12px; }
.ctrl-body.collapsed { display: none; }
.ctrl-columns { display: grid; grid-template-columns: 1fr; gap: 24px; }
.ctrl-col { min-width: 0; }
.control-box h3 {
  margin: 0 0 4px;
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.ctrl-section-note { margin: 0 0 16px; font-size: 12px; color: var(--muted); line-height: 1.45; opacity: 0.85; }
.ctrl-group { margin-bottom: 16px; }
.ctrl-group label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
}
.ctrl-group label strong {
  min-width: 44px;
  padding: 2px 8px;
  border-radius: 6px;
  background: var(--accent-soft);
  color: var(--accent);
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  text-align: center;
}
.ctrl-hint { display: block; margin-top: 4px; font-size: 11.5px; color: var(--muted); line-height: 1.4; opacity: 0.8; }

/* Custom range slider */
.ctrl-group input[type=range] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 22px;
  margin: 0;
  background: transparent;
  cursor: pointer;
  accent-color: var(--accent);
}
.ctrl-group input[type=range]::-webkit-slider-runnable-track {
  height: 4px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(99, 102, 241, 0.55), rgba(56, 189, 248, 0.55));
}
.ctrl-group input[type=range]::-moz-range-track {
  height: 4px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(99, 102, 241, 0.55), rgba(56, 189, 248, 0.55));
}
.ctrl-group input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  margin-top: -6px;
  border-radius: 50%;
  background: #FFFFFF;
  border: 3px solid var(--accent);
  box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.15);
  transition: box-shadow 0.15s;
}
.ctrl-group input[type=range]::-moz-range-thumb {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #FFFFFF;
  border: 3px solid var(--accent);
  box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.15);
}
.ctrl-group input[type=range]:hover::-webkit-slider-thumb,
.ctrl-group input[type=range]:focus-visible::-webkit-slider-thumb { box-shadow: 0 0 0 7px rgba(56, 189, 248, 0.22); }
.ctrl-group input[type=range]:focus { outline: none; }

.ctrl-group select {
  width: 100%;
  min-height: 40px;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-strong);
  background: var(--solid);
  color: var(--text);
  font-family: inherit;
  font-size: 13px;
}
.ctrl-group select:focus { outline: none; border-color: var(--accent); }

.ctrl-actions { display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 18px; }
.action-btn {
  width: 100%;
  min-height: 42px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  border: 0;
  background: var(--accent);
  color: #04121C;
  font-family: inherit;
  font-size: 13.5px;
  font-weight: 600;
  cursor: pointer;
  transition: filter 0.2s, background 0.2s, border-color 0.2s;
}
.action-btn:hover { filter: brightness(1.08); }
.action-btn.secondary { background: transparent; color: var(--text); border: 1px solid var(--border-strong); }
.action-btn.secondary:hover { border-color: var(--accent); color: var(--accent); filter: none; }
.action-btn:disabled { opacity: 0.4; cursor: not-allowed; filter: none; }

/* ---------- Cards, charts & tables ---------- */
.sim-results { min-width: 0; }
.grid-2 {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}
.chart-card {
  min-width: 0;
  margin-bottom: 16px;
  padding: 18px 20px;
  border-radius: var(--radius);
  background: var(--surface);
  border: 1px solid var(--border);
  overflow: hidden;
}
.grid-2 .chart-card { margin-bottom: 0; }
.section-subtitle {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
  line-height: 1.4;
  letter-spacing: -0.005em;
}
.plot-box { width: 100%; min-height: 320px; }
.plot-empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  padding: 24px;
  text-align: center;
  color: var(--muted);
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-sm);
  font-size: 13.5px;
}

.mobile-scroll-hint { display: none; margin: 0 0 8px; font-size: 11.5px; color: var(--muted); }
.table-wrap {
  width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  overscroll-behavior-x: contain;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
.table-scroll-y { max-height: 440px; overflow-y: auto; }

table.ranking-tbl, table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
table.ranking-tbl th, table.data-table th {
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 11px 14px;
  background: var(--solid);
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 500;
  letter-spacing: 0.02em;
  border-bottom: 1px solid var(--border-strong);
  white-space: nowrap;
}
table.ranking-tbl td, table.data-table td {
  padding: 11px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
  color: #CBD5E1;
}
table.ranking-tbl tbody tr:last-child td, table.data-table tbody tr:last-child td { border-bottom: 0; }
.txt-col { text-align: left; }
.num-col { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.ranking-tbl tbody tr, table.data-table tbody tr { transition: background 0.15s; }
table.ranking-tbl tbody tr:hover, table.data-table tbody tr:hover { background: var(--surface-2); }
.cell-rank { color: var(--muted); font-variant-numeric: tabular-nums; }
.cell-good { color: var(--mint); font-weight: 500; }
.cell-accent { color: var(--accent); font-weight: 600; }

/* ---------- Footer ---------- */
.site-footer {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px clamp(16px, 4vw, 40px) 32px;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  color: var(--muted);
  font-size: 12px;
  border-top: 1px solid var(--border);
}
.site-footer a { text-decoration: none; }
.site-footer a:hover { color: var(--accent); }

/* Tablet Landscape & Portrait (<= 1080px) */
@media (max-width: 1080px) {
  .sim-layout { grid-template-columns: 1fr; }
  .control-box { position: static; max-height: none; overflow: visible; }
  .ctrl-columns { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; }
}

/* Tablet Portrait & Phablets (<= 960px) */
@media (max-width: 960px) {
  .grid-2 { grid-template-columns: 1fr; }
  .criteria-pillars { grid-template-columns: 1fr; }
  .kpi-row, .kpi-row.kpi-3 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .kpi-row.kpi-3 .kpi:last-child { grid-column: 1 / -1; }
  .mobile-scroll-hint { display: block; }
  table.ranking-tbl th:first-child, table.ranking-tbl td:first-child,
  table.data-table th:first-child, table.data-table td:first-child {
    position: sticky;
    left: 0;
    z-index: 3;
    background: var(--solid);
  }
  table.ranking-tbl th:first-child, table.data-table th:first-child { z-index: 4; }
}

/* Smartphones (<= 640px) */
@media (max-width: 640px) {
  body { font-size: 13.5px; }
  .brand-sub { display: none; }
  .report-btn { padding: 8px 12px; font-size: 12.5px; }
  .hero { padding-top: 24px; }
  nav.tabs { position: sticky; top: 57px; z-index: 40; padding-top: 8px; padding-bottom: 8px; background: rgba(5, 7, 13, 0.85); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); }
  .tabs-inner { display: flex; width: 100%; }
  .tab-btn { padding: 8px 14px; font-size: 12.5px; }
  .kpi { padding: 14px 16px; }
  .chart-card { padding: 14px; }
  .criteria-intro-card summary { padding: 12px 14px; }
  .criteria-intro-lead { padding: 0 14px 12px; }
  .criteria-pillars { padding: 0 14px 14px; }
  .control-box { padding: 14px; }
  .mobile-filter-toggle { display: flex; }
  .ctrl-body { margin-top: 16px; }
  .ctrl-columns { grid-template-columns: 1fr; gap: 20px; }
  .ctrl-group select { font-size: 16px; }
  .ctrl-group input[type=range] { height: 30px; }
  table.ranking-tbl th, table.ranking-tbl td,
  table.data-table th, table.data-table td { padding: 9px 10px; }
}

/* Compact Smartphones (<= 380px) */
@media (max-width: 380px) {
  .kpi-row, .kpi-row.kpi-3 { grid-template-columns: 1fr; }
  .kpi-row.kpi-3 .kpi:last-child { grid-column: auto; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
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
    const exportBtn = document.getElementById('btn-export-csv');

    if (exportBtn) exportBtn.disabled = ranked.length === 0;

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
          '<td class="txt-col cell-rank">#' + (idx + 1) + '</td>' +
          '<td class="txt-col">' + escapeHtml(r.pdes) + '</td>' +
          '<td class="txt-col">' + escapeHtml(r.name) + '</td>' +
          '<td class="txt-col">' + escapeHtml(r.spec) + '</td>' +
          '<td class="num-col cell-good">' + r.dv.toFixed(3) + '</td>' +
          '<td class="num-col">' + r.dur + '</td>' +
          '<td class="num-col">' + r.diam.toFixed(1) + '</td>' +
          '<td class="num-col">' + r.obs.toLocaleString('pt-BR') + '</td>' +
          '<td class="num-col">' + r.occ + '</td>' +
          '<td class="num-col cell-accent">' + r.topsis.toFixed(2) + '</td>' +
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
    copy.paper_bgcolor = 'rgba(0,0,0,0)';
    copy.plot_bgcolor = '#05070D';
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
        colorscale: [[0, '#1E1B4B'], [0.35, '#6366F1'], [0.7, '#8B5CF6'], [1, '#38BDF8']],
        colorbar: {
          title: { text: isMobile ? 'TOPSIS' : 'Score<br>TOPSIS', font: { size: isMobile ? 10 : 11.5 } },
          thickness: isMobile ? 10 : 14,
          len: isMobile ? 0.80 : 0.90,
          outlinewidth: 0,
          tickfont: { size: isMobile ? 9.5 : 11, color: '#7C8599' }
        },
        line: { color: 'rgba(226,232,240,0.45)', width: 1 }
      }
    };

    const layout = {
      title: { text: '' },
      height: isMobile ? 355 : (isTablet ? 420 : 460),
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#05070D',
      font: {
        color: '#E2E8F0',
        family: "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
        size: isMobile ? 10.5 : (isTablet ? 11.2 : 12)
      },
      margin: { l: isMobile ? 44 : 55, r: isMobile ? 12 : 24, t: 18, b: 46 },
      xaxis: {
        title: { text: 'Δv Mínimo NASA NHATS (km/s)' },
        gridcolor: '#161B2B',
        zerolinecolor: '#161B2B',
        linecolor: '#161B2B',
        tickfont: { color: '#7C8599' },
        automargin: true,
        autorange: true
      },
      yaxis: {
        title: { text: 'Duração da Missão (Dias)' },
        gridcolor: '#161B2B',
        zerolinecolor: '#161B2B',
        linecolor: '#161B2B',
        tickfont: { color: '#7C8599' },
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
"""


NSA_CHATBOT_CSS = """/* NewSpace Analytics — Mini Chatbot Q&A para Leigos (100% Isolated Client-Side Styles) */
.nsa-bot-wrapper {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  font-family: var(--font, Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif);
  color: #E2E8F0;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

.nsa-bot-wrapper *,
.nsa-bot-wrapper *::before,
.nsa-bot-wrapper *::after {
  box-sizing: border-box;
}

/* Floating Launcher Button */
.nsa-bot-launcher {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-height: 46px;
  padding: 10px 18px;
  border-radius: 999px;
  background: linear-gradient(135deg, #0b0f19 0%, #0f172a 100%);
  border: 1px solid #0284c7;
  color: #F8FAFC;
  font-family: inherit;
  font-size: 13.5px;
  font-weight: 600;
  letter-spacing: 0.01em;
  cursor: pointer;
  box-shadow: 0 10px 28px rgba(5, 7, 13, 0.8), 0 0 0 1px rgba(56, 189, 248, 0.18), 0 0 18px rgba(2, 132, 199, 0.28);
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease, background 0.2s ease;
}

.nsa-bot-launcher:hover,
.nsa-bot-launcher:focus-visible {
  border-color: #38bdf8;
  box-shadow: 0 12px 32px rgba(5, 7, 13, 0.9), 0 0 0 2px rgba(56, 189, 248, 0.32), 0 0 24px rgba(56, 189, 248, 0.4);
  transform: translateY(-1px);
  outline: none;
}

.nsa-bot-launcher-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #38bdf8;
  box-shadow: 0 0 10px #38bdf8;
  flex-shrink: 0;
}

.nsa-bot-launcher-icon {
  width: 18px;
  height: 18px;
  color: #38bdf8;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.nsa-bot-launcher-label {
  white-space: nowrap;
}

/* Assistant Window Card */
.nsa-bot-window {
  width: 360px;
  max-width: calc(100vw - 32px);
  max-height: 520px;
  height: min(520px, calc(100vh - 96px));
  margin-bottom: 12px;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 16px;
  box-shadow: 0 22px 54px rgba(5, 7, 13, 0.88), 0 0 0 1px rgba(56, 189, 248, 0.14);
  display: none;
  flex-direction: column;
  overflow: hidden;
  animation: nsa-bot-fade-up 0.2s ease-out;
}

.nsa-bot-window.nsa-bot-open {
  display: flex;
}

@keyframes nsa-bot-fade-up {
  from {
    opacity: 0;
    transform: translateY(8px) scale(0.98);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

/* Header */
.nsa-bot-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  background: linear-gradient(135deg, #0b0f19 0%, #1e1b4b 100%);
  border-bottom: 1px solid #1e293b;
  flex-shrink: 0;
}

.nsa-bot-header-info {
  min-width: 0;
}

.nsa-bot-title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #F8FAFC;
  line-height: 1.3;
  letter-spacing: -0.01em;
}

.nsa-bot-subtitle {
  margin: 3px 0 0;
  font-size: 11.5px;
  color: #94A3B8;
  line-height: 1.35;
}

.nsa-bot-close {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  border: 1px solid #1e293b;
  background: rgba(255, 255, 255, 0.04);
  color: #CBD5E1;
  font-family: inherit;
  font-size: 18px;
  line-height: 1;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: border-color 0.18s ease, color 0.18s ease, background 0.18s ease;
}

.nsa-bot-close:hover,
.nsa-bot-close:focus-visible {
  border-color: #38bdf8;
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.12);
  outline: none;
}

/* Messages & Interactive Flow */
.nsa-bot-messages {
  flex: 1 1 auto;
  overflow-y: auto;
  scroll-behavior: smooth;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  scrollbar-width: thin;
  scrollbar-color: #1e293b transparent;
}

.nsa-bot-messages::-webkit-scrollbar {
  width: 6px;
}

.nsa-bot-messages::-webkit-scrollbar-thumb {
  background: #1e293b;
  border-radius: 999px;
}

.nsa-bot-msg {
  max-width: 92%;
  padding: 10px 13px;
  border-radius: 12px;
  font-size: 13px;
  line-height: 1.55;
  word-wrap: break-word;
}

.nsa-bot-msg-bot {
  align-self: flex-start;
  background: #0b0f19;
  border: 1px solid #1e293b;
  border-left: 3px solid #38bdf8;
  color: #E2E8F0;
  border-top-left-radius: 4px;
}

.nsa-bot-msg-user {
  align-self: flex-end;
  background: rgba(2, 132, 199, 0.22);
  border: 1px solid rgba(56, 189, 248, 0.42);
  color: #F0F9FF;
  font-weight: 500;
  border-top-right-radius: 4px;
}

/* Quick Replies / Question Chips Block */
.nsa-bot-chips-block {
  margin-top: 2px;
  padding-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.nsa-bot-chips-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #94A3B8;
  margin-bottom: 2px;
}

.nsa-bot-chips-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.nsa-bot-chip {
  width: 100%;
  text-align: left;
  padding: 9px 12px;
  border-radius: 10px;
  background: #0b0f19;
  border: 1px solid #1e293b;
  color: #E2E8F0;
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 500;
  line-height: 1.4;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease;
}

.nsa-bot-chip:hover,
.nsa-bot-chip:focus-visible {
  border-color: #38bdf8;
  background: rgba(56, 189, 248, 0.10);
  color: #38bdf8;
  outline: none;
}

.nsa-bot-chip-arrow {
  color: #38bdf8;
  font-size: 13px;
  flex-shrink: 0;
  opacity: 0.85;
}

/* Footer Navigation Bar */
.nsa-bot-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 14px;
  background: #0b0f19;
  border-top: 1px solid #1e293b;
  flex-shrink: 0;
}

.nsa-bot-footer-btn {
  padding: 6px 11px;
  border-radius: 8px;
  border: 1px solid #1e293b;
  background: rgba(255, 255, 255, 0.03);
  color: #CBD5E1;
  font-family: inherit;
  font-size: 11.5px;
  font-weight: 500;
  cursor: pointer;
  transition: border-color 0.18s ease, color 0.18s ease, background 0.18s ease;
}

.nsa-bot-footer-btn:hover,
.nsa-bot-footer-btn:focus-visible {
  border-color: #38bdf8;
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.10);
  outline: none;
}

.nsa-bot-footer-status {
  font-size: 11px;
  color: #64748B;
  white-space: nowrap;
}

/* Mobile Responsive Adjustments */
@media (max-width: 640px) {
  .nsa-bot-wrapper {
    bottom: 16px;
    right: 16px;
  }

  .nsa-bot-window {
    width: calc(100vw - 32px);
    max-width: calc(100vw - 32px);
    max-height: min(520px, calc(100vh - 88px));
  }

  .nsa-bot-launcher {
    min-height: 44px;
    padding: 9px 15px;
    font-size: 13px;
  }
}

@media print {
  .nsa-bot-wrapper {
    display: none !important;
  }
}
"""


NSA_CHATBOT_JS = """/* NewSpace Analytics — Mini Chatbot Q&A para Leigos (100% Client-Side Assisted Q&A) */
(() => {
  'use strict';

  const NSA_CHATBOT_KNOWLEDGE = [
    {
      id: "origem-dados",
      question: "De onde vêm os dados deste painel?",
      answer: "O NewSpace Analytics utiliza mais de 122 mil registros de dados 100% reais e auditados de agências oficiais: NASA (bases JPL SBDB & NHATS), USSPACECOM NORAD e Harvard-Smithsonian. Todo esse volume é processado em alta velocidade pelo motor analítico Polars."
    },
    {
      id: "delta-v",
      question: "O que é Δv (Viabilidade Astrodinâmica)?",
      answer: "De forma simples, o Δv (Delta-v) mede o 'esforço de empurrão' ou a quantidade de combustível necessária para uma nave sair da Terra e alcançar o asteroide. Quanto menor o Δv e o tempo de viagem, mais barata e viável é a missão."
    },
    {
      id: "incerteza-occ",
      question: "O que significa Precisão Navegacional (OCC)?",
      answer: "OCC é o filtro de incerteza orbital. Ele mede o quão bem conhecemos a rota futura de um asteroide. O painel filtra apenas alvos com rotas altamente previsíveis e rastreáveis, evitando o risco de enviar uma sonda e 'perder' o alvo no espaço."
    },
    {
      id: "mineralogia-25m",
      question: "Por que buscar asteroides maiores que 25m?",
      answer: "Asteroides acima de 25 metros possuem volume útil para Aplicação Mineralógica. Neles é possível extrair água e voláteis (que servem para produzir combustível no próprio espaço) ou metais valiosos para uso industrial."
    },
    {
      id: "ranking-topsis",
      question: "Como funciona o Ranking TOPSIS (Top 20)?",
      answer: "O método matemático TOPSIS cruza múltiplos critérios em tempo real para ranquear os 20 melhores asteroides. Ele encontra o equilíbrio ideal entre menor gasto de energia (fácil de chegar) e maior volume de recursos (alto retorno)."
    },
    {
      id: "era-newspace",
      question: "O que é a Era NewSpace Exponencial (2015–2026)?",
      answer: "Na seção de Evolução da Era Espacial, mostramos como o setor mudou: antes dominado apenas por governos (era estatal clássica), entre 2015 e 2026 houve um salto exponencial no lançamento de satélites e objetos orbitais liderado por empresas privadas."
    },
    {
      id: "risco-atuarial",
      question: "Como funciona a Análise de Risco Atuarial e os Ratings?",
      answer: "Funciona como uma análise de seguro e investimento para missões espaciais. A matriz cruza um alvo (como o asteroide Apophis) com um foguete lançador (como o Falcon 9 ou Chang Zheng) e calcula uma nota de risco operacional (Rating A, BB, B, etc.) baseada no histórico real de sucesso."
    }
  ];

  const WELCOME_MESSAGE =
    "Olá! Sou o guia rápido do NewSpace Analytics. Clique em qualquer pergunta abaixo para entender os conceitos técnicos do painel de forma simples:";

  const askedIds = new Set();
  let showAllQuestions = false;
  let isOpen = false;
  let isInitialized = false;

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function getAvailableQuestions() {
    if (showAllQuestions || askedIds.size === 0 || askedIds.size >= NSA_CHATBOT_KNOWLEDGE.length) {
      return NSA_CHATBOT_KNOWLEDGE.slice();
    }
    return NSA_CHATBOT_KNOWLEDGE.filter(function (item) {
      return !askedIds.has(item.id);
    });
  }

  function renderChipsBlock(messagesEl) {
    if (!messagesEl) return;

    const existingBlock = document.getElementById('nsa-bot-chips-block');
    if (existingBlock && existingBlock.parentNode) {
      existingBlock.parentNode.removeChild(existingBlock);
    }

    const available = getAvailableQuestions();
    const allAnswered = askedIds.size >= NSA_CHATBOT_KNOWLEDGE.length;

    let labelText = 'Perguntas principais (clique para traduzir):';
    if (showAllQuestions && askedIds.size > 0) {
      labelText = 'Todas as perguntas do guia:';
    } else if (allAnswered) {
      labelText = 'Todos os tópicos explorados — clique para rever:';
    } else if (askedIds.size > 0) {
      labelText = 'Perguntas restantes (' + available.length + '):';
    }

    const block = document.createElement('div');
    block.className = 'nsa-bot-chips-block';
    block.id = 'nsa-bot-chips-block';

    const chipsHtml = available
      .map(function (item) {
        return (
          '<button type="button" class="nsa-bot-chip" data-question-id="' +
          escapeHtml(item.id) +
          '">' +
          '<span>' +
          escapeHtml(item.question) +
          '</span>' +
          '<span class="nsa-bot-chip-arrow" aria-hidden="true">&rarr;</span>' +
          '</button>'
        );
      })
      .join('');

    block.innerHTML =
      '<div class="nsa-bot-chips-label">' +
      escapeHtml(labelText) +
      '</div>' +
      '<div class="nsa-bot-chips-list">' +
      chipsHtml +
      '</div>';

    messagesEl.appendChild(block);

    const chipButtons = block.querySelectorAll('.nsa-bot-chip');
    chipButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        const qId = btn.getAttribute('data-question-id');
        askQuestionById(qId);
      });
    });
  }

  function appendMessage(messagesEl, role, text) {
    const msg = document.createElement('div');
    msg.className = 'nsa-bot-msg ' + (role === 'user' ? 'nsa-bot-msg-user' : 'nsa-bot-msg-bot');
    msg.textContent = text;
    messagesEl.appendChild(msg);
    return msg;
  }

  function renderInitialConversation() {
    const messagesEl = document.getElementById('nsa-bot-messages');
    if (!messagesEl) return;
    messagesEl.innerHTML = '';
    askedIds.clear();
    showAllQuestions = false;
    appendMessage(messagesEl, 'bot', WELCOME_MESSAGE);
    renderChipsBlock(messagesEl);
  }

  function askQuestionById(questionId) {
    const item = NSA_CHATBOT_KNOWLEDGE.find(function (entry) {
      return entry.id === questionId;
    });
    if (!item) return null;

    const messagesEl = document.getElementById('nsa-bot-messages');
    if (!messagesEl) return null;

    const existingBlock = document.getElementById('nsa-bot-chips-block');
    if (existingBlock && existingBlock.parentNode) {
      existingBlock.parentNode.removeChild(existingBlock);
    }

    askedIds.add(item.id);
    showAllQuestions = false;

    const userMsgEl = appendMessage(messagesEl, 'user', item.question);
    appendMessage(messagesEl, 'bot', item.answer);
    renderChipsBlock(messagesEl);

    if (typeof userMsgEl.scrollIntoView === 'function') {
      userMsgEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    return item;
  }

  function showAllTopics() {
    const messagesEl = document.getElementById('nsa-bot-messages');
    if (!messagesEl) return;
    showAllQuestions = true;
    renderChipsBlock(messagesEl);
    const block = document.getElementById('nsa-bot-chips-block');
    if (block && typeof block.scrollIntoView === 'function') {
      block.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  function setWindowOpen(nextState) {
    isOpen = Boolean(nextState);
    const winEl = document.getElementById('nsa-bot-window');
    const launcherEl = document.getElementById('nsa-bot-launcher');
    if (winEl) {
      if (isOpen) {
        winEl.classList.add('nsa-bot-open');
      } else {
        winEl.classList.remove('nsa-bot-open');
      }
    }
    if (launcherEl) {
      launcherEl.setAttribute('aria-expanded', String(isOpen));
    }
  }

  function toggleWindow() {
    setWindowOpen(!isOpen);
  }

  function initChatbot() {
    if (typeof document === 'undefined' || !document.body) return;
    if (document.getElementById('nsa-bot-wrapper')) {
      isInitialized = true;
      return;
    }

    const wrapper = document.createElement('aside');
    wrapper.className = 'nsa-bot-wrapper';
    wrapper.id = 'nsa-bot-wrapper';
    wrapper.setAttribute('aria-label', 'Assistente de Perguntas e Respostas NewSpace');

    wrapper.innerHTML =
      '<div class="nsa-bot-window" id="nsa-bot-window" role="dialog" aria-modal="false" aria-labelledby="nsa-bot-title">' +
        '<div class="nsa-bot-header">' +
          '<div class="nsa-bot-header-info">' +
            '<h2 class="nsa-bot-title" id="nsa-bot-title">Assistente NewSpace (Guia para Leigos)</h2>' +
            '<p class="nsa-bot-subtitle">Tradução rápida das métricas do painel</p>' +
          '</div>' +
          '<button type="button" class="nsa-bot-close" id="nsa-bot-close" aria-label="Fechar assistente" title="Fechar">&times;</button>' +
        '</div>' +
        '<div class="nsa-bot-messages" id="nsa-bot-messages" aria-live="polite"></div>' +
        '<div class="nsa-bot-footer">' +
          '<button type="button" class="nsa-bot-footer-btn" id="nsa-bot-show-all">Ver todas as perguntas</button>' +
          '<button type="button" class="nsa-bot-footer-btn" id="nsa-bot-reset">Reiniciar guia</button>' +
        '</div>' +
      '</div>' +
      '<button type="button" class="nsa-bot-launcher" id="nsa-bot-launcher" aria-expanded="false" aria-controls="nsa-bot-window">' +
        '<span class="nsa-bot-launcher-dot" aria-hidden="true"></span>' +
        '<span class="nsa-bot-launcher-label">Guia Rápido • Dúvidas?</span>' +
      '</button>';

    document.body.appendChild(wrapper);
    renderInitialConversation();

    const launcherBtn = document.getElementById('nsa-bot-launcher');
    const closeBtn = document.getElementById('nsa-bot-close');
    const showAllBtn = document.getElementById('nsa-bot-show-all');
    const resetBtn = document.getElementById('nsa-bot-reset');

    if (launcherBtn) {
      launcherBtn.addEventListener('click', toggleWindow);
    }
    if (closeBtn) {
      closeBtn.addEventListener('click', function () {
        setWindowOpen(false);
      });
    }
    if (showAllBtn) {
      showAllBtn.addEventListener('click', showAllTopics);
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', renderInitialConversation);
    }

    document.addEventListener('keydown', function (ev) {
      if (ev && ev.key === 'Escape' && isOpen) {
        setWindowOpen(false);
      }
    });

    isInitialized = true;
  }

  const globalRef = typeof window !== 'undefined' ? window : globalThis;
  globalRef.NSA_CHATBOT_KNOWLEDGE = NSA_CHATBOT_KNOWLEDGE;
  globalRef.NSAChatbot = {
    KNOWLEDGE: NSA_CHATBOT_KNOWLEDGE,
    WELCOME_MESSAGE: WELCOME_MESSAGE,
    init: initChatbot,
    open: function () { setWindowOpen(true); },
    close: function () { setWindowOpen(false); },
    toggle: toggleWindow,
    ask: askQuestionById,
    showAll: showAllTopics,
    reset: renderInitialConversation,
    getAvailableQuestions: getAvailableQuestions,
    isInitialized: function () { return isInitialized; }
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initChatbot);
    } else {
      initChatbot();
    }
  }
})();
"""
